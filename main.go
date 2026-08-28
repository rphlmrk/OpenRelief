package main

import (
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"
	"net"
	"net/http"
	"path/filepath"
	"strconv"
	"sync"

	"github.com/disintegration/imaging"
	webview "github.com/webview/webview_go"
	ort "github.com/yalue/onnxruntime_go"
)

//go:embed index.html
var htmlContent []byte

const ModelInputSize = 518
const ShadowInputSize = 512 // Most lightweight U-Nets use 512x512

type Engine struct {
	session      *ort.AdvancedSession
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[float32]
	mu           sync.Mutex
	cache        map[string]*image.Gray16
}

var globalDepthEngine *Engine
var globalShadowEngines = make(map[string]*Engine) // Stores all shadow models

func main() {
	dllPath := "onnxruntime.dll"
	depthModelPath := filepath.Join("models", "anything v2.onnx")

	fmt.Println("🪙 Starting OpenRelief Studio (Tiled Depth & Multi-Scale Engine)...")

	// Initialize ONNX Environment
	absDllPath, _ := filepath.Abs(dllPath)
	ort.SetSharedLibraryPath(absDllPath)
	err := ort.InitializeEnvironment()
	if err != nil {
		fmt.Printf("ONNX Init Error: %v\n", err)
		return
	}
	defer ort.DestroyEnvironment()

	// Load Depth Engine
	globalDepthEngine, err = initEngine(depthModelPath, ModelInputSize, ModelInputSize, 1) // Depth outputs 1 channel
	if err != nil {
		fmt.Printf("Depth Engine Init Error: %v\n", err)
		return
	}
	defer globalDepthEngine.session.Destroy()

	// Preload all 3 Shadow Removal models
	shadowModels := []string{"docshadow_sd7k.onnx", "docshadow_jung.onnx", "docshadow_kligler.onnx"}
	for _, modelName := range shadowModels {
		fullPath := filepath.Join("models", modelName)
		eng, err := initEngine(fullPath, ShadowInputSize, ShadowInputSize, 3)
		if err != nil {
			fmt.Printf("Notice: Could not load %s (skipped)\n", modelName)
		} else {
			globalShadowEngines[modelName] = eng
			defer eng.session.Destroy()
			fmt.Printf("Loaded Shadow Model: %s\n", modelName)
		}
	}

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		panic(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	appURL := fmt.Sprintf("http://127.0.0.1:%d", port)

	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		w.Write(htmlContent)
	})
	mux.HandleFunc("/api/process", handleProcess)
	mux.HandleFunc("/api/download", handleDownload)

	go http.Serve(listener, mux)

	w := webview.New(false)
	defer w.Destroy()
	w.SetTitle("OpenRelief Studio - Tiled Multi-Scale Depth Studio")
	w.SetSize(1380, 920, webview.HintNone)
	w.Navigate(appURL)
	w.Run()
}

// Updated Init Engine to handle different output channels (1 for Depth, 3 for Shadow RGB)
func initEngine(modelPath string, width int, height int, outChannels int) (*Engine, error) {
	absModelPath, err := filepath.Abs(modelPath)
	if err != nil {
		return nil, err
	}

	// Auto-detect the exact tensor names from the ONNX file metadata
	inInfo, outInfo, err := ort.GetInputOutputInfo(absModelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to get metadata for %s: %w", modelPath, err)
	}
	inputName := inInfo[0].Name
	outputName := outInfo[0].Name

	inputShape := ort.NewShape(1, 3, int64(width), int64(height))
	var outputShape ort.Shape
	if outChannels == 1 {
		outputShape = ort.NewShape(1, int64(width), int64(height))
	} else {
		outputShape = ort.NewShape(1, int64(outChannels), int64(width), int64(height))
	}

	inTensor, _ := ort.NewTensor(inputShape, make([]float32, 1*3*width*height))
	outTensor, _ := ort.NewEmptyTensor[float32](outputShape)

	session, err := ort.NewAdvancedSession(absModelPath,
		[]string{inputName}, []string{outputName},
		[]ort.ArbitraryTensor{inTensor}, []ort.ArbitraryTensor{outTensor}, nil)
	if err != nil {
		return nil, err
	}

	return &Engine{
		session:      session,
		inputTensor:  inTensor,
		outputTensor: outTensor,
		cache:        make(map[string]*image.Gray16),
	}, nil
}

// NEW: Native Go ONNX Shadow Removal
func runShadowRemoval(img image.Image, engine *Engine) image.Image {
	if engine == nil {
		fmt.Println("Selected Shadow Engine not available, skipping...")
		return img
	}

	engine.mu.Lock()
	defer engine.mu.Unlock()

	resized := imaging.Resize(img, ShadowInputSize, ShadowInputSize, imaging.Lanczos)
	tensorData := engine.inputTensor.GetData()

	// Normalize image to 0.0 -> 1.0 (Standard U-Net input)
	for y := 0; y < ShadowInputSize; y++ {
		for x := 0; x < ShadowInputSize; x++ {
			rCol, gCol, bCol, _ := resized.At(x, y).RGBA()
			tensorData[0*ShadowInputSize*ShadowInputSize+y*ShadowInputSize+x] = float32(rCol>>8) / 255.0
			tensorData[1*ShadowInputSize*ShadowInputSize+y*ShadowInputSize+x] = float32(gCol>>8) / 255.0
			tensorData[2*ShadowInputSize*ShadowInputSize+y*ShadowInputSize+x] = float32(bCol>>8) / 255.0
		}
	}

	// Run AI Inference
	if err := engine.session.Run(); err != nil {
		fmt.Println("Shadow ONNX Run Error:", err)
		return img
	}

	rawOut := engine.outputTensor.GetData()
	outImg := image.NewRGBA(image.Rect(0, 0, ShadowInputSize, ShadowInputSize))

	// Reconstruct RGB Image from AI Output
	for y := 0; y < ShadowInputSize; y++ {
		for x := 0; x < ShadowInputSize; x++ {
			r := uint8(math.Max(0, math.Min(255, float64(rawOut[0*ShadowInputSize*ShadowInputSize+y*ShadowInputSize+x]*255.0))))
			g := uint8(math.Max(0, math.Min(255, float64(rawOut[1*ShadowInputSize*ShadowInputSize+y*ShadowInputSize+x]*255.0))))
			b := uint8(math.Max(0, math.Min(255, float64(rawOut[2*ShadowInputSize*ShadowInputSize+y*ShadowInputSize+x]*255.0))))
			outImg.SetRGBA(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}

	// Resize back to original dimensions to feed into the Depth model seamlessly
	return imaging.Resize(outImg, img.Bounds().Dx(), img.Bounds().Dy(), imaging.Lanczos)
}

// Helper: Run inference on a single image patch (Depth)
func runInference(img image.Image) [][]float64 {
	resized := imaging.Resize(img, ModelInputSize, ModelInputSize, imaging.Lanczos)
	tensorData := globalDepthEngine.inputTensor.GetData()

	mean := []float32{0.485, 0.456, 0.406}
	std := []float32{0.229, 0.224, 0.225}

	for y := 0; y < ModelInputSize; y++ {
		for x := 0; x < ModelInputSize; x++ {
			rCol, gCol, bCol, _ := resized.At(x, y).RGBA()
			tensorData[0*ModelInputSize*ModelInputSize+y*ModelInputSize+x] = (float32(rCol>>8)/255.0 - mean[0]) / std[0]
			tensorData[1*ModelInputSize*ModelInputSize+y*ModelInputSize+x] = (float32(gCol>>8)/255.0 - mean[1]) / std[1]
			tensorData[2*ModelInputSize*ModelInputSize+y*ModelInputSize+x] = (float32(bCol>>8)/255.0 - mean[2]) / std[2]
		}
	}

	if err := globalDepthEngine.session.Run(); err != nil {
		return nil
	}

	rawDepth := globalDepthEngine.outputTensor.GetData()
	minVal, maxVal := float32(math.MaxFloat32), float32(-math.MaxFloat32)
	for _, v := range rawDepth {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	diff := float64(maxVal - minVal)
	if diff == 0 {
		diff = 1.0
	}

	out := make([][]float64, ModelInputSize)
	for y := 0; y < ModelInputSize; y++ {
		out[y] = make([]float64, ModelInputSize)
		for x := 0; x < ModelInputSize; x++ {
			val := float64(rawDepth[y*ModelInputSize+x]-minVal) / diff
			out[y][x] = val
		}
	}
	return out
}

func calcStats(data [][]float64, startX, startY, w, h int) (mean, std float64) {
	var sum float64
	count := float64(w * h)
	for y := startY; y < startY+h; y++ {
		for x := startX; x < startX+w; x++ {
			sum += data[y][x]
		}
	}
	mean = sum / count

	var sumSq float64
	for y := startY; y < startY+h; y++ {
		for x := startX; x < startX+w; x++ {
			d := data[y][x] - mean
			sumSq += d * d
		}
	}
	std = math.Sqrt(sumSq / count)
	if std < 1e-5 {
		std = 1.0
	}
	return mean, std
}

func handleProcess(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	defer func() {
		if rec := recover(); rec != nil {
			fmt.Printf("Server Panic: %v\n", rec)
			json.NewEncoder(w).Encode(map[string]string{"error": fmt.Sprintf("Processing error: %v", rec)})
		}
	}()

	if err := r.ParseMultipartForm(64 << 20); err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}

	file, _, err := r.FormFile("image")
	if err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to read image file: " + err.Error()})
		return
	}
	defer file.Close()

	hpStrength, _ := strconv.ParseFloat(r.FormValue("hpStrength"), 64)
	reliefFlatten, _ := strconv.ParseFloat(r.FormValue("reliefFlatten"), 64)
	bodyInflation, _ := strconv.ParseFloat(r.FormValue("bodyInflation"), 64)
	useTiling := r.FormValue("useTiling") == "true"
	isCoinMask := r.FormValue("coinMask") == "true"
	isInvert := r.FormValue("invert") == "true"
	removeShadows := r.FormValue("removeShadows") == "true"

	srcImg, err := imaging.Decode(file)
	if err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to decode image: " + err.Error()})
		return
	}

	var delightBase64 string
	// --- Go-Native AI Shadow Removal with Selected Model ---
	if removeShadows {
		shadowModelName := r.FormValue("shadowModel")
		if shadowModelName == "" {
			shadowModelName = "docshadow_sd7k.onnx"
		}
		selectedEngine := globalShadowEngines[shadowModelName]

		fmt.Printf("[*] Running Shadow Removal with model: %s...\n", shadowModelName)
		srcImg = runShadowRemoval(srcImg, selectedEngine)
		fmt.Println("[+] Shadows removed! Passing to Depth Engine.")

		// Generate preview of the shadow-removed image
		thumbDelight := imaging.Resize(srcImg, 450, 0, imaging.Lanczos)
		var bufDelight []byte
		wDelight := &byteWriter{buf: &bufDelight}
		jpeg.Encode(wDelight, thumbDelight, &jpeg.Options{Quality: 85})
		delightBase64 = "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(bufDelight)
	}

	origW, origH := srcImg.Bounds().Dx(), srcImg.Bounds().Dy()

	globalDepthEngine.mu.Lock()
	defer globalDepthEngine.mu.Unlock()

	// 1. GLOBAL BASE DEPTH INFERENCE
	baseRaw := runInference(srcImg)
	if baseRaw == nil {
		json.NewEncoder(w).Encode(map[string]string{"error": "Inference failed"})
		return
	}

	// Rescale global depth to original dimensions
	baseDepth := make([][]float64, origH)
	for y := 0; y < origH; y++ {
		baseDepth[y] = make([]float64, origW)
		srcY := int(float64(y) / float64(origH) * float64(ModelInputSize-1))
		for x := 0; x < origW; x++ {
			srcX := int(float64(x) / float64(origW) * float64(ModelInputSize-1))
			baseDepth[y][x] = baseRaw[srcY][srcX]
		}
	}

	// 2. HIERARCHICAL 2x2 OVERLAPPING TILING ENGINE
	compiledTiles := make([][]float64, origH)
	tileWeights := make([][]float64, origH)
	for y := 0; y < origH; y++ {
		compiledTiles[y] = make([]float64, origW)
		tileWeights[y] = make([]float64, origW)
	}

	if useTiling {
		numX, numY := 2, 2
		tileW := origW / numX
		tileH := origH / numY

		xCoords := []int{0, origW - tileW}
		if numX > 1 {
			xCoords = append(xCoords, tileW/2)
		}
		yCoords := []int{0, origH - tileH}
		if numY > 1 {
			yCoords = append(yCoords, tileH/2)
		}

		for _, ty := range yCoords {
			for _, tx := range xCoords {
				tileRect := image.Rect(tx, ty, tx+tileW, ty+tileH)
				tileSubImg := imaging.Crop(srcImg, tileRect)

				tileRaw := runInference(tileSubImg)
				if tileRaw == nil {
					continue
				}

				meanLow, stdLow := calcStats(baseDepth, tx, ty, tileW, tileH)
				meanTile, stdTile := calcStats(tileRaw, 0, 0, ModelInputSize, ModelInputSize)

				for y := 0; y < tileH; y++ {
					imgY := ty + y
					if imgY >= origH {
						continue
					}
					srcY := int(float64(y) / float64(tileH) * float64(ModelInputSize-1))

					yVal := 0.998 * math.Pow(math.Cos((math.Abs(float64(tileH)/2.0-float64(y))/float64(tileH))*math.Pi), 2)
					if ty == 0 && y < tileH/2 {
						yVal = 0.998
					} else if ty == origH-tileH && y > tileH/2 {
						yVal = 0.998
					}

					for x := 0; x < tileW; x++ {
						imgX := tx + x
						if imgX >= origW {
							continue
						}
						srcX := int(float64(x) / float64(tileW) * float64(ModelInputSize-1))

						xVal := 0.998 * math.Pow(math.Cos((math.Abs(float64(tileW)/2.0-float64(x))/float64(tileW))*math.Pi), 2)
						if tx == 0 && x < tileW/2 {
							xVal = 0.998
						} else if tx == origW-tileW && x > tileW/2 {
							xVal = 0.998
						}

						weight := xVal * yVal
						rawVal := tileRaw[srcY][srcX]

						scaledDepth := meanLow + stdLow*((rawVal-meanTile)/stdTile)
						compiledTiles[imgY][imgX] += weight * scaledDepth
						tileWeights[imgY][imgX] += weight
					}
				}
			}
		}

		for y := 0; y < origH; y++ {
			for x := 0; x < origW; x++ {
				if tileWeights[y][x] > 0 {
					compiledTiles[y][x] /= tileWeights[y][x]
				} else {
					compiledTiles[y][x] = baseDepth[y][x]
				}
			}
		}
	} else {
		for y := 0; y < origH; y++ {
			copy(compiledTiles[y], baseDepth[y])
		}
	}

	// 3. GAUSSIAN EDGE DIFFERENCE GUIDANCE
	grayImg := imaging.Grayscale(srcImg)
	blurred20 := imaging.Blur(grayImg, 5.0)
	blurred40 := imaging.Blur(blurred20, 8.0)

	hpVisualizer := image.NewGray(image.Rect(0, 0, origW, origH))
	baseDepthVisualizer := image.NewGray16(image.Rect(0, 0, origW, origH))
	result16Bit := image.NewGray16(image.Rect(0, 0, origW, origH))

	centerX, centerY := float64(origW)/2.0, float64(origH)/2.0
	radius := math.Min(centerX, centerY) - 4.0

	for y := 0; y < origH; y++ {
		for x := 0; x < origW; x++ {
			dVal := compiledTiles[y][x]

			gCol, _, _, _ := grayImg.At(x, y).RGBA()
			gVal := float64(gCol) / 65535.0

			bCol20, _, _, _ := blurred20.At(x, y).RGBA()
			bVal20 := float64(bCol20) / 65535.0

			bCol40, _, _, _ := blurred40.At(x, y).RGBA()
			bVal40 := float64(bCol40) / 65535.0

			diff := (bVal20 - gVal)
			diffMask := math.Min(0.999, math.Max(0.0, (bVal40-gVal)*hpStrength*4.0))

			hpVal := 0.5 + (gVal - bVal20)
			hpVisualizer.SetGray(x, y, color.Gray{Y: uint8(math.Min(255.0, math.Max(0.0, hpVal*255.0)))})

			if reliefFlatten > 0 {
				dVal = math.Pow(math.Max(0.0, dVal), 1.0-(reliefFlatten*0.5))
			}

			if bodyInflation > 0 {
				inflate := math.Sin(math.Max(0.0, math.Min(1.0, dVal)) * math.Pi * 0.5)
				dVal = dVal*(1.0-bodyInflation*0.35) + inflate*(bodyInflation*0.35)
			}

			blended := diffMask*dVal + (1.0-diffMask)*(dVal+diff*hpStrength)

			baseVal := dVal
			if isCoinMask {
				dist := math.Hypot(float64(x)-centerX, float64(y)-centerY)
				if dist > radius {
					blended = 0.0
					baseVal = 0.0
				} else if dist > radius-3.5 {
					blended = 0.92
					baseVal = 0.92
				}
			}

			if isInvert {
				blended = 1.0 - blended
				baseVal = 1.0 - baseVal
			}

			baseDepthVisualizer.SetGray16(x, y, color.Gray16{Y: uint16(math.Min(1.0, math.Max(0.0, baseVal)) * 65535.0)})
			result16Bit.SetGray16(x, y, color.Gray16{Y: uint16(math.Min(1.0, math.Max(0.0, blended)) * 65535.0)})
		}
	}

	id := "current"
	globalDepthEngine.cache[id] = result16Bit

	thumbBase := imaging.Resize(baseDepthVisualizer, 450, 0, imaging.Lanczos)
	var bufBase []byte
	wBase := &byteWriter{buf: &bufBase}
	jpeg.Encode(wBase, thumbBase, &jpeg.Options{Quality: 85})

	thumbHP := imaging.Resize(hpVisualizer, 450, 0, imaging.Lanczos)
	var bufHP []byte
	wHP := &byteWriter{buf: &bufHP}
	jpeg.Encode(wHP, thumbHP, &jpeg.Options{Quality: 85})

	thumbDepth := imaging.Resize(result16Bit, 450, 0, imaging.Lanczos)
	var bufDepth []byte
	wDepth := &byteWriter{buf: &bufDepth}
	jpeg.Encode(wDepth, thumbDepth, &jpeg.Options{Quality: 85})

	json.NewEncoder(w).Encode(map[string]string{
		"id":             id,
		"delightPreview": delightBase64,
		"basePreview":    "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(bufBase),
		"hpPreview":      "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(bufHP),
		"depthPreview":   "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(bufDepth),
	})
}

func handleDownload(w http.ResponseWriter, r *http.Request) {
	globalDepthEngine.mu.Lock()
	defer globalDepthEngine.mu.Unlock()

	id := r.URL.Query().Get("id")
	img, ok := globalDepthEngine.cache[id]
	if !ok {
		http.Error(w, "Not found", 404)
		return
	}

	w.Header().Set("Content-Type", "image/png")
	w.Header().Set("Content-Disposition", "attachment; filename=openrelief_16bit.png")
	png.Encode(w, img)
}

type byteWriter struct{ buf *[]byte }

func (b *byteWriter) Write(p []byte) (n int, err error) {
	*b.buf = append(*b.buf, p...)
	return len(p), nil
}
