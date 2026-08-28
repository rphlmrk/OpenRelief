package main

import (
	"bytes"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"
	"math"
	"mime/multipart"
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

type Engine struct {
	session      *ort.AdvancedSession
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[float32]
	mu           sync.Mutex
	cache        map[string]*image.Gray16
}

var globalEngine *Engine

func main() {
	dllPath := "onnxruntime.dll"
	modelPath := "model.onnx"

	fmt.Println("🪙 Starting OpenRelief Studio (Tiled Depth & Multi-Scale Engine)...")
	eng, err := initEngine(dllPath, modelPath)
	if err != nil {
		fmt.Printf("Initialization Error: %v\n", err)
		return
	}
	defer eng.session.Destroy()
	globalEngine = eng

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

func initEngine(dllPath, modelPath string) (*Engine, error) {
	absDllPath, err := filepath.Abs(dllPath)
	if err != nil {
		return nil, err
	}
	absModelPath, err := filepath.Abs(modelPath)
	if err != nil {
		return nil, err
	}

	ort.SetSharedLibraryPath(absDllPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, err
	}

	inputShape := ort.NewShape(1, 3, ModelInputSize, ModelInputSize)
	outputShape := ort.NewShape(1, ModelInputSize, ModelInputSize)

	inTensor, _ := ort.NewTensor(inputShape, make([]float32, 1*3*ModelInputSize*ModelInputSize))
	outTensor, _ := ort.NewEmptyTensor[float32](outputShape)

	session, err := ort.NewAdvancedSession(absModelPath,
		[]string{"pixel_values"}, []string{"predicted_depth"},
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

// Helper: Run inference on a single image patch
func runInference(img image.Image) [][]float64 {
	resized := imaging.Resize(img, ModelInputSize, ModelInputSize, imaging.Lanczos)
	tensorData := globalEngine.inputTensor.GetData()

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

	if err := globalEngine.session.Run(); err != nil {
		return nil
	}

	rawDepth := globalEngine.outputTensor.GetData()
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

// Calculate mean & standard deviation of a region
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

	globalEngine.mu.Lock()
	defer globalEngine.mu.Unlock()

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
	removeShadows := r.FormValue("removeShadows") == "true" // New Parameter

	var srcImg image.Image

	// --- NEW: Call Python Server if Remove Shadows is checked ---
	if removeShadows {
		fmt.Println("Sending image to Python server for Shadow Removal...")

		// Copy uploaded file to a buffer
		var reqBody bytes.Buffer
		multipartWriter := multipart.NewWriter(&reqBody)
		fileWriter, _ := multipartWriter.CreateFormFile("image", "upload.png")
		file.Seek(0, 0)
		io.Copy(fileWriter, file)
		multipartWriter.Close()

		// Send to Python Flask server on port 5000
		resp, err := http.Post("http://127.0.0.1:5000/delight", multipartWriter.FormDataContentType(), &reqBody)
		if err != nil || resp.StatusCode != 200 {
			json.NewEncoder(w).Encode(map[string]string{"error": "Failed to connect to Python Delighting Server. Is it running?"})
			return
		}
		defer resp.Body.Close()

		// Decode the returned shadow-free image
		srcImg, err = imaging.Decode(resp.Body)
		if err != nil {
			json.NewEncoder(w).Encode(map[string]string{"error": "Failed to decode Python response."})
			return
		}
		fmt.Println("Shadow removal complete! Continuing with Depth processing...")
	} else {
		// Normal flow: just decode the uploaded file directly
		file.Seek(0, 0)
		srcImg, err = imaging.Decode(file)
		if err != nil {
			json.NewEncoder(w).Encode(map[string]string{"error": "Failed to decode image: " + err.Error()})
			return
		}
	}

	origW, origH := srcImg.Bounds().Dx(), srcImg.Bounds().Dy()

	// =========================================================================
	// 1. GLOBAL BASE DEPTH INFERENCE
	// =========================================================================
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

	// =========================================================================
	// 2. HIERARCHICAL 2x2 OVERLAPPING TILING ENGINE (BillFSmith Method)
	// =========================================================================
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

				// Local statistics
				meanLow, stdLow := calcStats(baseDepth, tx, ty, tileW, tileH)
				meanTile, stdTile := calcStats(tileRaw, 0, 0, ModelInputSize, ModelInputSize)

				for y := 0; y < tileH; y++ {
					imgY := ty + y
					if imgY >= origH {
						continue
					}
					srcY := int(float64(y) / float64(tileH) * float64(ModelInputSize-1))

					// 2D Cosine-squared windowing
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

						// Scale tile depth to local base depth mean & stddev
						scaledDepth := meanLow + stdLow*((rawVal-meanTile)/stdTile)

						compiledTiles[imgY][imgX] += weight * scaledDepth
						tileWeights[imgY][imgX] += weight
					}
				}
			}
		}

		// Normalize tile accumulation
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
		// Single-pass fallback
		for y := 0; y < origH; y++ {
			copy(compiledTiles[y], baseDepth[y])
		}
	}

	// =========================================================================
	// 3. GAUSSIAN EDGE DIFFERENCE GUIDANCE (BillFSmith Step 6)
	// =========================================================================
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

			// Original Grayscale & Blur pixels
			gCol, _, _, _ := grayImg.At(x, y).RGBA()
			gVal := float64(gCol) / 65535.0

			bCol20, _, _, _ := blurred20.At(x, y).RGBA()
			bVal20 := float64(bCol20) / 65535.0

			bCol40, _, _, _ := blurred40.At(x, y).RGBA()
			bVal40 := float64(bCol40) / 65535.0

			// High Pass difference
			diff := (bVal20 - gVal)
			diffMask := math.Min(0.999, math.Max(0.0, (bVal40-gVal)*hpStrength*4.0))

			// Visualizer: 50% neutral gray preview
			hpVal := 0.5 + (gVal - bVal20)
			hpVisualizer.SetGray(x, y, color.Gray{Y: uint8(math.Min(255.0, math.Max(0.0, hpVal*255.0)))})

			// 1. Perspective Flattening (Z-slope compression - no holes)
			if reliefFlatten > 0 {
				dVal = math.Pow(math.Max(0.0, dVal), 1.0-(reliefFlatten*0.5))
			}

			// 2. Continuous Body Inflation (Smooth anatomical puff - no holes)
			if bodyInflation > 0 {
				inflate := math.Sin(math.Max(0.0, math.Min(1.0, dVal)) * math.Pi * 0.5)
				dVal = dVal*(1.0-bodyInflation*0.35) + inflate*(bodyInflation*0.35)
			}

			// 3. Merge High-Frequency Detail onto Smooth Base Depth
			blended := diffMask*dVal + (1.0-diffMask)*(dVal+diff*hpStrength)

			// Coin Masking (if selected)
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
	globalEngine.cache[id] = result16Bit

	// Generate UI previews
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
		"id":           id,
		"basePreview":  "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(bufBase),
		"hpPreview":    "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(bufHP),
		"depthPreview": "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(bufDepth),
	})
}

func handleDownload(w http.ResponseWriter, r *http.Request) {
	globalEngine.mu.Lock()
	defer globalEngine.mu.Unlock()

	id := r.URL.Query().Get("id")
	img, ok := globalEngine.cache[id]
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
