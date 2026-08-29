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
	"os"
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
const ShadowInputSize = 512
const NafnetInputSize = 512
const U2NetInputSize = 320

type Engine struct {
	session       *ort.AdvancedSession
	inputTensor   *ort.Tensor[float32]
	inputTensorU8 *ort.Tensor[uint8]
	outputTensor  *ort.Tensor[float32]
	isUint8       bool
	isNHWC        bool
	inputWidth    int
	inputHeight   int
	mu            sync.Mutex
	cache         map[string]*image.Gray16
}

var globalDepthEngines = make(map[string]*Engine)
var defaultDepthModel string
var globalShadowEngines = make(map[string]*Engine)
var globalNafnetEngine *Engine
var globalBgrEngine *Engine

func main() {
	dllPath := "onnxruntime.dll"

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

	// Preload all available Depth Anything 3 (DA3) models
	da3Models := []string{"da3_small.onnx", "da3_base.onnx", "da3_metric_large.onnx", "anything v3.onnx"}
	for _, modelName := range da3Models {
		fullPath := filepath.Join("models", modelName)
		if _, statErr := os.Stat(fullPath); statErr == nil {
			eng, err := initEngine(fullPath, ModelInputSize, ModelInputSize, 1)
			if err != nil {
				fmt.Printf("Notice: Could not load %s (%v, skipped)\n", modelName, err)
			} else {
				globalDepthEngines[modelName] = eng
				if defaultDepthModel == "" {
					defaultDepthModel = modelName
				}
				defer eng.session.Destroy()
				fmt.Printf("Loaded DA3 Model: %s\n", modelName)
			}
		}
	}
	if len(globalDepthEngines) == 0 {
		fmt.Println("⚠️ Warning: No DA3 models found in 'models/' folder.")
	}

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

	// Load NAFNet HD Restoration Model
	nafnetPath := filepath.Join("models", "nafnet.onnx")
	globalNafnetEngine, err = initEngine(nafnetPath, NafnetInputSize, NafnetInputSize, 3)
	if err != nil {
		fmt.Printf("Notice: Could not load nafnet.onnx (%v). HD Restoration disabled.\n", err)
	} else {
		defer globalNafnetEngine.session.Destroy()
		fmt.Println("Loaded HD Restoration Model: nafnet.onnx")
	}

	// Load U-2-Net Background Removal Model
	u2netPath := filepath.Join("models", "u2net.onnx")
	globalBgrEngine, err = initEngine(u2netPath, U2NetInputSize, U2NetInputSize, 1)
	if err != nil {
		u2netPath = filepath.Join("models", "u2netp.onnx")
		globalBgrEngine, err = initEngine(u2netPath, U2NetInputSize, U2NetInputSize, 1)
	}
	if err != nil {
		fmt.Printf("Notice: Could not load u2net.onnx (%v). Background Removal disabled.\n", err)
	} else {
		defer globalBgrEngine.session.Destroy()
		fmt.Println("Loaded Background Removal Model: u2net.onnx")
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
	mux.HandleFunc("/api/info", handleInfo)
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

// Dynamic Init Engine that auto-detects uint8 vs float32 and 3D/4D/5D topologies
func initEngine(modelPath string, width int, height int, outChannels int) (*Engine, error) {
	absModelPath, err := filepath.Abs(modelPath)
	if err != nil {
		return nil, err
	}

	inInfo, outInfo, err := ort.GetInputOutputInfo(absModelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to get metadata for %s: %w", modelPath, err)
	}
	if len(inInfo) == 0 || len(outInfo) == 0 {
		return nil, fmt.Errorf("model has no inputs or outputs: %s", modelPath)
	}

	inputName := inInfo[0].Name
	outputName := outInfo[0].Name
	isUint8 := inInfo[0].DataType == ort.TensorElementDataTypeUint8

	// 1. Resolve Input Shapes
	inShapeDims := make([]int64, len(inInfo[0].Dimensions))
	copy(inShapeDims, inInfo[0].Dimensions)

	modelW, modelH := width, height
	isNHWC := false

	if len(inShapeDims) == 4 {
		if inShapeDims[3] == 3 || (inShapeDims[3] == -1 && inShapeDims[1] != 3) {
			// NHWC: [1, H, W, 3]
			isNHWC = true
			for i, d := range inShapeDims {
				if d <= 0 {
					if i == 0 {
						inShapeDims[i] = 1
					}
					if i == 1 {
						inShapeDims[i] = int64(height)
					}
					if i == 2 {
						inShapeDims[i] = int64(width)
					}
					if i == 3 {
						inShapeDims[i] = 3
					}
				}
			}
			modelH = int(inShapeDims[1])
			modelW = int(inShapeDims[2])
		} else {
			// NCHW: [1, 3, H, W]
			for i, d := range inShapeDims {
				if d <= 0 {
					if i == 0 {
						inShapeDims[i] = 1
					}
					if i == 1 {
						inShapeDims[i] = 3
					}
					if i == 2 {
						inShapeDims[i] = int64(height)
					}
					if i == 3 {
						inShapeDims[i] = int64(width)
					}
				}
			}
			modelH = int(inShapeDims[2])
			modelW = int(inShapeDims[3])
		}
	} else if len(inShapeDims) == 5 {
		// DA3 Multi-View: [1, 1, 3, H, W]
		for i, d := range inShapeDims {
			if d <= 0 {
				if i == 0 {
					inShapeDims[i] = 1
				}
				if i == 1 {
					inShapeDims[i] = 1
				}
				if i == 2 {
					inShapeDims[i] = 3
				}
				if i == 3 {
					inShapeDims[i] = int64(height)
				}
				if i == 4 {
					inShapeDims[i] = int64(width)
				}
			}
		}
		modelH = int(inShapeDims[3])
		modelW = int(inShapeDims[4])
	} else {
		inShapeDims = []int64{1, 3, int64(height), int64(width)}
	}

	// 2. Resolve Output Shapes
	outShapeDims := make([]int64, len(outInfo[0].Dimensions))
	copy(outShapeDims, outInfo[0].Dimensions)

	for i, d := range outShapeDims {
		if d <= 0 {
			if i == 0 {
				outShapeDims[i] = 1
			} else if i == 1 && len(outShapeDims) == 4 {
				outShapeDims[i] = int64(outChannels)
			} else if i == len(outShapeDims)-2 {
				outShapeDims[i] = int64(modelH)
			} else if i == len(outShapeDims)-1 {
				outShapeDims[i] = int64(modelW)
			} else {
				outShapeDims[i] = 1
			}
		}
	}
	if len(outShapeDims) == 0 {
		outShapeDims = []int64{1, int64(modelH), int64(modelW)}
	}

	inputShape := ort.NewShape(inShapeDims...)
	outputShape := ort.NewShape(outShapeDims...)

	var totalIn int64 = 1
	for _, dim := range inShapeDims {
		totalIn *= dim
	}

	var inTensor ort.ArbitraryTensor
	var inTensorF32 *ort.Tensor[float32]
	var inTensorU8 *ort.Tensor[uint8]

	if isUint8 {
		inTensorU8, err = ort.NewTensor(inputShape, make([]uint8, totalIn))
		if err != nil {
			return nil, fmt.Errorf("failed to create uint8 input tensor for %s: %w", modelPath, err)
		}
		inTensor = inTensorU8
	} else {
		inTensorF32, err = ort.NewTensor(inputShape, make([]float32, totalIn))
		if err != nil {
			return nil, fmt.Errorf("failed to create float32 input tensor for %s: %w", modelPath, err)
		}
		inTensor = inTensorF32
	}

	outTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inTensor.Destroy()
		return nil, fmt.Errorf("failed to create output tensor for %s: %w", modelPath, err)
	}

	session, err := ort.NewAdvancedSession(absModelPath,
		[]string{inputName}, []string{outputName},
		[]ort.ArbitraryTensor{inTensor}, []ort.ArbitraryTensor{outTensor}, nil)
	if err != nil {
		inTensor.Destroy()
		outTensor.Destroy()
		return nil, fmt.Errorf("failed to create session for %s: %w", modelPath, err)
	}

	return &Engine{
		session:       session,
		inputTensor:   inTensorF32,
		inputTensorU8: inTensorU8,
		outputTensor:  outTensor,
		isUint8:       isUint8,
		isNHWC:        isNHWC,
		inputWidth:    modelW,
		inputHeight:   modelH,
		cache:         make(map[string]*image.Gray16),
	}, nil
}

func runShadowRemoval(img image.Image, engine *Engine) image.Image {
	if engine == nil {
		return img
	}
	engine.mu.Lock()
	defer engine.mu.Unlock()

	resized := imaging.Resize(img, ShadowInputSize, ShadowInputSize, imaging.Lanczos)
	tensorData := engine.inputTensor.GetData()

	for y := 0; y < ShadowInputSize; y++ {
		for x := 0; x < ShadowInputSize; x++ {
			rCol, gCol, bCol, _ := resized.At(x, y).RGBA()
			tensorData[0*ShadowInputSize*ShadowInputSize+y*ShadowInputSize+x] = float32(rCol>>8) / 255.0
			tensorData[1*ShadowInputSize*ShadowInputSize+y*ShadowInputSize+x] = float32(gCol>>8) / 255.0
			tensorData[2*ShadowInputSize*ShadowInputSize+y*ShadowInputSize+x] = float32(bCol>>8) / 255.0
		}
	}

	if err := engine.session.Run(); err != nil {
		return img
	}

	rawOut := engine.outputTensor.GetData()
	outImg := image.NewRGBA(image.Rect(0, 0, ShadowInputSize, ShadowInputSize))

	for y := 0; y < ShadowInputSize; y++ {
		for x := 0; x < ShadowInputSize; x++ {
			r := uint8(math.Max(0, math.Min(255, float64(rawOut[0*ShadowInputSize*ShadowInputSize+y*ShadowInputSize+x]*255.0))))
			g := uint8(math.Max(0, math.Min(255, float64(rawOut[1*ShadowInputSize*ShadowInputSize+y*ShadowInputSize+x]*255.0))))
			b := uint8(math.Max(0, math.Min(255, float64(rawOut[2*ShadowInputSize*ShadowInputSize+y*ShadowInputSize+x]*255.0))))
			outImg.SetRGBA(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}

	return imaging.Resize(outImg, img.Bounds().Dx(), img.Bounds().Dy(), imaging.Lanczos)
}

func runNafnetRestoration(img image.Image, engine *Engine) image.Image {
	if engine == nil {
		return img
	}
	engine.mu.Lock()
	defer engine.mu.Unlock()

	resized := imaging.Resize(img, NafnetInputSize, NafnetInputSize, imaging.Lanczos)
	tensorData := engine.inputTensor.GetData()

	for y := 0; y < NafnetInputSize; y++ {
		for x := 0; x < NafnetInputSize; x++ {
			rCol, gCol, bCol, _ := resized.At(x, y).RGBA()
			tensorData[0*NafnetInputSize*NafnetInputSize+y*NafnetInputSize+x] = float32(rCol>>8) / 255.0
			tensorData[1*NafnetInputSize*NafnetInputSize+y*NafnetInputSize+x] = float32(gCol>>8) / 255.0
			tensorData[2*NafnetInputSize*NafnetInputSize+y*NafnetInputSize+x] = float32(bCol>>8) / 255.0
		}
	}

	if err := engine.session.Run(); err != nil {
		return img
	}

	rawOut := engine.outputTensor.GetData()
	outImg := image.NewRGBA(image.Rect(0, 0, NafnetInputSize, NafnetInputSize))

	for y := 0; y < NafnetInputSize; y++ {
		for x := 0; x < NafnetInputSize; x++ {
			r := uint8(math.Max(0, math.Min(255, float64(rawOut[0*NafnetInputSize*NafnetInputSize+y*NafnetInputSize+x]*255.0))))
			g := uint8(math.Max(0, math.Min(255, float64(rawOut[1*NafnetInputSize*NafnetInputSize+y*NafnetInputSize+x]*255.0))))
			b := uint8(math.Max(0, math.Min(255, float64(rawOut[2*NafnetInputSize*NafnetInputSize+y*NafnetInputSize+x]*255.0))))
			outImg.SetRGBA(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}

	return imaging.Resize(outImg, img.Bounds().Dx(), img.Bounds().Dy(), imaging.Lanczos)
}

func runBackgroundRemoval(img image.Image, engine *Engine) image.Image {
	if engine == nil {
		return img
	}
	engine.mu.Lock()
	defer engine.mu.Unlock()

	resized := imaging.Resize(img, U2NetInputSize, U2NetInputSize, imaging.Lanczos)
	tensorData := engine.inputTensor.GetData()

	mean := []float32{0.485, 0.456, 0.406}
	std := []float32{0.229, 0.224, 0.225}

	for y := 0; y < U2NetInputSize; y++ {
		for x := 0; x < U2NetInputSize; x++ {
			rCol, gCol, bCol, _ := resized.At(x, y).RGBA()
			tensorData[0*U2NetInputSize*U2NetInputSize+y*U2NetInputSize+x] = (float32(rCol>>8)/255.0 - mean[0]) / std[0]
			tensorData[1*U2NetInputSize*U2NetInputSize+y*U2NetInputSize+x] = (float32(gCol>>8)/255.0 - mean[1]) / std[1]
			tensorData[2*U2NetInputSize*U2NetInputSize+y*U2NetInputSize+x] = (float32(bCol>>8)/255.0 - mean[2]) / std[2]
		}
	}

	if err := engine.session.Run(); err != nil {
		return img
	}

	rawMask := engine.outputTensor.GetData()
	minM, maxM := float32(math.MaxFloat32), float32(-math.MaxFloat32)
	for _, v := range rawMask {
		if v < minM {
			minM = v
		}
		if v > maxM {
			maxM = v
		}
	}
	diffM := maxM - minM
	if diffM == 0 {
		diffM = 1
	}

	maskImg := image.NewGray(image.Rect(0, 0, U2NetInputSize, U2NetInputSize))
	for y := 0; y < U2NetInputSize; y++ {
		for x := 0; x < U2NetInputSize; x++ {
			norm := (rawMask[y*U2NetInputSize+x] - minM) / diffM
			val := uint8(math.Min(255, math.Max(0, float64(norm)*255.0)))
			if norm < 0.25 {
				val = 0
			}
			maskImg.SetGray(x, y, color.Gray{Y: val})
		}
	}

	origW, origH := img.Bounds().Dx(), img.Bounds().Dy()
	fullMask := imaging.Resize(maskImg, origW, origH, imaging.Lanczos)
	outImg := image.NewRGBA(image.Rect(0, 0, origW, origH))

	for y := 0; y < origH; y++ {
		for x := 0; x < origW; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			mVal, _, _, _ := fullMask.At(x, y).RGBA()
			alphaRatio := float64(mVal>>8) / 255.0

			outR := uint8(float64(r>>8) * alphaRatio)
			outG := uint8(float64(g>>8) * alphaRatio)
			outB := uint8(float64(b>>8) * alphaRatio)
			outImg.SetRGBA(x, y, color.RGBA{R: outR, G: outG, B: outB, A: 255})
		}
	}

	return outImg
}

func runInference(img image.Image, engine *Engine) [][]float64 {
	if engine == nil {
		fmt.Println("❌ Error: Depth engine is nil")
		return nil
	}

	inW := engine.inputWidth
	inH := engine.inputHeight
	if inW <= 0 {
		inW = ModelInputSize
	}
	if inH <= 0 {
		inH = ModelInputSize
	}

	resized := imaging.Resize(img, inW, inH, imaging.Lanczos)

	if engine.isUint8 {
		tensorData := engine.inputTensorU8.GetData()
		if engine.isNHWC {
			// Packed NHWC format (R, G, B contiguous per pixel)
			for y := 0; y < inH; y++ {
				for x := 0; x < inW; x++ {
					rCol, gCol, bCol, _ := resized.At(x, y).RGBA()
					idx := (y*inW + x) * 3
					tensorData[idx+0] = uint8(rCol >> 8)
					tensorData[idx+1] = uint8(gCol >> 8)
					tensorData[idx+2] = uint8(bCol >> 8)
				}
			}
		} else {
			// Planar NCHW format (R plane, G plane, B plane)
			for y := 0; y < inH; y++ {
				for x := 0; x < inW; x++ {
					rCol, gCol, bCol, _ := resized.At(x, y).RGBA()
					tensorData[0*inH*inW+y*inW+x] = uint8(rCol >> 8)
					tensorData[1*inH*inW+y*inW+x] = uint8(gCol >> 8)
					tensorData[2*inH*inW+y*inW+x] = uint8(bCol >> 8)
				}
			}
		}
	} else {
		// Float32 Planar NCHW format with ImageNet normalization
		tensorData := engine.inputTensor.GetData()
		mean := []float32{0.485, 0.456, 0.406}
		std := []float32{0.229, 0.224, 0.225}

		for y := 0; y < inH; y++ {
			for x := 0; x < inW; x++ {
				rCol, gCol, bCol, _ := resized.At(x, y).RGBA()
				tensorData[0*inH*inW+y*inW+x] = (float32(rCol>>8)/255.0 - mean[0]) / std[0]
				tensorData[1*inH*inW+y*inW+x] = (float32(gCol>>8)/255.0 - mean[1]) / std[1]
				tensorData[2*inH*inW+y*inW+x] = (float32(bCol>>8)/255.0 - mean[2]) / std[2]
			}
		}
	}

	if err := engine.session.Run(); err != nil {
		fmt.Printf("❌ Depth ONNX Run Error: %v\n", err)
		return nil
	}

	rawDepth := engine.outputTensor.GetData()
	if len(rawDepth) == 0 {
		fmt.Println("❌ Error: Output tensor returned 0 elements")
		return nil
	}

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

	out := make([][]float64, inH)
	for y := 0; y < inH; y++ {
		out[y] = make([]float64, inW)
		for x := 0; x < inW; x++ {
			idx := y*inW + x
			if idx < len(rawDepth) {
				out[y][x] = float64(rawDepth[idx]-minVal) / diff
			}
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
			json.NewEncoder(w).Encode(map[string]string{"error": fmt.Sprintf("Processing error: %v", rec)})
		}
	}()

	if err := r.ParseMultipartForm(64 << 20); err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}

	file, _, err := r.FormFile("image")
	if err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to read image: " + err.Error()})
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
	removeBg := r.FormValue("removeBg") == "true"
	hdRestoration := r.FormValue("hdRestoration") == "true"
	targetDim := r.FormValue("targetDim")

	srcImg, err := imaging.Decode(file)
	if err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to decode image: " + err.Error()})
		return
	}

	var delightBase64 string
	var bgBase64 string
	bgOrder := r.FormValue("bgOrder")
	if bgOrder == "" {
		bgOrder = "after"
	}
	tileDensity, _ := strconv.Atoi(r.FormValue("tileDensity"))
	if tileDensity < 2 || tileDensity > 4 {
		tileDensity = 2
	}

	if removeShadows {
		shadowModelName := r.FormValue("shadowModel")
		if shadowModelName == "" {
			shadowModelName = "docshadow_sd7k.onnx"
		}
		selectedEngine := globalShadowEngines[shadowModelName]
		srcImg = runShadowRemoval(srcImg, selectedEngine)

		thumbDelight := imaging.Resize(srcImg, 450, 0, imaging.Lanczos)
		var bufDelight []byte
		wDelight := &byteWriter{buf: &bufDelight}
		jpeg.Encode(wDelight, thumbDelight, &jpeg.Options{Quality: 85})
		delightBase64 = "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(bufDelight)
	}

	if removeBg && bgOrder == "before" && globalBgrEngine != nil {
		srcImg = runBackgroundRemoval(srcImg, globalBgrEngine)
		thumbBg := imaging.Resize(srcImg, 450, 0, imaging.Lanczos)
		var bufBg []byte
		wBg := &byteWriter{buf: &bufBg}
		jpeg.Encode(wBg, thumbBg, &jpeg.Options{Quality: 85})
		bgBase64 = "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(bufBg)
	}

	if hdRestoration && globalNafnetEngine != nil {
		srcImg = runNafnetRestoration(srcImg, globalNafnetEngine)
	}

	origW, origH := srcImg.Bounds().Dx(), srcImg.Bounds().Dy()
	if targetDim == "2k" || targetDim == "4k" {
		maxTarget := 2048
		if targetDim == "4k" {
			maxTarget = 4096
		}
		if origW < maxTarget || origH < maxTarget {
			scale := float64(maxTarget) / math.Max(float64(origW), float64(origH))
			newW := int(float64(origW) * scale)
			newH := int(float64(origH) * scale)
			srcImg = imaging.Resize(srcImg, newW, newH, imaging.Lanczos)
			origW, origH = newW, newH
		}
	}

	if removeBg && bgOrder == "after" && globalBgrEngine != nil {
		srcImg = runBackgroundRemoval(srcImg, globalBgrEngine)
		thumbBg := imaging.Resize(srcImg, 450, 0, imaging.Lanczos)
		var bufBg []byte
		wBg := &byteWriter{buf: &bufBg}
		jpeg.Encode(wBg, thumbBg, &jpeg.Options{Quality: 85})
		bgBase64 = "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(bufBg)
	}

	thumbUpscale := imaging.Resize(srcImg, 650, 0, imaging.Lanczos)
	var bufUpscale []byte
	wUpscale := &byteWriter{buf: &bufUpscale}
	jpeg.Encode(wUpscale, thumbUpscale, &jpeg.Options{Quality: 90})
	upscaleBase64 := "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(bufUpscale)

	depthModelName := r.FormValue("depthModel")
	if depthModelName == "" {
		depthModelName = defaultDepthModel
	}
	selectedDepthEngine := globalDepthEngines[depthModelName]
	if selectedDepthEngine == nil {
		selectedDepthEngine = globalDepthEngines[defaultDepthModel]
	}
	if selectedDepthEngine == nil {
		json.NewEncoder(w).Encode(map[string]string{"error": "No DA3 Depth model is loaded in models/ folder"})
		return
	}

	selectedDepthEngine.mu.Lock()
	defer selectedDepthEngine.mu.Unlock()

	// 1. GLOBAL BASE DEPTH INFERENCE
	baseRaw := runInference(srcImg, selectedDepthEngine)
	if baseRaw == nil {
		json.NewEncoder(w).Encode(map[string]string{"error": "Inference failed"})
		return
	}

	baseDepth := make([][]float64, origH)
	for y := 0; y < origH; y++ {
		baseDepth[y] = make([]float64, origW)
		srcY := int(float64(y) / float64(origH) * float64(ModelInputSize-1))
		for x := 0; x < origW; x++ {
			srcX := int(float64(x) / float64(origW) * float64(ModelInputSize-1))
			baseDepth[y][x] = baseRaw[srcY][srcX]
		}
	}

	compiledTiles := make([][]float64, origH)
	tileWeights := make([][]float64, origH)
	for y := 0; y < origH; y++ {
		compiledTiles[y] = make([]float64, origW)
		tileWeights[y] = make([]float64, origW)
	}

	if useTiling {
		numX, numY := tileDensity, tileDensity
		tileW := origW / numX
		tileH := origH / numY

		var xCoords []int
		for i := 0; i < numX; i++ {
			xCoords = append(xCoords, i*(origW-tileW)/max(1, numX-1))
		}
		for i := 0; i < numX-1; i++ {
			xCoords = append(xCoords, i*tileW+tileW/2)
		}

		var yCoords []int
		for j := 0; j < numY; j++ {
			yCoords = append(yCoords, j*(origH-tileH)/max(1, numY-1))
		}
		for j := 0; j < numY-1; j++ {
			yCoords = append(yCoords, j*tileH+tileH/2)
		}

		for _, ty := range yCoords {
			for _, tx := range xCoords {
				tileRect := image.Rect(tx, ty, tx+tileW, ty+tileH)
				tileSubImg := imaging.Crop(srcImg, tileRect)

				tileRaw := runInference(tileSubImg, selectedDepthEngine)
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
	selectedDepthEngine.cache[id] = result16Bit

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
		"bgPreview":      bgBase64,
		"upscalePreview": upscaleBase64,
		"basePreview":    "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(bufBase),
		"hpPreview":      "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(bufHP),
		"depthPreview":   "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(bufDepth),
	})
}

func handleDownload(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	for _, eng := range globalDepthEngines {
		eng.mu.Lock()
		img, ok := eng.cache[id]
		eng.mu.Unlock()
		if ok {
			w.Header().Set("Content-Type", "image/png")
			w.Header().Set("Content-Disposition", "attachment; filename=openrelief_16bit.png")
			png.Encode(w, img)
			return
		}
	}
	http.Error(w, "Not found", 404)
}

type byteWriter struct{ buf *[]byte }

func (b *byteWriter) Write(p []byte) (n int, err error) {
	*b.buf = append(*b.buf, p...)
	return len(p), nil
}

func handleInfo(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	loadedShadows := []string{}
	for name := range globalShadowEngines {
		loadedShadows = append(loadedShadows, name)
	}

	loadedDA3 := []string{}
	for name := range globalDepthEngines {
		loadedDA3 = append(loadedDA3, name)
	}

	json.NewEncoder(w).Encode(map[string]interface{}{
		"da3Models":    loadedDA3,
		"nafnetLoaded": globalNafnetEngine != nil,
		"u2netLoaded":  globalBgrEngine != nil,
		"shadowModels": loadedShadows,
	})
}
