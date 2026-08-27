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

	fmt.Println("🪙 Starting OpenRelief Studio (Dual 3D View & High-Pass Engine)...")
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
	w.SetTitle("OpenRelief Studio - Base vs Filtered 3D Relief Studio")
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

	hpRadius, _ := strconv.ParseFloat(r.FormValue("hpRadius"), 64)
	if hpRadius <= 0 {
		hpRadius = 3.5
	}
	hpStrength, _ := strconv.ParseFloat(r.FormValue("hpStrength"), 64)
	isCoinMask := r.FormValue("coinMask") == "true"
	isInvert := r.FormValue("invert") == "true"

	srcImg, err := imaging.Decode(file)
	if err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to decode image: " + err.Error()})
		return
	}

	origW, origH := srcImg.Bounds().Dx(), srcImg.Bounds().Dy()

	// 1. AI BASE DEPTH MAP
	resized := imaging.Resize(srcImg, ModelInputSize, ModelInputSize, imaging.Lanczos)
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
		json.NewEncoder(w).Encode(map[string]string{"error": "ONNX Model error: " + err.Error()})
		return
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

	depth518 := image.NewGray16(image.Rect(0, 0, ModelInputSize, ModelInputSize))
	for y := 0; y < ModelInputSize; y++ {
		for x := 0; x < ModelInputSize; x++ {
			val := (rawDepth[y*ModelInputSize+x] - minVal) / (maxVal - minVal)
			depth518.SetGray16(x, y, color.Gray16{Y: uint16(val * 65535.0)})
		}
	}

	layer1_DepthMap := imaging.Resize(depth518, origW, origH, imaging.Lanczos)

	// 2. HIGH PASS TOP LAYER (From Original Image)
	origGray := imaging.Grayscale(srcImg)
	origBlurred := imaging.Blur(origGray, hpRadius)

	hpVisualizer := image.NewGray(image.Rect(0, 0, origW, origH))
	baseDepthVisualizer := image.NewGray16(image.Rect(0, 0, origW, origH))
	result16Bit := image.NewGray16(image.Rect(0, 0, origW, origH))

	centerX, centerY := float64(origW)/2.0, float64(origH)/2.0
	radius := math.Min(centerX, centerY) - 4.0

	// 3. COLOR-TO-ALPHA COMPOSITING OVER BASE DEPTH
	for y := 0; y < origH; y++ {
		for x := 0; x < origW; x++ {
			dr, _, _, _ := layer1_DepthMap.At(x, y).RGBA()
			dVal := float64(dr) / 65535.0

			gr, _, _, _ := origGray.At(x, y).RGBA()
			gVal := float64(gr) / 65535.0

			br, _, _, _ := origBlurred.At(x, y).RGBA()
			bVal := float64(br) / 65535.0

			// High Pass output = 0.5 + (Original - Blurred)
			hpVal := 0.5 + (gVal - bVal)
			hpVisualizer.SetGray(x, y, color.Gray{Y: uint8(math.Min(255.0, math.Max(0.0, hpVal*255.0)))})

			// GIMP Color-to-Alpha (#808080) Composited Over Depth
			var blended float64
			if hpVal >= 0.5 {
				alpha := math.Min(1.0, 2.0*(hpVal-0.5)*hpStrength)
				blended = dVal + alpha*(1.0-dVal)
			} else {
				alpha := math.Min(1.0, 2.0*(0.5-hpVal)*hpStrength)
				blended = dVal - (alpha * dVal)
			}

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

	// Thumbnails for UI
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
