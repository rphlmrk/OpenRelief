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
	"net/http"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"

	"github.com/disintegration/imaging"
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
	modelPath := "model.onnx" // Uses model.onnx & model.onnx_data

	fmt.Println("🪙 Starting OpenRelief Engine...")
	eng, err := initEngine(dllPath, modelPath)
	if err != nil {
		fmt.Printf("ERROR: %v\n", err)
		fmt.Println("Please make sure onnxruntime.dll, model.onnx, and model.onnx_data are in D:\\OpenRelief")
		return
	}
	defer eng.session.Destroy()
	globalEngine = eng

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		w.Write(htmlContent)
	})

	http.HandleFunc("/api/process", handleProcess)
	http.HandleFunc("/api/download", handleDownload)

	port := 8080
	url := fmt.Sprintf("http://localhost:%d", port)
	fmt.Printf("✅ OpenRelief is running at %s\n", url)

	go openBrowser(url)

	err = http.ListenAndServe(fmt.Sprintf(":%d", port), nil)
	if err != nil {
		fmt.Printf("Server Error: %v\n", err)
	}
}

func initEngine(dllPath, modelPath string) (*Engine, error) {
	// Force absolute paths to prevent Windows from loading System32 dll
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
	globalEngine.mu.Lock()
	defer globalEngine.mu.Unlock()

	file, _, err := r.FormFile("image")
	if err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}
	defer file.Close()

	hpStr := r.FormValue("hpStrength")
	hpStrength, _ := strconv.ParseFloat(hpStr, 64)

	srcImg, err := imaging.Decode(file)
	if err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}

	origW, origH := srcImg.Bounds().Dx(), srcImg.Bounds().Dy()
	resized := imaging.Resize(srcImg, ModelInputSize, ModelInputSize, imaging.Lanczos)
	tensorData := globalEngine.inputTensor.GetData()

	mean := []float32{0.485, 0.456, 0.406}
	std := []float32{0.229, 0.224, 0.225}

	for y := 0; y < ModelInputSize; y++ {
		for x := 0; x < ModelInputSize; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			tensorData[0*ModelInputSize*ModelInputSize+y*ModelInputSize+x] = (float32(r>>8)/255.0 - mean[0]) / std[0]
			tensorData[1*ModelInputSize*ModelInputSize+y*ModelInputSize+x] = (float32(g>>8)/255.0 - mean[1]) / std[1]
			tensorData[2*ModelInputSize*ModelInputSize+y*ModelInputSize+x] = (float32(b>>8)/255.0 - mean[2]) / std[2]
		}
	}

	if err := globalEngine.session.Run(); err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
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

	finalDepth := imaging.Resize(depth518, origW, origH, imaging.Lanczos)
	blurredOrig := imaging.Blur(srcImg, 4.0)

	result16Bit := image.NewGray16(image.Rect(0, 0, origW, origH))
	for y := 0; y < origH; y++ {
		for x := 0; x < origW; x++ {
			dVal := float64(color.Gray16Model.Convert(finalDepth.At(x, y)).(color.Gray16).Y) / 65535.0
			oVal := float64(color.GrayModel.Convert(srcImg.At(x, y)).(color.Gray).Y) / 255.0
			bVal := float64(color.GrayModel.Convert(blurredOrig.At(x, y)).(color.Gray).Y) / 255.0

			highPass := oVal - bVal
			blended := math.Min(1.0, math.Max(0.0, dVal+(highPass*hpStrength)))
			result16Bit.SetGray16(x, y, color.Gray16{Y: uint16(blended * 65535.0)})
		}
	}

	id := "current"
	globalEngine.cache[id] = result16Bit

	thumb := imaging.Resize(result16Bit, 450, 0, imaging.Lanczos)
	var buf []byte
	wWriter := &byteWriter{buf: &buf}
	jpeg.Encode(wWriter, thumb, &jpeg.Options{Quality: 85})

	json.NewEncoder(w).Encode(map[string]string{
		"id":      id,
		"preview": "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(buf),
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
	w.Header().Set("Content-Disposition", "attachment; filename=coin_relief_16bit.png")
	png.Encode(w, img)
}

func openBrowser(url string) {
	if runtime.GOOS == "windows" {
		exec.Command("rundll32", "url.dll,FileProtocolHandler", url).Start()
	}
}

type byteWriter struct{ buf *[]byte }

func (b *byteWriter) Write(p []byte) (n int, err error) {
	*b.buf = append(*b.buf, p...)
	return len(p), nil
}
