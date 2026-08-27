# 🪙 OpenRelief

> **Open-Source, 100% Offline 16-Bit Depth & Bas-Relief Generator for ZBrush, CNC & 3D Printing.**

**OpenRelief** is a lightweight, standalone desktop tool built with **Go** and **ONNX Runtime**. It transforms standard 2D images (such as AI-generated art or photos) into high-precision, 16-bit displacement maps tailored for coins, medallions, CNC wood/metal carving, and digital sculpting.

---

## ✨ Features

- **⚡ Blazing Fast & Offline:** Runs locally on your CPU/GPU using Depth Anything V2 ONNX models in < 1 second. No cloud queues, no credit subscriptions, 100% private.
- **🎨 Micro-Detail High-Pass Engine:** Preserves fine line art, hair, and sharp edges by blending macro depth geometry with high-frequency surface details.
- **💎 Pure 16-Bit Grayscale Output:** Eliminates stepping/terracing artifacts when displacing geometry in ZBrush, Blender, or Vectric Aspire.
- **🪶 Zero Heavy Dependencies:** Built in pure Go with native desktop graphics (Fyne). No Node.js, Python, or npm required.

---

## 🚀 Quick Start (Running from Source)

### Prerequisites
- [Go 1.20+](https://go.dev/dl/)
- A C compiler (e.g., MinGW-w64 or w64devkit on Windows)
- `onnxruntime.dll` (or `.so` / `.dylib`) in the root directory
- `depth_anything_v2_vits.onnx` model file in the root directory

### Installation & Run

```bash
# Clone the repository
git clone https://github.com/yourusername/openrelief.git
cd openrelief

# Download Go dependencies
go mod tidy

# Run the app
go run main.go