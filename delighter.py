import io
import os
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, send_file

from chrislib.general import view
from intrinsic.pipeline import load_models, run_pipeline

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading Intrinsic Model into memory using device: {device}...")

intrinsic_model = load_models('v2', device=device)
print("Model loaded! Python Delighting Server is ready on port 5000.")

def process_image(img, max_dim=1024):
    # Step 1: Size Check
    w, h = img.size
    print(f"[*] Original image size: {w}x{h}")
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"[*] Resized to {new_w}x{new_h} for safe processing.")
    
    # Step 2: Shadow Removal
    print("[*] Running Intrinsic AI to remove shadows (This takes 1-3 mins on CPU)...")
    img_np = np.array(img).astype(np.float32) / 255.0
    result = run_pipeline(intrinsic_model, img_np, device=device)
    
    albedo_np = view(result['hr_alb'])
    albedo_uint8 = (np.clip(albedo_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    print("[+] Shadows successfully removed!")
    return Image.fromarray(albedo_uint8)

@app.route('/delight', methods=['POST'])
def delight_endpoint():
    print("\n--------------------------------------------------")
    print("[+] Incoming request: AI Shadow Removal started...")
    
    if 'image' not in request.files:
        print("[-] Error: No image provided")
        return {"error": "No image provided"}, 400
    
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    
    # Run the AI processing
    delighted_img = process_image(img)
    
    print("[+] Packaging image and sending back to Go App...")
    # Save to bytes and return to Go
    img_io = io.BytesIO()
    delighted_img.save(img_io, 'PNG')
    img_io.seek(0)
    
    print("[+] Request complete!")
    print("--------------------------------------------------\n")
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(port=5000, debug=False)