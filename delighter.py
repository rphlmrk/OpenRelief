# delighter.py
import io
import os
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, send_file

from chrislib.general import view
from intrinsic.pipeline import load_models, run_pipeline

app = Flask(__name__)

print("Loading Intrinsic Model into VRAM...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
intrinsic_model = load_models('v2')
print("Model loaded! Python Delighting Server is ready on port 5000.")

def process_image(img, max_dim=1024):
    # Step 1: Size Check
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    
    # Step 2: Shadow Removal
    img_np = np.array(img).astype(np.float32) / 255.0
    result = run_pipeline(intrinsic_model, img_np, device=device)
    
    albedo_np = view(result['hr_alb'])
    albedo_uint8 = (np.clip(albedo_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(albedo_uint8)

@app.route('/delight', methods=['POST'])
def delight_endpoint():
    if 'image' not in request.files:
        return {"error": "No image provided"}, 400
    
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    
    # Run the AI processing
    delighted_img = process_image(img)
    
    # Save to bytes and return to Go
    img_io = io.BytesIO()
    delighted_img.save(img_io, 'PNG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(port=5000, debug=False)