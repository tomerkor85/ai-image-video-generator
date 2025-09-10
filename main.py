#!/usr/bin/env python3
"""
AI Image & Video Generator Server - Fixed Version
"""

import os
import torch
import gc
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from PIL import Image
import numpy as np
import cv2
import moviepy.editor as mp

# Import our custom generators
from flux_generator import FluxGenerator
from wan_generator import WanGenerator
from config import Config

# GPU Setup
torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"🎮 Total GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name} - {props.total_memory / 1024**3:.2f} GB")
else:
    device = "cpu"
    print("⚠️ No GPU detected, using CPU")

# Environment setup
os.environ['HF_HOME'] = '/workspace/cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/cache'
os.environ['DIFFUSERS_CACHE'] = '/workspace/cache'

# Create directories FIRST - before using them
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "images").mkdir(exist_ok=True)
(OUTPUT_DIR / "videos").mkdir(exist_ok=True)

# FastAPI App
app = FastAPI(
    title="AI Image & Video Generator",
    version="1.0.0",
    description="Generate images and videos using AI models"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files AFTER creating directories
if (OUTPUT_DIR / "images").exists():
    app.mount("/static", StaticFiles(directory=str(OUTPUT_DIR)), name="static")

# Global model cache
models = {}
flux_generator = None
wan_generator = None

# Request Models
class ImageRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt")
    negative_prompt: str = Field("", description="Negative prompt")
    width: int = Field(512, ge=128, le=2048)
    height: int = Field(512, ge=128, le=2048)
    steps: int = Field(20, ge=1, le=100)
    guidance: float = Field(7.5, ge=1.0, le=20.0)
    seed: Optional[int] = Field(None)
    model_type: str = Field("flux", description="Model type: flux or stable-diffusion")
    lora_scale: float = Field(1.0, ge=0.0, le=2.0, description="LORA scale")

class VideoRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt")
    negative_prompt: str = Field("", description="Negative prompt")
    width: int = Field(512, ge=256, le=1024)
    height: int = Field(512, ge=256, le=1024)
    num_frames: int = Field(16, ge=8, le=32)
    steps: int = Field(25, ge=10, le=50)
    guidance: float = Field(7.5, ge=1.0, le=20.0)
    seed: Optional[int] = Field(None)
    lora_scale: float = Field(1.0, ge=0.0, le=2.0, description="LORA scale")
    lora_type: str = Field("high", description="LORA type: high or low noise")

def get_flux_generator():
    """Get or create FLUX generator"""
    global flux_generator
    
    if flux_generator is None:
        print("📥 Loading FLUX generator...")
        flux_generator = FluxGenerator()
        # Note: We'll load the model on first use to avoid blocking startup
    
    return flux_generator

def get_wan_generator():
    """Get or create WAN generator"""
    global wan_generator
    
    if wan_generator is None:
        print("📥 Loading WAN generator...")
        wan_generator = WanGenerator()
        # Note: We'll load the model on first use to avoid blocking startup
    
    return wan_generator

def get_image_model(model_id: str = "runwayml/stable-diffusion-v1-5"):
    """Load or get cached image model (legacy support)"""
    global models
    
    if model_id not in models:
        print(f"📥 Loading model: {model_id}")
        try:
            from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
            
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            )
            
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config
            )
            
            pipe = pipe.to(device)
            pipe.enable_attention_slicing()
            
            # Try to enable xformers
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("✅ XFormers enabled")
            except:
                print("ℹ️ XFormers not available, using default attention")
            
            models[model_id] = pipe
            print(f"✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    return models[model_id]

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "AI Generator",
        "version": "1.0.0",
        "gpu": {
            "available": torch.cuda.is_available(),
            "count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [torch.cuda.get_device_name(i) 
                       for i in range(torch.cuda.device_count())] 
                       if torch.cuda.is_available() else []
        },
        "endpoints": {
            "/": "This info",
            "/health": "Health check",
            "/generate/image": "Generate image",
            "/generate/video": "Generate video",
            "/ui": "Web UI",
            "/outputs": "List outputs",
            "/docs": "API documentation"
        }
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "cuda": torch.cuda.is_available(),
        "models_loaded": len(models),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate/image")
async def generate_image(req: ImageRequest):
    """Generate image from prompt"""
    try:
        print(f"🎨 Generating: {req.prompt[:50]}...")
        
        if req.model_type == "flux":
            # Use FLUX generator
            generator = get_flux_generator()
            
            # Load model if not already loaded
            if generator.pipeline is None:
                await generator.load_model()
            
            # Generate image
            image = await generator.generate(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                width=req.width,
                height=req.height,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance,
                seed=req.seed,
                lora_scale=req.lora_scale
            )
            
        else:
            # Use legacy Stable Diffusion
            pipe = get_image_model()
            
            # Setup generator
            generator = None
            if req.seed is not None:
                generator = torch.Generator(device=device).manual_seed(req.seed)
            
            # Generate
            with torch.inference_mode():
                result = pipe(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    width=req.width,
                    height=req.height,
                    num_inference_steps=req.steps,
                    guidance_scale=req.guidance,
                    generator=generator
                )
            
            image = result.images[0]
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"img_{timestamp}.png"
        filepath = OUTPUT_DIR / "images" / filename
        
        image.save(filepath)
        print(f"✅ Saved: {filename}")
        
        # Cleanup
        torch.cuda.empty_cache()
        
        return {
            "success": True,
            "filename": filename,
            "path": str(filepath),
            "url": f"/outputs/images/{filename}",
            "model_type": req.model_type
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/video")
async def generate_video(req: VideoRequest):
    """Generate video from prompt"""
    try:
        print(f"🎬 Generating video: {req.prompt[:50]}...")
        
        # Use WAN generator
        generator = get_wan_generator()
        
        # Load model if not already loaded
        if generator.pipeline is None:
            await generator.load_model(lora_type=req.lora_type)
        
        # Generate video frames
        frames = await generator.generate(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance,
            seed=req.seed,
            lora_scale=req.lora_scale
        )
        
        # Convert frames to video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vid_{timestamp}.mp4"
        filepath = OUTPUT_DIR / "videos" / filename
        
        # Create video from frames
        video_path = await frames_to_video(frames, filepath, fps=8)
        
        print(f"✅ Video saved: {filename}")
        
        # Cleanup
        torch.cuda.empty_cache()
        
        return {
            "success": True,
            "filename": filename,
            "path": str(filepath),
            "url": f"/outputs/videos/{filename}",
            "num_frames": len(frames)
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def frames_to_video(frames: List[Image.Image], output_path: Path, fps: int = 8) -> str:
    """Convert PIL frames to MP4 video"""
    try:
        # Convert PIL images to numpy arrays
        frame_arrays = []
        for frame in frames:
            # Convert PIL to numpy
            frame_array = np.array(frame)
            # Convert RGB to BGR for OpenCV
            if len(frame_array.shape) == 3:
                frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            frame_arrays.append(frame_array)
        
        # Get video dimensions
        height, width = frame_arrays[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Write frames
        for frame_array in frame_arrays:
            video_writer.write(frame_array)
        
        # Release video writer
        video_writer.release()
        
        return str(output_path)
        
    except Exception as e:
        print(f"❌ Error creating video: {e}")
        raise e

@app.get("/outputs")
async def list_outputs():
    """List generated files"""
    images = list((OUTPUT_DIR / "images").glob("*.png"))
    videos = list((OUTPUT_DIR / "videos").glob("*.mp4"))
    
    return {
        "images": [f.name for f in sorted(images, reverse=True)[:20]],
        "videos": [f.name for f in sorted(videos, reverse=True)[:20]]
    }

@app.get("/outputs/images/{filename}")
async def get_image(filename: str):
    """Get generated image"""
    filepath = OUTPUT_DIR / "images" / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(filepath)

@app.get("/outputs/videos/{filename}")
async def get_video(filename: str):
    """Get generated video"""
    filepath = OUTPUT_DIR / "videos" / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(filepath)

@app.get("/ui", response_class=HTMLResponse)
async def serve_ui():
    """Serve web UI"""
    html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>AI Image Generator</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            margin: 0;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        
        h1 {
            color: white;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #333;
        }
        
        input, textarea, select {
            width: 100%;
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            box-sizing: border-box;
        }
        
        textarea {
            min-height: 80px;
            resize: vertical;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
        }
        
        button:hover {
            opacity: 0.9;
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        #status {
            margin-top: 15px;
            padding: 12px;
            border-radius: 8px;
            display: none;
        }
        
        #status.show {
            display: block;
        }
        
        #status.info {
            background: #e3f2fd;
            color: #1565c0;
        }
        
        #status.success {
            background: #e8f5e9;
            color: #2e7d32;
        }
        
        #status.error {
            background: #ffebee;
            color: #c62828;
        }
        
        #result {
            margin-top: 20px;
            text-align: center;
        }
        
        #result img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        
        .gallery img {
            width: 100%;
            height: 150px;
            object-fit: cover;
            border-radius: 8px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 AI Image Generator</h1>
        
        <div class="card">
            <h2>🎨 Generate Content</h2>
            <form id="genForm">
                <div class="form-group">
                    <label>Content Type:</label>
                    <select id="content_type">
                        <option value="image" selected>🖼️ Image</option>
                        <option value="video">🎬 Video</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>Model Type:</label>
                    <select id="model_type">
                        <option value="flux" selected>🚀 FLUX.1-dev (with LORA)</option>
                        <option value="stable-diffusion">🎯 Stable Diffusion</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>Prompt:</label>
                    <textarea id="prompt" required>a beautiful sunset over mountains, highly detailed digital art</textarea>
                </div>
                
                <div class="form-group">
                    <label>Negative Prompt (optional):</label>
                    <input type="text" id="negative_prompt" placeholder="blurry, low quality">
                </div>
                
                <div class="grid">
                    <div class="form-group">
                        <label>Width:</label>
                        <select id="width">
                            <option value="512" selected>512</option>
                            <option value="768">768</option>
                            <option value="1024">1024</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Height:</label>
                        <select id="height">
                            <option value="512" selected>512</option>
                            <option value="768">768</option>
                            <option value="1024">1024</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Steps:</label>
                        <input type="number" id="steps" value="25" min="10" max="50">
                    </div>
                    
                    <div class="form-group">
                        <label>Guidance:</label>
                        <input type="number" id="guidance" value="7.5" min="1" max="20" step="0.5">
                    </div>
                </div>
                
                <div class="grid" id="video_options" style="display: none;">
                    <div class="form-group">
                        <label>Frames:</label>
                        <select id="num_frames">
                            <option value="8">8 frames</option>
                            <option value="16" selected>16 frames</option>
                            <option value="24">24 frames</option>
                            <option value="32">32 frames</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>LORA Type:</label>
                        <select id="lora_type">
                            <option value="high" selected>🔊 High Noise LORA</option>
                            <option value="low">🔇 Low Noise LORA</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>LORA Scale:</label>
                        <input type="number" id="lora_scale" value="1.0" min="0.0" max="2.0" step="0.1">
                    </div>
                </div>
                
                <button type="submit" id="genBtn">🚀 Generate Image</button>
            </form>
            
            <div id="status"></div>
            <div id="result"></div>
        </div>
        
        <div class="card">
            <h2>📸 Recent Creations</h2>
            <div class="gallery" id="gallery"></div>
        </div>
    </div>
    
    <script>
        const form = document.getElementById('genForm');
        const status = document.getElementById('status');
        const result = document.getElementById('result');
        const btn = document.getElementById('genBtn');
        const gallery = document.getElementById('gallery');
        const contentType = document.getElementById('content_type');
        const videoOptions = document.getElementById('video_options');
        
        // Show/hide video options based on content type
        contentType.addEventListener('change', function() {
            if (this.value === 'video') {
                videoOptions.style.display = 'block';
                btn.textContent = '🎬 Generate Video';
            } else {
                videoOptions.style.display = 'none';
                btn.textContent = '🚀 Generate Image';
            }
        });
        
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const isVideo = contentType.value === 'video';
            const endpoint = isVideo ? '/generate/video' : '/generate/image';
            
            const data = {
                prompt: document.getElementById('prompt').value,
                negative_prompt: document.getElementById('negative_prompt').value,
                width: parseInt(document.getElementById('width').value),
                height: parseInt(document.getElementById('height').value),
                steps: parseInt(document.getElementById('steps').value),
                guidance: parseFloat(document.getElementById('guidance').value),
                model_type: document.getElementById('model_type').value,
                lora_scale: parseFloat(document.getElementById('lora_scale').value)
            };
            
            if (isVideo) {
                data.num_frames = parseInt(document.getElementById('num_frames').value);
                data.lora_type = document.getElementById('lora_type').value;
            }
            
            btn.disabled = true;
            btn.textContent = isVideo ? '🎬 Generating Video...' : '🚀 Generating Image...';
            status.className = 'show info';
            status.textContent = isVideo ? 
                'Creating your video... (30-60 seconds)' : 
                'Creating your image... (10-30 seconds)';
            result.innerHTML = '';
            
            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                const res = await response.json();
                
                if (res.success) {
                    status.className = 'show success';
                    status.textContent = isVideo ? 
                        'Video generated successfully!' : 
                        'Image generated successfully!';
                    
                    if (isVideo) {
                        result.innerHTML = `
                            <video controls style="max-width: 100%; border-radius: 10px;">
                                <source src="${res.url}" type="video/mp4">
                                Your browser does not support the video tag.
                            </video>
                            <p style="margin-top: 10px; color: #666;">
                                <small>${res.filename} (${res.num_frames} frames)</small>
                            </p>
                        `;
                    } else {
                        result.innerHTML = `<img src="${res.url}" alt="Generated">`;
                    }
                    
                    loadGallery();
                } else {
                    throw new Error(res.message || 'Failed');
                }
            } catch (err) {
                status.className = 'show error';
                status.textContent = 'Error: ' + err.message;
            } finally {
                btn.disabled = false;
                btn.textContent = isVideo ? '🎬 Generate Video' : '🚀 Generate Image';
            }
        });
        
        async function loadGallery() {
            try {
                const res = await fetch('/outputs');
                const data = await res.json();
                
                let galleryHTML = '';
                
                // Add recent images
                data.images.slice(0, 8).forEach(img => {
                    galleryHTML += `
                        <div style="position: relative;">
                            <img src="/outputs/images/${img}" 
                                 onclick="window.open('/outputs/images/${img}')"
                                 style="width: 100%; height: 150px; object-fit: cover; border-radius: 8px; cursor: pointer;">
                            <div style="position: absolute; top: 5px; left: 5px; background: rgba(0,0,0,0.7); color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px;">
                                IMG
                            </div>
                        </div>
                    `;
                });
                
                // Add recent videos
                data.videos.slice(0, 4).forEach(vid => {
                    galleryHTML += `
                        <div style="position: relative;">
                            <video style="width: 100%; height: 150px; object-fit: cover; border-radius: 8px; cursor: pointer;"
                                   onclick="window.open('/outputs/videos/${vid}')"
                                   onmouseover="this.play()" 
                                   onmouseout="this.pause()"
                                   muted loop>
                                <source src="/outputs/videos/${vid}" type="video/mp4">
                            </video>
                            <div style="position: absolute; top: 5px; left: 5px; background: rgba(0,0,0,0.7); color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px;">
                                VID
                            </div>
                        </div>
                    `;
                });
                
                gallery.innerHTML = galleryHTML;
            } catch (err) {
                console.error(err);
            }
        }
        
        loadGallery();
    </script>
</body>
</html>
    '''
    return html_content

if __name__ == "__main__":
    print("="*60)
    print("🚀 AI Image Generator Server")
    print("="*60)
    print(f"📍 URL: http://0.0.0.0:8888")
    print(f"📚 API Docs: http://0.0.0.0:8888/docs")
    print(f"🎨 Web UI: http://0.0.0.0:8888/ui")
    print("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8888)