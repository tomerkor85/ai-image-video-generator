#!/usr/bin/env python3
"""
AI Image & Video Generator for RunPod - Uncensored
Supports FLUX LORA and WAN2.2 models
"""

import os
import torch
import gc
import getpass
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import asyncio
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from PIL import Image
import numpy as np

# Import generators
from generators.flux_generator import FluxGenerator
from generators.wan_generator import WanGenerator
from utils.model_downloader import download_models

# GPU Setup
torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"
    logger.info(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"🎮 Total GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"   GPU {i}: {props.name} - {props.total_memory / 1024**3:.2f} GB")
else:
    device = "cpu"
    logger.warning("⚠️ No GPU detected, using CPU")

# Environment setup for RunPod
os.environ['HF_HOME'] = '/workspace/cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/cache'
os.environ['DIFFUSERS_CACHE'] = '/workspace/cache'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Hugging Face Token Setup
def setup_huggingface_token():
    """Setup Hugging Face token from various sources"""
    token = None
    
    # Method 1: Environment variable from .env file or system
    token = (
        os.environ.get("HUGGINGFACE_TOKEN") or 
        os.environ.get("HF_TOKEN") or
        os.environ.get("HUGGING_FACE_TOKEN")
    )
    
    if token:
        logger.info("✅ Hugging Face token loaded from environment (.env file or system)")
        return token
    
    # Method 2: Check for token file
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        try:
            token = token_file.read_text().strip()
            logger.info("✅ Hugging Face token loaded from cache file")
            return token
        except:
            pass
    
    # Method 3: Manual input (for interactive setup)
    logger.warning("⚠️ No Hugging Face token found!")
    logger.info("💡 You can set it by:")
    logger.info("   1. Environment: export HUGGINGFACE_TOKEN='your_token'")
    logger.info("   2. Or run: huggingface-cli login")
    
    return None

# Setup HF token on startup
HF_TOKEN = setup_huggingface_token()
if HF_TOKEN:
    os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN

# Create directories
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "images").mkdir(exist_ok=True)
(OUTPUT_DIR / "videos").mkdir(exist_ok=True)

# FastAPI App
app = FastAPI(
    title="🔥 Uncensored AI Generator",
    version="2.0.0",
    description="Professional AI Image & Video Generation for Adults - No Censorship"
)

# CORS - Allow all origins for RunPod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Global generators
flux_generator = None
wan_generator = None

# Request Models
class ImageRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: str = Field("", description="Negative prompt")
    model: str = Field("flux_schnell", description="Model to use for generation")
    width: int = Field(1024, ge=512, le=2048, description="Image width")
    height: int = Field(1024, ge=512, le=2048, description="Image height")
    steps: int = Field(25, ge=10, le=50, description="Inference steps")
    guidance: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale")
    seed: Optional[int] = Field(None, description="Random seed")
    lora_scale: float = Field(1.0, ge=0.0, le=2.0, description="LORA strength")
    num_images: int = Field(1, ge=1, le=8, description="Number of images to generate")
    use_lora: bool = Field(True, description="Whether to use LORA")

class VideoRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for video generation")
    negative_prompt: str = Field("", description="Negative prompt")
    width: int = Field(512, ge=256, le=1024, description="Video width")
    height: int = Field(512, ge=256, le=1024, description="Video height")
    num_frames: int = Field(16, ge=8, le=32, description="Number of frames")
    steps: int = Field(25, ge=10, le=50, description="Inference steps")
    guidance: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale")
    seed: Optional[int] = Field(None, description="Random seed")
    fps: int = Field(8, ge=4, le=30, description="Frames per second")
    duration: float = Field(2.0, ge=0.5, le=10.0, description="Video duration in seconds")
    use_lora: bool = Field(True, description="Whether to use LORA")

def get_flux_generator():
    """Get or create FLUX generator"""
    global flux_generator
    if flux_generator is None:
        logger.info("📥 Initializing FLUX generator...")
        flux_generator = FluxGenerator()
    return flux_generator

def get_wan_generator():
    """Get or create WAN generator"""
    global wan_generator
    if wan_generator is None:
        logger.info("📥 Initializing WAN generator...")
        wan_generator = WanGenerator()
    return wan_generator

@app.on_event("startup")
async def startup_event():
    """Download models on startup"""
    logger.info("🚀 Starting AI Generator...")
    await download_models()
    logger.info("✅ Ready to generate!")

@app.get("/")
async def root():
    """API information"""
    hf_status = "✅ Connected" if os.environ.get("HUGGINGFACE_TOKEN") else "❌ No token"
    
    return {
        "name": "🔥 Uncensored AI Generator",
        "version": "2.0.0",
        "description": "Professional AI Generation - No Limits",
        "huggingface_status": hf_status,
        "gpu": {
            "available": torch.cuda.is_available(),
            "count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        },
        "endpoints": {
            "/": "API info",
            "/health": "Health check",
            "/generate/image": "Generate image with FLUX",
            "/generate/video": "Generate video with WAN2.2",
            "/ui": "Web interface",
            "/outputs": "Generated files",
            "/setup/hf-token": "Setup Hugging Face token",
            "/docs": "API documentation"
        }
    }

@app.get("/health")
async def health():
    """Health check for RunPod"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gpu_available": torch.cuda.is_available(),
        "memory_free": torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
    }

@app.post("/generate/image")
async def generate_image(req: ImageRequest, background_tasks: BackgroundTasks):
    """Generate uncensored image using FLUX with LORA"""
    try:
        logger.info(f"🎨 Generating image: {req.prompt[:50]}...")
        
        generator = get_flux_generator()
        
        # Switch model if requested
        if req.model != generator.current_model:
            generator.switch_model(req.model)
        
        # Load model if needed
        if not generator.is_loaded():
            await generator.load_model()
        
        # Generate multiple images
        images = []
        filenames = []
        
        for i in range(req.num_images):
            # Use different seed for each image if seed is provided
            current_seed = req.seed + i if req.seed is not None else None
            
            image = await generator.generate(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                width=req.width,
                height=req.height,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance,
                seed=current_seed,
                lora_scale=req.lora_scale if req.use_lora else 0.0,
                use_lora=req.use_lora
            )
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flux_{timestamp}_{i+1}.png"
            filepath = OUTPUT_DIR / "images" / filename
            
            image.save(filepath, quality=95)
            images.append(image)
            filenames.append(filename)
            
            logger.info(f"✅ Image {i+1}/{req.num_images} saved: {filename}")
        
        logger.info(f"✅ All {req.num_images} images generated successfully!")
        
        # Cleanup
        background_tasks.add_task(cleanup_memory)
        
        return {
            "success": True,
            "count": req.num_images,
            "filenames": filenames,
            "urls": [f"/outputs/images/{fn}" for fn in filenames],
            "model": f"{generator.available_models[generator.current_model]['description']} + LORA",
            "settings": {
                "size": f"{req.width}x{req.height}",
                "steps": req.steps,
                "guidance": req.guidance,
                "lora_scale": req.lora_scale if req.use_lora else "disabled",
                "use_lora": req.use_lora
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Image generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/video")
async def generate_video(req: VideoRequest, background_tasks: BackgroundTasks):
    """Generate uncensored video using WAN2.2"""
    try:
        logger.info(f"🎬 Generating video: {req.prompt[:50]}...")
        
        # Calculate frames from duration and FPS
        calculated_frames = int(req.duration * req.fps)
        actual_frames = min(max(calculated_frames, 8), 32)  # Clamp between 8-32
        
        generator = get_wan_generator()
        
        # Load model if needed
        if not generator.is_loaded():
            await generator.load_model()
        
        # Generate video frames
        frames = await generator.generate(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            num_frames=actual_frames,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance,
            seed=req.seed,
            lora_scale=1.0 if req.use_lora else 0.0,
            use_lora=req.use_lora
        )
        
        # Save video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"wan_{timestamp}.mp4"
        filepath = OUTPUT_DIR / "videos" / filename
        
        await generator.save_video(frames, filepath, fps=req.fps)
        logger.info(f"✅ Video saved: {filename}")
        
        # Cleanup
        background_tasks.add_task(cleanup_memory)
        
        return {
            "success": True,
            "filename": filename,
            "url": f"/outputs/videos/{filename}",
            "model": "WAN2.2",
            "settings": {
                "size": f"{req.width}x{req.height}",
                "frames": actual_frames,
                "duration": f"{req.duration}s",
                "fps": req.fps,
                "steps": req.steps,
                "guidance": req.guidance,
                "use_lora": req.use_lora
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Video generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/outputs/list")
async def list_outputs():
    """List generated files"""
    try:
        images = list((OUTPUT_DIR / "images").glob("*.png"))
        videos = list((OUTPUT_DIR / "videos").glob("*.mp4"))
        
        return {
            "images": [{"name": f.name, "size": f.stat().st_size, "created": f.stat().st_mtime} for f in sorted(images, reverse=True)[:20]],
            "videos": [{"name": f.name, "size": f.stat().st_size, "created": f.stat().st_mtime} for f in sorted(videos, reverse=True)[:20]]
        }
    except Exception as e:
        logger.error(f"Error listing outputs: {e}")
        return {
            "images": [],
            "videos": []
        }
    
    return {
        "images": [{"name": f.name, "size": f.stat().st_size, "created": f.stat().st_mtime} for f in sorted(images, reverse=True)[:20]],
        "videos": [{"name": f.name, "size": f.stat().st_size, "created": f.stat().st_mtime} for f in sorted(videos, reverse=True)[:20]]
    }

@app.get("/models/available")
async def get_available_models():
    """Get available models for image generation"""
    generator = get_flux_generator()
    return {
        "models": generator.get_available_models(),
        "current": generator.current_model
    }

@app.post("/setup/hf-token")
async def setup_hf_token(request: dict):
    """Setup Hugging Face token"""
    try:
        token = request.get("token") or request.get("hf_token")
        if not token:
            raise HTTPException(status_code=400, detail="Token is required")
        
        # Validate token format (basic check)
        if not token.startswith("hf_") or len(token) < 20:
            raise HTTPException(status_code=400, detail="Invalid token format")
        
        # Set environment variable
        os.environ["HUGGINGFACE_TOKEN"] = token
        
        # Save to cache file for persistence
        cache_dir = Path.home() / ".cache" / "huggingface"
        cache_dir.mkdir(parents=True, exist_ok=True)
        token_file = cache_dir / "token"
        token_file.write_text(token)
        
        logger.info("✅ Hugging Face token updated successfully")
        
        return {
            "success": True,
            "message": "Hugging Face token updated successfully",
            "status": "✅ Connected"
        }
        
    except Exception as e:
        logger.error(f"❌ Error setting HF token: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def cleanup_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

@app.get("/ui", response_class=HTMLResponse)
async def serve_ui():
    """Serve advanced web UI"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔥 Uncensored AI Generator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            color: white;
        }
        
        .header h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .tabs {
            display: flex;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 5px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
        }
        
        .tab {
            flex: 1;
            padding: 15px 20px;
            text-align: center;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            color: white;
            font-weight: 600;
        }
        
        .tab.active {
            background: rgba(255,255,255,0.2);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .tab-content {
            display: none;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(20px);
        }
        
        .tab-content.active {
            display: block;
            animation: fadeIn 0.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .form-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        .form-group input,
        .form-group textarea,
        .form-group select {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 14px;
            transition: border-color 0.3s ease;
        }
        
        .form-group input:focus,
        .form-group textarea:focus,
        .form-group select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .form-group textarea {
            min-height: 100px;
            resize: vertical;
        }
        
        .settings-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .generate-btn {
            width: 100%;
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 20px;
        }
        
        .generate-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        .generate-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .status {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: none;
        }
        
        .status.show { display: block; }
        .status.info { background: #e3f2fd; color: #1565c0; }
        .status.success { background: #e8f5e9; color: #2e7d32; }
        .status.error { background: #ffebee; color: #c62828; }
        
        .result {
            text-align: center;
            margin-top: 20px;
        }
        
        .result img, .result video {
            max-width: 100%;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 30px;
        }
        
        .gallery-item {
            position: relative;
            border-radius: 10px;
            overflow: hidden;
            cursor: pointer;
            transition: transform 0.3s ease;
        }
        
        .gallery-item:hover {
            transform: scale(1.05);
        }
        
        .gallery-item img,
        .gallery-item video {
            width: 100%;
            height: 200px;
            object-fit: cover;
        }
        
        .gallery-item .overlay {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .warning {
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
            text-align: center;
        }
        
        .warning h3 {
            margin-bottom: 10px;
            font-size: 1.3em;
        }
        
        @media (max-width: 768px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
            
            .settings-grid {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔥 Uncensored AI Generator</h1>
            <p>Professional AI Image & Video Generation - No Limits</p>
        </div>
        
        <div class="warning">
            <h3>⚠️ Adults Only Content</h3>
            <p>This tool is designed for professional adult content creation. Use responsibly.</p>
        </div>
        
        <!-- HF Token Setup -->
        <div id="token-setup" class="tab-content" style="display: none; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
            <h3>🔑 Setup Hugging Face Token</h3>
            <p>Enter your Hugging Face token to access models:</p>
            <div style="display: flex; gap: 10px; margin-top: 10px;">
                <input type="password" id="hf-token-input" placeholder="hf_your_token_here" style="flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                <button onclick="setupToken()" style="padding: 10px 20px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer;">Save Token</button>
            </div>
            <p style="margin-top: 10px; font-size: 12px; color: #666;">
                Get your token from: <a href="https://huggingface.co/settings/tokens" target="_blank">huggingface.co/settings/tokens</a>
            </p>
        </div>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('image')">
                🎨 Image Generation (FLUX)
            </div>
            <div class="tab" onclick="switchTab('video')">
                🎬 Video Generation (WAN2.2)
            </div>
        </div>
        
        <!-- Image Generation Tab -->
        <div id="image-tab" class="tab-content active">
            <form id="imageForm">
                <div class="form-grid">
                    <div>
                        <div class="form-group">
                            <label>🤖 Model Selection</label>
                            <select id="image-model">
                                <option value="flux_dev" selected>FLUX.1-dev + NAYA2 LORA - Best quality</option>
                                <option value="flux_schnell">FLUX.1-schnell - Fast generation</option>
                                <option value="sdxl_base">SDXL Base - Very permissive</option>
                                <option value="realistic_vision">Realistic Vision V6 - Best for photorealistic</option>
                                <option value="dreamshaper">DreamShaper - Versatile styles</option>
                                <option value="deliberate">Deliberate - Detailed content</option>
                                <option value="civitai_custom">Custom CivitAI Model</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>🎯 Prompt</label>
                            <textarea id="image-prompt" placeholder="Describe your image in detail..." required>a beautiful woman, professional photography, high quality, detailed</textarea>
                        </div>
                        
                        <div class="form-group">
                            <label>🚫 Negative Prompt</label>
                            <textarea id="image-negative" placeholder="What to avoid...">blurry, low quality, distorted, ugly</textarea>
                        </div>
                    </div>
                    
                    <div>
                        <div class="settings-grid">
                            <div class="form-group">
                                <label>🔢 Number of Images</label>
                                <select id="image-count">
                                    <option value="1" selected>1 image</option>
                                    <option value="2">2 images</option>
                                    <option value="4">4 images</option>
                                    <option value="8">8 images</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label>📐 Width</label>
                                <select id="image-width">
                                    <option value="512">512px</option>
                                    <option value="768">768px</option>
                                    <option value="1024" selected>1024px</option>
                                    <option value="1536">1536px</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label>📏 Height</label>
                                <select id="image-height">
                                    <option value="512">512px</option>
                                    <option value="768">768px</option>
                                    <option value="1024" selected>1024px</option>
                                    <option value="1536">1536px</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label>⚡ Steps</label>
                                <input type="number" id="image-steps" value="25" min="10" max="50">
                            </div>
                            
                            <div class="form-group">
                                <label>🎛️ Guidance</label>
                                <input type="number" id="image-guidance" value="7.5" min="1" max="20" step="0.5">
                            </div>
                            
                            <div class="form-group">
                                <label>🔧 LORA Scale</label>
                                <input type="number" id="image-lora" value="1.0" min="0" max="2" step="0.1">
                            </div>
                            
                            <div class="form-group">
                                <label>🎲 Seed (Optional)</label>
                                <input type="number" id="image-seed" placeholder="Random">
                            </div>
                            
                            <div class="form-group">
                                <label>🔧 Use LORA</label>
                                <select id="image-use-lora">
                                    <option value="true" selected>Yes (Enhanced)</option>
                                    <option value="false">No (Base Model)</option>
                                </select>
                            </div>
                        </div>
                    </div>
                </div>
                                <label>🎲 Seed (Optional)</label>
                                <input type="number" id="image-seed" placeholder="Random">
                            </div>
                        </div>
                    </div>
                </div>
                
                <button type="submit" class="generate-btn" id="imageBtn">
                    🚀 Generate Image
                </button>
            </form>
            
            <div id="image-status" class="status"></div>
            <div id="image-result" class="result"></div>
        </div>
        
        <!-- Video Generation Tab -->
        <div id="video-tab" class="tab-content">
            <form id="videoForm">
                <div class="form-grid">
                    <div>
                        <div class="form-group">
                            <label>🎯 Prompt</label>
                            <textarea id="video-prompt" placeholder="Describe your video..." required>a woman dancing, smooth motion, cinematic</textarea>
                        </div>
                        
                        <div class="form-group">
                            <label>🚫 Negative Prompt</label>
                            <textarea id="video-negative" placeholder="What to avoid...">static, blurry, low quality, jumpy motion</textarea>
                        </div>
                    </div>
                    
                    <div>
                        <div class="settings-grid">
                            <div class="form-group">
                                <label>⏱️ Duration</label>
                                <select id="video-duration">
                                    <option value="0.5">0.5 seconds</option>
                                    <option value="1">1 second</option>
                                    <option value="2" selected>2 seconds</option>
                                    <option value="3">3 seconds</option>
                                    <option value="4">4 seconds</option>
                                    <option value="5">5 seconds</option>
                                    <option value="8">8 seconds</option>
                                    <option value="10">10 seconds</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label>📐 Width</label>
                                <select id="video-width">
                                    <option value="256">256px</option>
                                    <option value="512" selected>512px</option>
                                    <option value="768">768px</option>
                                    <option value="1024">1024px</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label>📏 Height</label>
                                <select id="video-height">
                                    <option value="256">256px</option>
                                    <option value="512" selected>512px</option>
                                    <option value="768">768px</option>
                                    <option value="1024">1024px</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label>⚡ Steps</label>
                                <input type="number" id="video-steps" value="25" min="10" max="50">
                            </div>
                            
                            <div class="form-group">
                                <label>🎛️ Guidance</label>
                                <input type="number" id="video-guidance" value="7.5" min="1" max="20" step="0.5">
                            </div>
                            
                            <div class="form-group">
                                <label>🎬 FPS</label>
                                <select id="video-fps">
                                    <option value="4">4 FPS</option>
                                    <option value="8" selected>8 FPS</option>
                                    <option value="12">12 FPS</option>
                                    <option value="24">24 FPS</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label>🔧 Use LORA</label>
                                <select id="video-use-lora">
                                    <option value="true" selected>Yes (Enhanced)</option>
                                    <option value="false">No (Base Model)</option>
                                </select>
                            </div>
                        </div>
                    </div>
                </div>
                
                <button type="submit" class="generate-btn" id="videoBtn">
                    🎬 Generate Video
                </button>
            </form>
            
            <div id="video-status" class="status"></div>
            <div id="video-result" class="result"></div>
        </div>
        
        <!-- Gallery -->
        <div class="tab-content active">
            <h2 style="margin-bottom: 20px; color: #333;">📸 Recent Creations</h2>
            <div id="gallery" class="gallery"></div>
        </div>
    </div>
    
    <script>
        let currentTab = 'image';
        
        function switchTab(tab) {
            // Update tab buttons
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelector(`[onclick="switchTab('${tab}')"]`).classList.add('active');
            
            // Update tab content
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.getElementById(`${tab}-tab`).classList.add('active');
            
            currentTab = tab;
        }
        
        // Image generation
        document.getElementById('imageForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const btn = document.getElementById('imageBtn');
            const status = document.getElementById('image-status');
            const result = document.getElementById('image-result');
            
            const data = {
                prompt: document.getElementById('image-prompt').value,
                negative_prompt: document.getElementById('image-negative').value,
                model: document.getElementById('image-model').value,
                num_images: parseInt(document.getElementById('image-count').value),
                width: parseInt(document.getElementById('image-width').value),
                height: parseInt(document.getElementById('image-height').value),
                steps: parseInt(document.getElementById('image-steps').value),
                guidance: parseFloat(document.getElementById('image-guidance').value),
                lora_scale: parseFloat(document.getElementById('image-lora').value),
                seed: document.getElementById('image-seed').value ? parseInt(document.getElementById('image-seed').value) : null,
                use_lora: document.getElementById('image-use-lora').value === 'true'
            };
            
            btn.disabled = true;
            btn.textContent = `🔄 Generating ${data.num_images} image(s)...`;
            status.className = 'status show info';
            status.textContent = `🎨 Creating ${data.num_images} image(s)... This may take ${data.num_images * 30}-${data.num_images * 60} seconds.`;
            result.innerHTML = '';
            
            try {
                const response = await fetch('/generate/image', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                const res = await response.json();
                
                if (res.success) {
                    status.className = 'status show success';
                    status.textContent = `✅ ${res.count} image(s) generated successfully!`;
                    
                    let imagesHtml = '';
                    res.urls.forEach((url, index) => {
                        imagesHtml += `
                            <div style="display: inline-block; margin: 10px;">
                                <img src="${url}" alt="Generated Image ${index + 1}" onclick="window.open('${url}')" style="max-width: 300px; cursor: pointer; border-radius: 10px;">
                                <p style="text-align: center; margin-top: 5px; font-size: 12px; color: #666;">${res.filenames[index]}</p>
                            </div>
                        `;
                    });
                    
                    result.innerHTML = `
                        <div style="text-align: center;">
                            ${imagesHtml}
                            <p style="margin-top: 15px; color: #666;">
                                Model: ${res.model} | Size: ${res.settings.size} | Steps: ${res.settings.steps} | LORA: ${res.settings.use_lora ? 'Enabled' : 'Disabled'}
                            </p>
                        </div>
                    `;
                    loadGallery();
                } else {
                    throw new Error(res.message || 'Generation failed');
                }
            } catch (err) {
                status.className = 'status show error';
                status.textContent = '❌ Error: ' + err.message;
            } finally {
                btn.disabled = false;
                btn.textContent = '🚀 Generate Image';
            }
        });
        
        // Video generation
        document.getElementById('videoForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const btn = document.getElementById('videoBtn');
            const status = document.getElementById('video-status');
            const result = document.getElementById('video-result');
            
            const data = {
                prompt: document.getElementById('video-prompt').value,
                negative_prompt: document.getElementById('video-negative').value,
                duration: parseFloat(document.getElementById('video-duration').value),
                width: parseInt(document.getElementById('video-width').value),
                height: parseInt(document.getElementById('video-height').value),
                steps: parseInt(document.getElementById('video-steps').value),
                guidance: parseFloat(document.getElementById('video-guidance').value),
                fps: parseInt(document.getElementById('video-fps').value),
                use_lora: document.getElementById('video-use-lora').value === 'true'
            };
            
            btn.disabled = true;
            btn.textContent = '🔄 Generating...';
            status.className = 'status show info';
            status.textContent = `🎬 Creating ${data.duration}s video... This may take 2-5 minutes.`;
            result.innerHTML = '';
            
            try {
                const response = await fetch('/generate/video', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                const res = await response.json();
                
                if (res.success) {
                    status.className = 'status show success';
                    status.textContent = '✅ Video generated successfully!';
                    result.innerHTML = `
                        <video controls onclick="window.open('${res.url}')">
                            <source src="${res.url}" type="video/mp4">
                        </video>
                        <p style="margin-top: 15px; color: #666;">
                            <strong>${res.filename}</strong><br>
                            Model: ${res.model} | Size: ${res.settings.size} | Duration: ${res.settings.duration} | FPS: ${res.settings.fps} | LORA: ${res.settings.use_lora ? 'Enabled' : 'Disabled'}
                        </p>
                    `;
                    loadGallery();
                } else {
                    throw new Error(res.message || 'Generation failed');
                }
            } catch (err) {
                status.className = 'status show error';
                status.textContent = '❌ Error: ' + err.message;
            } finally {
                btn.disabled = false;
                btn.textContent = '🎬 Generate Video';
            }
        });
        
        // Load gallery
        async function loadGallery() {
            try {
                const response = await fetch('/outputs/list');
                const data = await response.json();
                
                let galleryHTML = '';
                
                // Add recent images
                data.images.slice(0, 8).forEach(img => {
                    galleryHTML += `
                        <div class="gallery-item" onclick="window.open('/outputs/images/${img.name}')">
                            <img src="/outputs/images/${img.name}" alt="Generated Image">
                            <div class="overlay">IMG</div>
                        </div>
                    `;
                });
                
                // Add recent videos
                data.videos.slice(0, 4).forEach(vid => {
                    galleryHTML += `
                        <div class="gallery-item" onclick="window.open('/outputs/videos/${vid.name}')">
                            <video muted loop onmouseover="this.play()" onmouseout="this.pause()">
                                <source src="/outputs/videos/${vid.name}" type="video/mp4">
                            </video>
                            <div class="overlay">VID</div>
                        </div>
                    `;
                });
                
                document.getElementById('gallery').innerHTML = galleryHTML;
            } catch (err) {
                console.error('Failed to load gallery:', err);
            }
        }
        
        // Load gallery on page load
        loadGallery();
        
        // Auto-refresh gallery every 30 seconds
        setInterval(loadGallery, 30000);
        
        // Setup token function
        async function setupToken() {
            const token = document.getElementById('hf-token-input').value;
            if (!token) {
                alert('Please enter a token');
                return;
            }
            
            try {
                const response = await fetch('/setup/hf-token', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({token: token})
                });
                
                const result = await response.json();
                if (result.success) {
                    alert('✅ Token saved successfully!');
                    document.getElementById('token-setup').style.display = 'none';
                    location.reload();
                } else {
                    alert('❌ Error: ' + result.message);
                }
            } catch (err) {
                alert('❌ Error saving token: ' + err.message);
            }
        }
        
        // Check if token is needed
        fetch('/')
            .then(r => r.json())
            .then(data => {
                if (data.huggingface_status.includes('❌')) {
                    document.getElementById('token-setup').style.display = 'block';
                }
            });
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🔥 Uncensored AI Generator for RunPod")
    logger.info("=" * 60)
    logger.info(f"📍 URL: http://0.0.0.0:8000")
    logger.info(f"📚 API Docs: http://0.0.0.0:8000/docs")
    logger.info(f"🎨 Web UI: http://0.0.0.0:8000/ui")
    logger.info("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)