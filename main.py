import os
import asyncio
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from PIL import Image
import base64
import io
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Image & Video Generation API", version="1.0.0")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
flux_pipeline = None
wan_pipeline = None

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    lora_scale: float = 1.0

class VideoRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    width: int = 512
    height: int = 512
    num_frames: int = 16
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    lora_scale: float = 1.0

class GenerationResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global flux_pipeline, wan_pipeline
    
    try:
        logger.info("Loading FLUX model...")
        from flux_generator import FluxGenerator
        flux_pipeline = FluxGenerator()
        await flux_pipeline.load_model()
        
        logger.info("Loading WAN model...")
        from wan_generator import WanGenerator
        wan_pipeline = WanGenerator()
        await wan_pipeline.load_model()
        
        logger.info("All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise e

@app.get("/")
async def root():
    return {"message": "AI Image & Video Generation API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "flux_loaded": flux_pipeline is not None,
        "wan_loaded": wan_pipeline is not None
    }

@app.post("/generate/image", response_model=GenerationResponse)
async def generate_image(request: ImageRequest):
    """Generate image using FLUX with LORA"""
    try:
        if flux_pipeline is None:
            raise HTTPException(status_code=503, detail="FLUX model not loaded")
        
        logger.info(f"Generating image with prompt: {request.prompt}")
        
        # Generate image
        image = await flux_pipeline.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            lora_scale=request.lora_scale
        )
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return GenerationResponse(
            success=True,
            message="Image generated successfully",
            data={
                "image": img_str,
                "prompt": request.prompt,
                "parameters": request.dict()
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/video", response_model=GenerationResponse)
async def generate_video(request: VideoRequest):
    """Generate video using WAN with LORA"""
    try:
        if wan_pipeline is None:
            raise HTTPException(status_code=503, detail="WAN model not loaded")
        
        logger.info(f"Generating video with prompt: {request.prompt}")
        
        # Generate video
        video_frames = await wan_pipeline.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            lora_scale=request.lora_scale
        )
        
        # Convert frames to base64
        frames_base64 = []
        for frame in video_frames:
            buffered = io.BytesIO()
            frame.save(buffered, format="PNG")
            frame_str = base64.b64encode(buffered.getvalue()).decode()
            frames_base64.append(frame_str)
        
        return GenerationResponse(
            success=True,
            message="Video generated successfully",
            data={
                "frames": frames_base64,
                "prompt": request.prompt,
                "parameters": request.dict()
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/info")
async def get_models_info():
    """Get information about loaded models"""
    return {
        "flux": {
            "loaded": flux_pipeline is not None,
            "type": "FLUX 1 DEV with LORA"
        },
        "wan": {
            "loaded": wan_pipeline is not None,
            "type": "WAN 2.2 with LORA"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
