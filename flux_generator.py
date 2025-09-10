import torch
import logging
from diffusers import FluxPipeline, FluxInpaintPipeline
from PIL import Image
import numpy as np
from typing import Optional, Union
import os

logger = logging.getLogger(__name__)

class FluxGenerator:
    def __init__(self):
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "black-forest-labs/FLUX.1-dev"
        self.lora_path = "models/flux_naya.safetensors"
        
    async def load_model(self):
        """Load FLUX model with LORA"""
        try:
            logger.info(f"Loading FLUX model on {self.device}")
            
            # Get Hugging Face token from environment
            hf_token = os.environ.get("HUGGINGFACE_TOKEN")
            
            # Load base FLUX model
            self.pipeline = FluxPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                safety_checker=None,  # Disable safety checker for uncensored generation
                requires_safety_checker=False,
                token=hf_token  # Use authentication token
            )
            
            # Move to device if not using device_map
            if self.device != "cuda" or not torch.cuda.is_available():
                self.pipeline = self.pipeline.to(self.device)
            
            # Load LORA if exists
            if os.path.exists(self.lora_path):
                logger.info(f"Loading LORA from {self.lora_path}")
                self.pipeline.load_lora_weights(self.lora_path)
                logger.info("LORA loaded successfully")
            else:
                logger.warning(f"LORA file not found at {self.lora_path}")
            
            # Enable memory efficient attention
            self.pipeline.enable_attention_slicing()
            if self.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
            
            logger.info("FLUX model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading FLUX model: {e}")
            raise e
    
    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        lora_scale: float = 1.0
    ) -> Image.Image:
        """Generate image with FLUX"""
        try:
            if self.pipeline is None:
                raise RuntimeError("Model not loaded")
            
            # Set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Prepare prompt
            full_prompt = prompt
            if negative_prompt:
                full_prompt = f"{prompt}, {negative_prompt}"
            
            logger.info(f"Generating image: {prompt[:100]}...")
            
            # Generate image
            with torch.no_grad():
                result = self.pipeline(
                    prompt=full_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None
                )
            
            image = result.images[0]
            
            # Apply LORA scaling if needed
            if lora_scale != 1.0 and hasattr(self.pipeline, 'fuse_lora'):
                self.pipeline.fuse_lora(lora_scale=lora_scale)
            
            logger.info("Image generated successfully")
            return image
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            raise e
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("FLUX model unloaded")
