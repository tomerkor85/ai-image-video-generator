import torch
import logging
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import numpy as np
from typing import Optional, List
import os
import cv2

logger = logging.getLogger(__name__)
 
class WanGenerator:
    def __init__(self):
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
        self.lora_high_path = "naya_wan_lora/lora_t2v_A14B_separate_high.safetensors"
        self.lora_low_path = "naya_wan_lora/lora_t2v_A14B_separate_low.safetensors"
        
    async def load_model(self, lora_type: str = "high"):
        """Load WAN model with LORA"""
        try:
            logger.info(f"Loading WAN model on {self.device}")
            
            # Load base WAN model
            self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                safety_checker=None,  # Disable safety checker for uncensored generation
                requires_safety_checker=False
            )
            
            # Move to device if not using device_map
            if self.device != "cuda" or not torch.cuda.is_available():
                self.pipeline = self.pipeline.to(self.device)
            
            # Load LORA based on type
            lora_loaded = False
            if lora_type == "high" and os.path.exists(self.lora_high_path):
                logger.info(f"Loading HIGH noise LORA from {self.lora_high_path}")
                self.pipeline.load_lora_weights(self.lora_high_path)
                logger.info("HIGH noise LORA loaded successfully")
                lora_loaded = True
            elif lora_type == "low" and os.path.exists(self.lora_low_path):
                logger.info(f"Loading LOW noise LORA from {self.lora_low_path}")
                self.pipeline.load_lora_weights(self.lora_low_path)
                logger.info("LOW noise LORA loaded successfully")
                lora_loaded = True
            else:
                # Fallback: try to load any available LORA
                if os.path.exists(self.lora_high_path):
                    logger.info(f"Fallback: Loading HIGH noise LORA from {self.lora_high_path}")
                    self.pipeline.load_lora_weights(self.lora_high_path)
                    logger.info("HIGH noise LORA loaded successfully")
                    lora_loaded = True
                elif os.path.exists(self.lora_low_path):
                    logger.info(f"Fallback: Loading LOW noise LORA from {self.lora_low_path}")
                    self.pipeline.load_lora_weights(self.lora_low_path)
                    logger.info("LOW noise LORA loaded successfully")
                    lora_loaded = True
            
            if not lora_loaded:
                logger.warning(f"No LORA files found at {self.lora_high_path} or {self.lora_low_path}")
            
            # Enable memory efficient attention
            self.pipeline.enable_attention_slicing()
            if self.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
            
            logger.info("WAN model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading WAN model: {e}")
            raise e
    
    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_frames: int = 16,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        lora_scale: float = 1.0
    ) -> List[Image.Image]:
        """Generate video frames with WAN"""
        try:
            if self.pipeline is None:
                raise RuntimeError("Model not loaded")
            
            # Set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            logger.info(f"Generating video: {prompt[:100]}...")
            
            # Create a base image from the prompt (you might want to use a text-to-image model here)
            # For now, we'll create a simple colored image
            base_image = self._create_base_image(width, height, prompt)
            
            # Generate video frames
            with torch.no_grad():
                result = self.pipeline(
                    image=base_image,
                    decode_chunk_size=8,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None
                )
            
            frames = result.frames[0]
            
            # Convert to PIL Images
            pil_frames = []
            for frame in frames:
                # Convert tensor to PIL Image
                if isinstance(frame, torch.Tensor):
                    frame = frame.cpu().numpy()
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                pil_image = Image.fromarray(frame)
                pil_frames.append(pil_image)
            
            # Apply LORA scaling if needed
            if lora_scale != 1.0 and hasattr(self.pipeline, 'fuse_lora'):
                self.pipeline.fuse_lora(lora_scale=lora_scale)
            
            logger.info(f"Video generated successfully with {len(pil_frames)} frames")
            return pil_frames
            
        except Exception as e:
            logger.error(f"Error generating video: {e}")
            raise e
    
    def _create_base_image(self, width: int, height: int, prompt: str) -> Image.Image:
        """Create a base image for video generation"""
        # This is a simple implementation - you might want to use a text-to-image model
        # to create a more relevant base image from the prompt
        
        # Create a gradient image based on prompt keywords
        colors = {
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'purple': (128, 0, 128),
            'orange': (255, 165, 0),
            'pink': (255, 192, 203),
            'black': (0, 0, 0),
            'white': (255, 255, 255)
        }
        
        # Find color keywords in prompt
        prompt_lower = prompt.lower()
        base_color = (128, 128, 128)  # Default gray
        
        for color_name, color_value in colors.items():
            if color_name in prompt_lower:
                base_color = color_value
                break
        
        # Create gradient image
        image = Image.new('RGB', (width, height))
        pixels = []
        
        for y in range(height):
            for x in range(width):
                # Create a simple gradient
                intensity = (x + y) / (width + height)
                r = int(base_color[0] * intensity)
                g = int(base_color[1] * intensity)
                b = int(base_color[2] * intensity)
                pixels.append((r, g, b))
        
        image.putdata(pixels)
        return image
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("WAN model unloaded")
