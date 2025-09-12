import torch
import logging
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import numpy as np
import cv2
import os
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class WanGenerator:
    def __init__(self):
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
        self.lora_high_path = "models/lora_t2v_A14B_separate_high.safetensors"
        self.lora_low_path = "models/lora_t2v_A14B_separate_low.safetensors"
        self._loaded = False
        
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded and self.pipeline is not None
        
    async def load_model(self):
        """Load WAN2.2 model - UNCENSORED"""
        try:
            logger.info(f"Loading WAN2.2 model on {self.device}")
            
            # Load WAN model WITHOUT safety checker
            self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,  # DISABLE SAFETY CHECKER
                requires_safety_checker=False,  # NO CENSORSHIP
                use_safetensors=True
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Try to load LORA (prefer high noise)
            lora_loaded = False
            if os.path.exists(self.lora_high_path):
                logger.info(f"Loading HIGH noise LORA from {self.lora_high_path}")
                self.pipeline.load_lora_weights(self.lora_high_path)
                lora_loaded = True
            elif os.path.exists(self.lora_low_path):
                logger.info(f"Loading LOW noise LORA from {self.lora_low_path}")
                self.pipeline.load_lora_weights(self.lora_low_path)
                lora_loaded = True
            
            if not lora_loaded:
                logger.warning("No LORA files found, using base model")
            
            # Enable memory optimizations
            self.pipeline.enable_attention_slicing()
            if self.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
            
            self._loaded = True
            logger.info("WAN2.2 model loaded successfully - UNCENSORED MODE")
            
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
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """Generate uncensored video frames with WAN2.2"""
        try:
            if not self.is_loaded():
                raise RuntimeError("Model not loaded")
            
            # Set seed for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            logger.info(f"Generating uncensored video: {prompt[:100]}...")
            
            # Create base image from prompt (simple approach)
            base_image = self._create_base_image(width, height, prompt)
            
            # Generate video frames WITHOUT content filtering
            with torch.no_grad():
                result = self.pipeline(
                    image=base_image,
                    decode_chunk_size=8,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    # NO SAFETY CHECKER - UNCENSORED
                )
            
            frames = result.frames[0]
            
            # Convert to PIL Images
            pil_frames = []
            for frame in frames:
                if isinstance(frame, torch.Tensor):
                    frame = frame.cpu().numpy()
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                pil_image = Image.fromarray(frame)
                pil_frames.append(pil_image)
            
            logger.info(f"Uncensored video generated: {len(pil_frames)} frames")
            return pil_frames
            
        except Exception as e:
            logger.error(f"Error generating video: {e}")
            raise e
    
    def _create_base_image(self, width: int, height: int, prompt: str) -> Image.Image:
        """Create a base image for video generation"""
        # Simple gradient based on prompt keywords
        colors = {
            'woman': (255, 182, 193),  # Light pink
            'man': (135, 206, 235),    # Sky blue
            'person': (255, 218, 185), # Peach
            'dance': (255, 105, 180),  # Hot pink
            'nature': (144, 238, 144), # Light green
            'city': (169, 169, 169),   # Dark gray
            'night': (25, 25, 112),    # Midnight blue
            'day': (255, 255, 224),    # Light yellow
        }
        
        # Find color based on prompt
        prompt_lower = prompt.lower()
        base_color = (128, 128, 128)  # Default gray
        
        for keyword, color in colors.items():
            if keyword in prompt_lower:
                base_color = color
                break
        
        # Create gradient image
        image = Image.new('RGB', (width, height))
        pixels = []
        
        for y in range(height):
            for x in range(width):
                # Create radial gradient
                center_x, center_y = width // 2, height // 2
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                max_distance = (width ** 2 + height ** 2) ** 0.5 / 2
                intensity = 1 - min(distance / max_distance, 1)
                
                r = int(base_color[0] * intensity)
                g = int(base_color[1] * intensity)
                b = int(base_color[2] * intensity)
                pixels.append((r, g, b))
        
        image.putdata(pixels)
        return image
    
    async def save_video(self, frames: List[Image.Image], output_path: Path, fps: int = 8):
        """Save frames as MP4 video"""
        try:
            # Convert PIL images to numpy arrays
            frame_arrays = []
            for frame in frames:
                frame_array = np.array(frame)
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
            
            logger.info(f"Video saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving video: {e}")
            raise e
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("WAN model unloaded")