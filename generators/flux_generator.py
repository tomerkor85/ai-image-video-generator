import torch
import logging
from diffusers import FluxPipeline
from PIL import Image
import os
from typing import Optional

logger = logging.getLogger(__name__)

class FluxGenerator:
    def __init__(self):
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "black-forest-labs/FLUX.1-schnell"  # More permissive version
        self.lora_path = "models/naya2.safetensors"
        self._loaded = False
        
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded and self.pipeline is not None
        
    async def load_model(self):
        """Load FLUX model with LORA - UNCENSORED"""
        try:
            logger.info(f"Loading FLUX model on {self.device}")
            
            # Get Hugging Face token
            hf_token = os.environ.get("HUGGINGFACE_TOKEN")
            
            # Load FLUX model WITHOUT safety checker
            self.pipeline = FluxPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                token=hf_token,
                safety_checker=None,  # DISABLE SAFETY CHECKER
                requires_safety_checker=False,  # NO CENSORSHIP
                use_safetensors=True,
                trust_remote_code=True  # Allow custom code execution
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Load LORA if exists
            if os.path.exists(self.lora_path):
                logger.info(f"Loading LORA from {self.lora_path}")
                self.pipeline.load_lora_weights(self.lora_path)
                logger.info("LORA loaded successfully")
            else:
                logger.warning(f"LORA file not found at {self.lora_path}")
            
            # Enable memory optimizations
            self.pipeline.enable_attention_slicing()
            if self.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
                
            # Try to enable xformers for better performance
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("XFormers enabled for better performance")
            except:
                logger.info("XFormers not available, using default attention")
            
            self._loaded = True
            logger.info("FLUX model loaded successfully - UNCENSORED MODE")
            
        except Exception as e:
            logger.error(f"Error loading FLUX model: {e}")
            raise e
    
    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        lora_scale: float = 1.0
    ) -> Image.Image:
        """Generate uncensored image with FLUX"""
        try:
            if not self.is_loaded():
                raise RuntimeError("Model not loaded")
            
            # Set seed for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            logger.info(f"Generating uncensored image: {prompt[:100]}...")
            
            # Generate image WITHOUT any content filtering
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    # NO SAFETY CHECKER - UNCENSORED
                )
            
            image = result.images[0]
            
            logger.info("Uncensored image generated successfully")
            return image
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            raise e
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("FLUX model unloaded")