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
        
        # Available uncensored models - you can switch between them
        self.available_models = {
            "flux_dev": {
                "id": "black-forest-labs/FLUX.1-dev",
                "description": "FLUX.1-dev - Best with NAYA2 LORA (requires HF token)",
                "type": "huggingface",
                "requires_token": True
            },
            "flux_schnell": {
                "id": "black-forest-labs/FLUX.1-schnell", 
                "description": "FLUX.1-schnell - Fast generation (requires HF token)",
                "type": "huggingface",
                "requires_token": True
            },
            "sdxl_base": {
                "id": "stabilityai/stable-diffusion-xl-base-1.0", 
                "description": "SDXL Base - very permissive for adult content",
                "type": "huggingface",
                "requires_token": False
            },
            "realistic_vision": {
                "id": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
                "description": "Realistic Vision V6 - best for photorealistic adult content",
                "type": "huggingface",
                "requires_token": False
            },
            "dreamshaper": {
                "id": "Lykon/DreamShaper",
                "description": "DreamShaper - versatile for various adult styles",
                "type": "huggingface",
                "requires_token": False
            },
            "deliberate": {
                "id": "XpucT/Deliberate",
                "description": "Deliberate - popular for detailed adult content",
                "type": "huggingface",
                "requires_token": False
            },
            # CivitAI models (need manual download)
            "civitai_custom": {
                "id": "models/civitai_model.safetensors",
                "description": "Custom CivitAI model (place in models/ folder)",
                "type": "local",
                "requires_token": False
            }
        }
        
        # Default model
        self.current_model = "flux_dev"
        self.model_id = self.available_models[self.current_model]["id"]
        self.lora_path = "models/naya2.safetensors"
        self._loaded = False
        
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded and self.pipeline is not None
    
    def switch_model(self, model_key: str):
        """Switch to a different model"""
        if model_key in self.available_models:
            if self.is_loaded():
                self.unload_model()
            self.current_model = model_key
            self.model_id = self.available_models[model_key]["id"]
            logger.info(f"Switched to model: {self.available_models[model_key]['description']}")
        else:
            logger.error(f"Model {model_key} not found in available models")
    
    def get_available_models(self):
        """Get list of available models"""
        return self.available_models
        
    async def load_model(self):
        """Load FLUX model with LORA - UNCENSORED"""
        try:
            logger.info(f"Loading FLUX model on {self.device}")
            logger.info(f"Current model: {self.available_models[self.current_model]['description']}")
            
            # Get Hugging Face token (with fallbacks)
            hf_token = (
                os.environ.get("HUGGINGFACE_TOKEN") or 
                os.environ.get("HF_TOKEN") or
                None
            )
            
            if not hf_token and model_info.get("requires_token", False):
                logger.warning("⚠️ No Hugging Face token found - some models may not be accessible")
                logger.info("💡 Set token with: export HUGGINGFACE_TOKEN='your_token'")

            
            model_info = self.available_models[self.current_model]
            
            if model_info["type"] == "local":
                # Load local CivitAI model
                if os.path.exists(self.model_id):
                    from diffusers import StableDiffusionXLPipeline
                    self.pipeline = StableDiffusionXLPipeline.from_single_file(
                        self.model_id,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=True
                    )
                else:
                    raise FileNotFoundError(f"Local model not found: {self.model_id}")
            elif "flux" in self.current_model:
                # Load FLUX model (dev or schnell)
                try:
                    from diffusers import FluxPipeline
                    self.pipeline = FluxPipeline.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        token=hf_token,
                        safety_checker=None,  # DISABLE SAFETY CHECKER
                        requires_safety_checker=False,  # NO CENSORSHIP
                        use_safetensors=True,
                        trust_remote_code=True
                    )
                except Exception as e:
                    logger.error(f"Failed to load FLUX model: {e}")
                    logger.info("Falling back to SDXL...")
                    # Fallback to SDXL
                    from diffusers import StableDiffusionXLPipeline
                    self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        token=hf_token,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=True,
                        trust_remote_code=True
                    )
                    self.current_model = "sdxl_base"
                    self.model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    token=hf_token,
                    safety_checker=None,  # DISABLE SAFETY CHECKER
                    requires_safety_checker=False,  # NO CENSORSHIP
                    use_safetensors=True,
                    trust_remote_code=True
                )
            else:
                # Load HuggingFace model WITHOUT safety checker - all as SDXL
                from diffusers import StableDiffusionXLPipeline
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    token=hf_token,
                    safety_checker=None,  # DISABLE SAFETY CHECKER
                    requires_safety_checker=False,  # NO CENSORSHIP
                    use_safetensors=True,
                    trust_remote_code=True
                )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Load LORA if exists - NOW WORKS WITH SDXL!
            if os.path.exists(self.lora_path):
                logger.info(f"Loading LORA from {self.lora_path}")
                try:
                    if "flux" in self.current_model:
                        # FLUX models support LORA differently
                        self.pipeline.load_lora_weights(self.lora_path)
                        logger.info("✅ NAYA2 LORA loaded successfully for FLUX - UNCENSORED MODE!")
                    else:
                        # SDXL and other models
                        self.pipeline.load_lora_weights(self.lora_path)
                        logger.info("✅ NAYA2 LORA loaded successfully - UNCENSORED MODE!")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load LORA: {e}")
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
        lora_scale: float = 1.0,
        use_lora: bool = True
    ) -> Image.Image:
        """Generate uncensored image with FLUX"""
        try:
            if not self.is_loaded():
                raise RuntimeError("Model not loaded")
            
            # Set seed for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            lora_status = "with LORA" if use_lora and os.path.exists(self.lora_path) else "without LORA"
            logger.info(f"Generating uncensored image {lora_status}: {prompt[:100]}...")
            
            # Temporarily disable LORA if requested
            if not use_lora and hasattr(self.pipeline, 'unload_lora_weights'):
                try:
                    self.pipeline.unload_lora_weights()
                except:
                    pass
            elif use_lora and os.path.exists(self.lora_path):
                try:
                    self.pipeline.load_lora_weights(self.lora_path)
                except:
                    pass
            
            # Generate image WITHOUT any content filtering
            with torch.no_grad():
                if "flux" in self.current_model:
                    # FLUX models have different parameters
                    result = self.pipeline(
                        prompt=prompt,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        joint_attention_kwargs={"scale": lora_scale} if use_lora else None,
                        # NO SAFETY CHECKER - UNCENSORED
                    )
                else:
                    # SDXL and other models
                    result = self.pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        cross_attention_kwargs={"scale": lora_scale} if use_lora else None,
                        # NO SAFETY CHECKER - UNCENSORED
                    )
            
            image = result.images[0]
            
            logger.info(f"Uncensored image generated successfully {lora_status}")
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