import torch
import logging
from diffusers import FluxPipeline, StableDiffusionXLPipeline, DiffusionPipeline
from PIL import Image
import os
from typing import Optional
from huggingface_hub import hf_hub_download
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

class FluxGenerator:
    def __init__(self):
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Available models with full configuration
        self.available_models = {
            "flux_dev": {
                "id": "black-forest-labs/FLUX.1-dev",
                "description": "FLUX.1-dev + NAYA2 LORA - Best quality",
                "type": "flux",
                "requires_token": True,
                "supports_lora": True,
                "lora_path": "models/naya2.safetensors",
                "pipeline_class": "FluxPipeline"
            },
            "flux_schnell": {
                "id": "black-forest-labs/FLUX.1-schnell", 
                "description": "FLUX.1-schnell - Fast generation",
                "type": "flux",
                "requires_token": True,
                "supports_lora": True,
                "lora_path": "models/naya2.safetensors",
                "pipeline_class": "FluxPipeline"
            },
            "sdxl_base": {
                "id": "stabilityai/stable-diffusion-xl-base-1.0", 
                "description": "SDXL Base - Very permissive",
                "type": "sdxl",
                "requires_token": False,
                "supports_lora": True,
                "lora_path": "models/sdxl_lora.safetensors",
                "pipeline_class": "StableDiffusionXLPipeline"
            },
            "realistic_vision": {
                "id": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
                "description": "Realistic Vision V6 - Photorealistic",
                "type": "sdxl",
                "requires_token": False,
                "supports_lora": True,
                "lora_path": "models/realistic_lora.safetensors",
                "pipeline_class": "StableDiffusionXLPipeline"
            },
            "civitai_custom": {
                "id": "custom",
                "description": "Custom CivitAI Model",
                "type": "civitai",
                "requires_token": False,
                "supports_lora": True,
                "lora_path": "models/civitai_lora.safetensors",
                "pipeline_class": "StableDiffusionXLPipeline"
            }
        }
        
        self.current_model = "flux_dev"
        self.current_civitai_url = None
        self._loaded = False
        self.lora_loaded = False
        
    def is_loaded(self) -> bool:
        return self._loaded and self.pipeline is not None
    
    def get_available_models(self):
        return self.available_models
    
    async def download_civitai_model(self, url: str, filename: str = None):
        """Download model from CivitAI URL"""
        try:
            logger.info(f"📥 Downloading CivitAI model from: {url}")
            
            if not filename:
                filename = f"civitai_model_{hash(url) % 10000}.safetensors"
            
            model_path = f"models/{filename}"
            
            # Check if already exists
            if os.path.exists(model_path):
                logger.info(f"✅ Model already exists: {model_path}")
                return model_path
            
            # Download with progress
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            os.makedirs("models", exist_ok=True)
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (1024*1024*50) == 0:  # Log every 50MB
                                logger.info(f"📥 Downloaded: {progress:.1f}%")
            
            logger.info(f"✅ CivitAI model downloaded: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"❌ Failed to download CivitAI model: {e}")
            raise e
    
    async def download_lora(self, model_type: str):
        """Download appropriate LORA for model type"""
        try:
            hf_token = os.environ.get("HUGGINGFACE_TOKEN")
            
            if model_type == "flux" or model_type == "sdxl":
                # Download NAYA2 LORA for FLUX
                lora_path = "models/naya2.safetensors"
                if not os.path.exists(lora_path):
                    logger.info(f"📥 Downloading NAYA2 LORA for {model_type.upper()}...")
                    os.makedirs("models", exist_ok=True)
                    downloaded_path = hf_hub_download(
                        repo_id="tomerkor1985/test",
                        filename="naya2.safetensors",
                        token=hf_token,
                        local_dir="models",
                        local_dir_use_symlinks=False
                    )
                    logger.info(f"✅ NAYA2 LORA downloaded: {lora_path}")
                else:
                    logger.info(f"✅ NAYA2 LORA already exists: {lora_path}")
                return lora_path
                
        except Exception as e:
            logger.warning(f"⚠️ Could not download LORA: {e}")
            return None
    
    def switch_model(self, model_key: str, civitai_url: str = None):
        """Switch to different model"""
        if model_key in self.available_models:
            if self.is_loaded():
                self.unload_model()
            
            self.current_model = model_key
            self.current_civitai_url = civitai_url
            
            logger.info(f"🔄 Switched to: {self.available_models[model_key]['description']}")
        else:
            raise ValueError(f"Model {model_key} not available")
    
    async def load_model(self):
        """Load model with full error handling and LORA support"""
        try:
            model_info = self.available_models[self.current_model]
            logger.info(f"📥 Loading {model_info['description']}...")
            
            hf_token = os.environ.get("HUGGINGFACE_TOKEN")
            
            # Check token requirement
            if model_info["requires_token"] and not hf_token:
                raise RuntimeError("🔑 Hugging Face token required for this model")
            
            # Handle CivitAI models
            if model_info["type"] == "civitai":
                if not self.current_civitai_url:
                    raise RuntimeError("❌ CivitAI URL required for custom model")
                
                model_path = await self.download_civitai_model(self.current_civitai_url)
                self.pipeline = StableDiffusionXLPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True
                )
                
            # Handle FLUX models
            elif model_info["type"] == "flux":
                try:
                    # Try with specific revision to avoid tokenizer issues
                    self.pipeline = FluxPipeline.from_pretrained(
                        model_info["id"],
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        token=hf_token,
                        use_safetensors=True,
                        revision="main"
                    )
                except Exception as e:
                    logger.error(f"❌ FLUX loading failed: {e}")
                    logger.info("🔄 Falling back to SDXL...")
                    # Fallback to SDXL
                    self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=True
                    )
                    model_info["type"] = "sdxl"  # Update type for LORA loading
            
            # Handle SDXL models
            elif model_info["type"] == "sdxl":
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_info["id"],
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    token=hf_token,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True,
                    trust_remote_code=True
                )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Load LORA if supported
            self.lora_loaded = False
            if model_info["supports_lora"]:
                lora_path = await self.download_lora(model_info["type"])
                if lora_path and os.path.exists(lora_path):
                    try:
                        logger.info(f"📥 Loading LORA: {lora_path}")
                        if model_info["type"] == "flux":
                            # FLUX LORA loading
                            self.pipeline.load_lora_weights(lora_path, adapter_name="naya2")
                            self.pipeline.set_adapters("naya2")
                        else:
                            # SDXL LORA loading
                            self.pipeline.load_lora_weights(lora_path)
                        self.lora_loaded = True
                        logger.info("✅ LORA loaded successfully!")
                    except Exception as e:
                        logger.warning(f"⚠️ LORA loading failed: {e}")
                        # Try alternative LORA loading for SDXL
                        if model_info["type"] == "sdxl":
                            try:
                                logger.info("🔄 Trying alternative LORA loading...")
                                from diffusers import LoraLoaderMixin
                                self.pipeline.load_lora_weights(".", weight_name=os.path.basename(lora_path))
                                self.lora_loaded = True
                                logger.info("✅ LORA loaded with alternative method!")
                            except Exception as e2:
                                logger.warning(f"⚠️ Alternative LORA loading also failed: {e2}")
                else:
                    logger.warning(f"⚠️ LORA file not found: {lora_path}")
            
            # Optimizations
            self.pipeline.enable_attention_slicing()
            if self.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
                
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("✅ XFormers enabled")
            except:
                logger.info("⚠️ XFormers not available")
            
            self._loaded = True
            lora_status = "with LORA" if self.lora_loaded else "base model"
            logger.info(f"✅ Model loaded successfully ({lora_status})")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
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
        """Generate image with full LORA control"""
        try:
            if not self.is_loaded():
                raise RuntimeError("❌ Model not loaded")
            
            model_info = self.available_models[self.current_model]
            
            # Set seed
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Handle LORA usage
            effective_lora = use_lora and self.lora_loaded
            lora_status = "with LORA" if effective_lora else "base model"
            
            logger.info(f"🎨 Generating ({lora_status}): {prompt[:50]}...")
            
            # Handle LORA enable/disable
            if model_info["type"] == "flux":
                # FLUX LORA handling
                if effective_lora:
                    try:
                        self.pipeline.set_adapters("naya2")
                        logger.info("🔧 FLUX LORA enabled")
                    except:
                        effective_lora = False
                        logger.warning("⚠️ Failed to enable FLUX LORA")
                else:
                    try:
                        self.pipeline.set_adapters([])  # Disable all adapters
                        logger.info("🔧 FLUX LORA disabled")
                    except:
                        pass
            else:
                # SDXL LORA handling
                if not use_lora and self.lora_loaded:
                    try:
                        self.pipeline.unload_lora_weights()
                        logger.info("🔧 SDXL LORA temporarily disabled")
                    except:
                        pass
                try:
                    self.pipeline.unload_lora_weights()
                    logger.info("🔧 LORA temporarily disabled")
                except:
                    pass
            elif use_lora and not self.lora_loaded and model_info["supports_lora"]:
                # Try to reload LORA
                lora_path = await self.download_lora(model_info["type"])
                if lora_path and os.path.exists(lora_path):
                    try:
                        self.pipeline.load_lora_weights(lora_path)
                        self.lora_loaded = True
                        effective_lora = True
                        logger.info("✅ LORA reloaded")
                    except:
                        pass
            
            # Generate based on model type
            with torch.no_grad():
                if model_info["type"] == "flux":
                    # FLUX generation
                    generation_kwargs = {
                        "prompt": prompt,
                        "height": height,
                        "width": width,
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "generator": generator,
                    }
                    
                    # Add LORA scale only if LORA is active
                    if effective_lora:
                        generation_kwargs["joint_attention_kwargs"] = {"scale": lora_scale}
                    
                    result = self.pipeline(
                        **generation_kwargs
                    )
                else:
                    # SDXL generation
                    generation_kwargs = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "height": height,
                        "width": width,
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "generator": generator,
                    }
                    
                    # Add LORA scale only if LORA is active
                    if effective_lora:
                        generation_kwargs["cross_attention_kwargs"] = {"scale": lora_scale}
                    
                    result = self.pipeline(
                        **generation_kwargs
                    )
            
            image = result.images[0]
            logger.info(f"✅ Image generated successfully ({lora_status})")
            return image
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            raise e
    
    def unload_model(self):
        """Clean unload"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self._loaded = False
            self.lora_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("🗑️ Model unloaded")