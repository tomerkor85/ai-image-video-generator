import torch
import logging
import gc
from diffusers import FluxPipeline, StableDiffusionXLPipeline, DiffusionPipeline
from peft import LoraConfig, get_peft_model
from PIL import Image
import os
from typing import Optional
from huggingface_hub import hf_hub_download
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

def clear_gpu_memory():
    """Aggressively clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

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
                "lora_path": "models",
                "lora_weight_name": "naya2.safetensors",
                "pipeline_class": "FluxPipeline",
                "recommended_steps": 50,
                "recommended_guidance": 3.5
            },
            "flux_schnell": {
                "id": "black-forest-labs/FLUX.1-schnell", 
                "description": "FLUX.1-schnell - Fast generation (4 steps)",
                "type": "flux",
                "requires_token": True,
                "supports_lora": True,
                "lora_path": "models",
                "lora_weight_name": "naya2.safetensors",
                "pipeline_class": "FluxPipeline",
                "recommended_steps": 4,
                "recommended_guidance": 0.0,
                "max_sequence_length": 256
            },
            "sdxl_base": {
                "id": "stabilityai/stable-diffusion-xl-base-1.0", 
                "description": "SDXL Base - Very permissive",
                "type": "sdxl",
                "requires_token": False,
                "supports_lora": True,
                "lora_path": "models/naya2.safetensors",
                "pipeline_class": "StableDiffusionXLPipeline",
                "recommended_steps": 25,
                "recommended_guidance": 7.5
            },
            "realistic_vision": {
                "id": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
                "description": "Realistic Vision V6 - Photorealistic",
                "type": "sdxl",
                "requires_token": False,
                "supports_lora": True,
                "lora_path": "models/naya2.safetensors",
                "pipeline_class": "StableDiffusionXLPipeline",
                "recommended_steps": 30,
                "recommended_guidance": 8.0
            },
            "civitai_custom": {
                "id": "custom",
                "description": "Custom CivitAI Model",
                "type": "civitai",
                "requires_token": False,
                "supports_lora": True,
                "lora_path": "models/naya2.safetensors",
                "pipeline_class": "StableDiffusionXLPipeline",
                "recommended_steps": 25,
                "recommended_guidance": 7.5
            }
        }
        
        self.current_model = "flux_dev"
        self.current_civitai_url = None
        self._loaded = False
        self.lora_loaded = False
        
        # Memory optimization settings
        self.memory_optimizations = {
            "enable_sequential_cpu_offload": True,
            "enable_model_cpu_offload": True,
            "enable_attention_slicing": True,
            "enable_vae_slicing": True,
            "enable_vae_tiling": True,
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32
        }
        
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
        """Load model with official FLUX LORA loading method"""
        try:
            # Clear memory before loading
            clear_gpu_memory()
            
            model_info = self.available_models[self.current_model]
            logger.info(f"📥 Loading {model_info['description']}...")
            logger.info(f"🔧 GPU Memory before loading: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB total")
            if torch.cuda.is_available():
                logger.info(f"🔧 GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
                logger.info(f"🔧 GPU Memory cached: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
            
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
                
            # Handle FLUX models - OFFICIAL WAY
            elif model_info["type"] == "flux":
                logger.info(f"🔥 Loading FLUX model: {model_info['id']}")
                logger.info("🔧 Using aggressive memory optimizations for FLUX...")
                
                # Load FLUX pipeline with official method
                self.pipeline = FluxPipeline.from_pretrained(
                    model_info["id"],
                    torch_dtype=self.memory_optimizations["torch_dtype"],
                    token=hf_token,
                    use_safetensors=True,
                    low_cpu_mem_usage=self.memory_optimizations["low_cpu_mem_usage"],
                    device_map="balanced"  # Automatic device mapping
                )
                
                # Don't move to device manually - let device_map handle it
                logger.info("🔧 Model loaded with automatic device mapping")
                
                # Enable ALL memory optimizations for FLUX
                logger.info("🔧 Enabling memory optimizations...")
                
                # Use sequential CPU offload instead of model CPU offload for better memory usage
                if self.memory_optimizations["enable_sequential_cpu_offload"]:
                    self.pipeline.enable_sequential_cpu_offload()
                    logger.info("✅ Sequential CPU offload enabled")
                
                if self.memory_optimizations["enable_vae_slicing"]:
                    self.pipeline.enable_vae_slicing()
                    logger.info("✅ VAE slicing enabled")
                    
                if self.memory_optimizations["enable_vae_tiling"]:
                    self.pipeline.enable_vae_tiling()
                    logger.info("✅ VAE tiling enabled")
                
                if self.memory_optimizations["enable_attention_slicing"]:
                    self.pipeline.enable_attention_slicing()
                    logger.info("✅ Attention slicing enabled")
                
                # Additional FLUX-specific optimizations
                try:
                    # Enable memory efficient attention if available
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("✅ XFormers memory efficient attention enabled")
                except Exception as e:
                    logger.info(f"⚠️ XFormers not available: {e}")
                
                # Clear memory after loading
                clear_gpu_memory()
                
            # Handle SDXL models
            elif model_info["type"] == "sdxl":
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_info["id"],
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    token=hf_token,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True
                )
                
                # Move to device
                self.pipeline = self.pipeline.to(self.device)
                
                # Enable memory optimizations for SDXL
                if self.memory_optimizations["enable_attention_slicing"]:
                    self.pipeline.enable_attention_slicing()
                if self.memory_optimizations["enable_model_cpu_offload"] and self.device == "cuda":
                    self.pipeline.enable_model_cpu_offload()
            
            # Load NAYA2 LORA - OFFICIAL METHOD
            self.lora_loaded = False
            if model_info["supports_lora"]:
                lora_path = model_info.get("lora_path", "models")
                lora_weight_name = model_info.get("lora_weight_name", "naya2.safetensors")
                full_lora_path = os.path.join(lora_path, lora_weight_name)
                
                if os.path.exists(full_lora_path):
                    try:
                        logger.info(f"📥 Loading NAYA2 LORA: {full_lora_path}")
                        
                        if model_info["type"] == "flux":
                            # FLUX LORA loading - OFFICIAL METHOD
                            self.pipeline.load_lora_weights(
                                lora_path, 
                                weight_name=lora_weight_name,
                                adapter_name="naya2",
                                low_cpu_mem_usage=True
                            )
                            # Set adapter strength - OFFICIAL METHOD
                            self.pipeline.set_adapters("naya2", 0.85)
                            logger.info("✅ FLUX NAYA2 LORA loaded successfully!")
                            self.lora_loaded = True
                            
                        else:
                            # SDXL LORA loading method
                            self.pipeline.load_lora_weights(lora_path, weight_name=lora_weight_name)
                            logger.info("✅ SDXL NAYA2 LORA loaded successfully!")
                            self.lora_loaded = True
                        
                    except Exception as e:
                        logger.warning(f"⚠️ NAYA2 LORA loading failed: {e}")
                        logger.info("🔄 Continuing with base model (no LORA)...")
                        self.lora_loaded = False
                else:
                    logger.warning(f"⚠️ NAYA2 LORA file not found: {full_lora_path}")
                    logger.info("💡 To use LORA, place naya2.safetensors in the models/ directory")
            
            # Final memory cleanup
            clear_gpu_memory()
            
            self._loaded = True
            lora_status = "with NAYA2 LORA" if self.lora_loaded else "base model"
            logger.info(f"✅ Model loaded successfully ({lora_status})")
            
            # Log final memory usage
            if torch.cuda.is_available():
                logger.info(f"🔧 Final GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
                logger.info(f"🔧 Final GPU Memory cached: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            logger.error(f"🔧 GPU Memory at error: {torch.cuda.memory_allocated() / 1024**3:.1f} GB allocated")
            self._loaded = False
            self.lora_loaded = False
            # Clear memory on error
            clear_gpu_memory()
            raise e
    
    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = None,
        guidance_scale: float = None,
        seed: Optional[int] = None,
        lora_scale: float = 1.0,
        use_lora: bool = True
    ) -> Image.Image:
        """Generate image with OFFICIAL FLUX parameters"""
        try:
            if not self.is_loaded():
                raise RuntimeError("❌ Model not loaded")
            
            model_info = self.available_models[self.current_model]
            
            # Use recommended parameters if not specified
            if num_inference_steps is None:
                num_inference_steps = model_info.get("recommended_steps", 25)
            if guidance_scale is None:
                guidance_scale = model_info.get("recommended_guidance", 7.5)
            
            # Set seed
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Handle LORA usage
            effective_lora = use_lora and self.lora_loaded
            lora_status = "with NAYA2 LORA" if effective_lora else "base model"
            
            logger.info(f"🎨 Generating ({lora_status}): {prompt[:50]}...")
            
            # Generate based on model type - OFFICIAL METHODS
            with torch.no_grad():
                # Clear memory before generation
                clear_gpu_memory()
                
                if model_info["type"] == "flux":
                    # FLUX generation - OFFICIAL METHOD
                    generation_kwargs = {
                        "prompt": prompt,
                        "height": height,
                        "width": width,
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "generator": generator,
                    }
                    
                    # Add max_sequence_length for schnell
                    if "max_sequence_length" in model_info:
                        generation_kwargs["max_sequence_length"] = model_info["max_sequence_length"]
                    
                    # Update LORA adapter strength if needed
                    if effective_lora:
                        self.pipeline.set_adapters("naya2", lora_scale)
                    
                    result = self.pipeline(**generation_kwargs)
                    
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
                    
                    # Add LORA scale for SDXL
                    if effective_lora:
                        generation_kwargs["cross_attention_kwargs"] = {"scale": lora_scale}
                    
                    result = self.pipeline(**generation_kwargs)
            
            image = result.images[0]
            
            # Clear memory after generation
            clear_gpu_memory()
            
            logger.info(f"✅ Image generated successfully ({lora_status})")
            return image
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            # Clear memory on error
            clear_gpu_memory()
            raise e
    
    def unload_model(self):
        """Clean unload with proper LORA cleanup"""
        if self.pipeline is not None:
            # Unload LORA weights properly for FLUX
            if hasattr(self.pipeline, 'unload_lora_weights'):
                try:
                    # For FLUX models - use reset_to_overwritten_params=True
                    model_info = self.available_models[self.current_model]
                    if model_info["type"] == "flux":
                        self.pipeline.unload_lora_weights(reset_to_overwritten_params=True)
                    else:
                        self.pipeline.unload_lora_weights()
                    logger.info("🗑️ LORA weights unloaded")
                except Exception as e:
                    logger.warning(f"⚠️ LORA unload warning: {e}")
            
            del self.pipeline
            self.pipeline = None
            self._loaded = False
            self.lora_loaded = False
            
            # Aggressive memory cleanup
            clear_gpu_memory()
            
            logger.info("🗑️ Model unloaded")