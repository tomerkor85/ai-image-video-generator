import torch
import logging
import gc
import psutil
from diffusers import FluxPipeline, StableDiffusionXLPipeline, DiffusionPipeline
from peft import LoraConfig, get_peft_model
from PIL import Image
import os
from typing import Optional
from huggingface_hub import hf_hub_download
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

def clear_gpu_memory(force=False):
    """Aggressively clear GPU memory with optional force mode"""
    if torch.cuda.is_available():
        # Force garbage collection first
        gc.collect()
        
        # Clear PyTorch cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Force mode - more aggressive cleanup
        if force:
            # Try to clear all cached memory
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            
            # Multiple rounds of cleanup
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

def get_memory_info():
    """Get detailed memory information"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated()
    cached = torch.cuda.memory_reserved()
    free_gpu = gpu_memory - allocated
    
    # System RAM
    ram = psutil.virtual_memory()
    
    return {
        "gpu_total_gb": gpu_memory / 1024**3,
        "gpu_allocated_gb": allocated / 1024**3,
        "gpu_cached_gb": cached / 1024**3,
        "gpu_free_gb": free_gpu / 1024**3,
        "ram_total_gb": ram.total / 1024**3,
        "ram_available_gb": ram.available / 1024**3,
        "ram_used_percent": ram.percent
    }

def check_memory_requirements(model_type: str) -> bool:
    """Check if we have enough memory for the model"""
    memory_info = get_memory_info()
    free_gb = memory_info.get("gpu_free_gb", 0)
    
    # Estimated memory requirements (GB)
    requirements = {
        "flux_dev": 16.0,      # FLUX.1-dev is very large
        "flux_schnell": 12.0,  # FLUX.1-schnell is smaller
        "sdxl_base": 8.0,      # SDXL models
        "realistic_vision": 8.0,
        "civitai_custom": 8.0
    }
    
    required = requirements.get(model_type, 8.0)
    has_enough = free_gb >= required
    
    logger.info(f"🔧 Memory check: {free_gb:.1f}GB free, {required:.1f}GB required for {model_type}")
    
    return has_enough

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
            "enable_sequential_cpu_offload": False,  # Will enable conditionally
            "enable_model_cpu_offload": False,       # Will enable conditionally
            "enable_attention_slicing": True,
            "enable_vae_slicing": True,
            "enable_vae_tiling": True,
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.float16,  # Use float16 for better memory efficiency
            "variant": "fp16"  # Use fp16 variant when available
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
            # Aggressive memory clearing before loading
            logger.info("🧹 Clearing GPU memory before model loading...")
            clear_gpu_memory(force=True)
            
            model_info = self.available_models[self.current_model]
            logger.info(f"📥 Loading {model_info['description']}...")
            
            # Check memory requirements
            memory_info = get_memory_info()
            logger.info(f"🔧 Memory status: {memory_info['gpu_free_gb']:.1f}GB free, {memory_info['gpu_allocated_gb']:.1f}GB allocated")
            
            # Check if we have enough memory
            if not check_memory_requirements(self.current_model):
                logger.warning(f"⚠️ Insufficient GPU memory for {self.current_model}")
                
                # Auto-fallback to smaller model
                if self.current_model == "flux_dev":
                    logger.info("🔄 Auto-switching to FLUX Schnell (smaller, faster)")
                    self.current_model = "flux_schnell"
                    model_info = self.available_models[self.current_model]
                elif self.current_model == "flux_schnell":
                    logger.info("🔄 Auto-switching to SDXL Base (much smaller)")
                    self.current_model = "sdxl_base"
                    model_info = self.available_models[self.current_model]
                else:
                    logger.error("❌ Not enough memory even for smallest model")
                    raise RuntimeError("Insufficient GPU memory. Try clearing memory or using a smaller model.")
            
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
                logger.info("🔧 Using maximum memory optimizations for FLUX...")
                
                # Load FLUX pipeline with maximum memory optimization
                load_kwargs = {
                    "torch_dtype": self.memory_optimizations["torch_dtype"],
                    "token": hf_token,
                    "use_safetensors": True,
                    "low_cpu_mem_usage": True,
                    "device_map": "auto",  # Let it handle device placement
                }
                
                # Try to use fp16 variant if available
                try:
                    load_kwargs["variant"] = "fp16"
                    self.pipeline = FluxPipeline.from_pretrained(model_info["id"], **load_kwargs)
                    logger.info("✅ Loaded FP16 variant")
                except:
                    # Fallback to regular loading
                    load_kwargs.pop("variant", None)
                    self.pipeline = FluxPipeline.from_pretrained(model_info["id"], **load_kwargs)
                    logger.info("✅ Loaded regular variant")
                
                # Don't manually move to device when using device_map="auto"
                logger.info(f"🔧 Model loaded with automatic device mapping")
                
                # Enable memory optimizations (but not CPU offload since we're using device_map)
                logger.info("🔧 Enabling memory optimizations...")
                
                # Don't use CPU offload with device_map="auto"
                logger.info("ℹ️ Using device_map instead of CPU offload")
                
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
                
                # Final memory check
                final_memory = get_memory_info()
                logger.info(f"🔧 Final memory: {final_memory['gpu_free_gb']:.1f}GB free")
                
            # Handle SDXL models
            elif model_info["type"] == "sdxl":
                load_kwargs = {
                    "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                    "token": hf_token,
                    "safety_checker": None,
                    "requires_safety_checker": False,
                    "use_safetensors": True,
                    "low_cpu_mem_usage": True,
                    "device_map": "auto"
                }
                
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(model_info["id"], **load_kwargs)
                
                # Don't manually move when using device_map
                logger.info(f"🔧 SDXL loaded with automatic device mapping")
                
                # Enable memory optimizations for SDXL
                if self.memory_optimizations["enable_attention_slicing"]:
                    self.pipeline.enable_attention_slicing()
                if self.memory_optimizations["enable_vae_slicing"]:
                    self.pipeline.enable_vae_slicing()
                if self.memory_optimizations["enable_vae_tiling"]:
                    self.pipeline.enable_vae_tiling()
            
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
            logger.info(f"✅ {model_info['description']} loaded successfully ({lora_status})")
            
            # Log final memory usage
            final_memory = get_memory_info()
            logger.info(f"🔧 Final memory status: {final_memory['gpu_free_gb']:.1f}GB free")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            error_memory = get_memory_info()
            logger.error(f"🔧 GPU Memory at error: {error_memory['gpu_allocated_gb']:.1f}GB allocated")
            self._loaded = False
            self.lora_loaded = False
            # Clear memory on error
            clear_gpu_memory(force=True)
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
            clear_gpu_memory(force=True)
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
            clear_gpu_memory(force=True)
            
            logger.info("🗑️ Model unloaded")