import os
from typing import Dict, Any

class Config:
    """Configuration class for uncensored AI generation"""
    
    # Model paths
    FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
    WAN_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"
    
    # LORA paths
    FLUX_LORA_PATH = "models/flux_naya.safetensors"
    WAN_LORA_HIGH_PATH = "naya_wan_lora/lora_t2v_A14B_separate_high.safetensors"
    WAN_LORA_LOW_PATH = "naya_wan_lora/lora_t2v_A14B_separate_low.safetensors"
    
    # Device configuration
    DEVICE = "cuda" if os.environ.get("CUDA_AVAILABLE", "true").lower() == "true" else "cpu"
    TORCH_DTYPE = "float16" if DEVICE == "cuda" else "float32"
    
    # Uncensored settings
    SAFETY_CHECKER_ENABLED = False
    REQUIRES_SAFETY_CHECKER = False
    
    # Generation defaults
    DEFAULT_IMAGE_SIZE = (1024, 1024)
    DEFAULT_VIDEO_SIZE = (512, 512)
    DEFAULT_INFERENCE_STEPS = 20
    DEFAULT_GUIDANCE_SCALE = 7.5
    DEFAULT_NUM_FRAMES = 16
    
    # Memory optimization
    ENABLE_ATTENTION_SLICING = True
    ENABLE_MODEL_CPU_OFFLOAD = True
    ENABLE_MEMORY_EFFICIENT_ATTENTION = True
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    MAX_CONCURRENT_REQUESTS = 4
    
    # File paths
    OUTPUT_DIR = "outputs"
    TEMP_DIR = "temp"
    
    # Logging
    LOG_LEVEL = "INFO"
    
    @classmethod
    def get_model_config(cls, model_type: str) -> Dict[str, Any]:
        """Get model-specific configuration"""
        base_config = {
            "torch_dtype": getattr(torch, cls.TORCH_DTYPE) if hasattr(torch, cls.TORCH_DTYPE) else torch.float16,
            "device_map": "balanced" if cls.DEVICE == "cuda" else None,
            "safety_checker": None,
            "requires_safety_checker": cls.REQUIRES_SAFETY_CHECKER,
        }
        
        if model_type == "flux":
            base_config.update({
                "model_id": cls.FLUX_MODEL_ID,
                "lora_path": cls.FLUX_LORA_PATH,
            })
        elif model_type == "wan":
            base_config.update({
                "model_id": cls.WAN_MODEL_ID,
                "lora_high_path": cls.WAN_LORA_HIGH_PATH,
                "lora_low_path": cls.WAN_LORA_LOW_PATH,
            })
        
        return base_config
    
    @classmethod
    def ensure_directories(cls):
        """Ensure required directories exist"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.TEMP_DIR, exist_ok=True)

# Import torch here to avoid circular imports
try:
    import torch
except ImportError:
    torch = None
