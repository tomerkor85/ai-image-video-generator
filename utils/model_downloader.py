import os
import requests
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

async def download_models():
    """Download required LORA models from Hugging Face"""
    logger.info("📥 Checking and downloading LORA models...")
    
    # Get HF token
    hf_token = (
        os.environ.get("HUGGINGFACE_TOKEN") or 
        os.environ.get("HF_TOKEN") or
        None
    )
    
    models = {
        # Image LORAs
        "naya2.safetensors": {
            "repo_id": "tomerkor1985/test",
            "filename": "naya2.safetensors",
            "description": "NAYA2 LORA for uncensored images (SDXL)",
            "type": "image"
        },
        
        # Video LORAs
        "naya_wan_lora.safetensors": {
            "repo_id": "ByteDance/AnimateDiff-Lightning",
            "filename": "animatediff_lightning_4step_lora.safetensors",
            "description": "WAN LORA for video generation",
            "type": "video",
            "optional": True
        },
        "lora_t2v_A14B_separate_high.safetensors": {
            "repo_id": "ali-vilab/i2vgen-xl",
            "filename": "lora_t2v_A14B_separate_high.safetensors", 
            "description": "High noise LORA for video",
            "type": "video",
            "optional": True
        },
        "lora_t2v_A14B_separate_low.safetensors": {
            "repo_id": "ali-vilab/i2vgen-xl",
            "filename": "lora_t2v_A14B_separate_low.safetensors",
            "description": "Low noise LORA for video", 
            "type": "video",
            "optional": True
        }
    }
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    for filename, info in models.items():
        local_path = f"models/{filename}"
        
        if os.path.exists(local_path):
            logger.info(f"✅ {info['description']}: {local_path}")
        else:
            if info.get('optional'):
                logger.info(f"⚠️ Optional LORA not found: {info['description']}")
                logger.info(f"   Will try to download from: {info['repo_id']}")
                try:
                    if hf_token:
                        downloaded_path = hf_hub_download(
                            repo_id=info['repo_id'],
                            filename=info['filename'],
                            token=hf_token,
                            cache_dir="models",
                            local_dir="models",
                            local_dir_use_symlinks=False
                        )
                        logger.info(f"✅ Downloaded optional LORA: {info['description']}")
                    else:
                        logger.warning(f"⚠️ No HF token - skipping optional LORA: {info['description']}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not download optional LORA {info['description']}: {e}")
            else:
                logger.info(f"📥 Downloading {info['description']}...")
                try:
                    downloaded_path = hf_hub_download(
                        repo_id=info['repo_id'],
                        filename=info['filename'],
                        token=hf_token,
                        cache_dir="models",
                        local_dir="models", 
                        local_dir_use_symlinks=False
                    )
                    logger.info(f"✅ Downloaded {info['description']}: {local_path}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to download {info['description']}: {e}")
                    logger.error(f"   Repo: {info['repo_id']}")
    
    logger.info("✅ Model download check completed")