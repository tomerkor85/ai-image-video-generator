import os
import requests
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

async def download_models():
    """Download required LORA models"""
    logger.info("📥 Checking and downloading LORA models...")
    
    models = {
        "models/lora_t2v_A14B_separate_high.safetensors": {
            "url": "https://huggingface.co/tomerkor1985/test/resolve/main/lora_t2v_A14B_separate_high.safetensors",
            "description": "High noise LORA for WAN2.2"
        },
        "models/lora_t2v_A14B_separate_low.safetensors": {
            "url": "https://huggingface.co/tomerkor1985/test/resolve/main/lora_t2v_A14B_separate_low.safetensors", 
            "description": "Low noise LORA for WAN2.2"
        },
        "models/naya2.safetensors": {
            "url": "https://huggingface.co/tomerkor1985/test/resolve/main/naya2.safetensors",
            "description": "FLUX LORA for uncensored images"
        }
    }
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    for local_path, info in models.items():
        if os.path.exists(local_path):
            logger.info(f"✅ {info['description']}: {local_path}")
        else:
            logger.info(f"📥 Downloading {info['description']}...")
            try:
                response = requests.get(info['url'], stream=True)
                response.raise_for_status()
                
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"✅ Downloaded {info['description']}: {local_path}")
                
            except Exception as e:
                logger.error(f"❌ Failed to download {info['description']}: {e}")
                logger.error(f"   URL: {info['url']}")
    
    logger.info("✅ Model download check completed")