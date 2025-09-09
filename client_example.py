import requests
import base64
import json
from PIL import Image
import io

class AIGeneratorClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def generate_image(self, prompt, **kwargs):
        """Generate image using FLUX LORA"""
        url = f"{self.base_url}/generate/image"
        
        payload = {
            "prompt": prompt,
            "negative_prompt": kwargs.get("negative_prompt", ""),
            "width": kwargs.get("width", 1024),
            "height": kwargs.get("height", 1024),
            "num_inference_steps": kwargs.get("num_inference_steps", 20),
            "guidance_scale": kwargs.get("guidance_scale", 7.5),
            "seed": kwargs.get("seed", None),
            "lora_scale": kwargs.get("lora_scale", 1.0)
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        if result["success"]:
            # Decode base64 image
            image_data = base64.b64decode(result["data"]["image"])
            image = Image.open(io.BytesIO(image_data))
            return image, result["data"]
        else:
            raise Exception(f"Generation failed: {result['message']}")
    
    def generate_video(self, prompt, **kwargs):
        """Generate video using WAN LORA"""
        url = f"{self.base_url}/generate/video"
        
        payload = {
            "prompt": prompt,
            "negative_prompt": kwargs.get("negative_prompt", ""),
            "width": kwargs.get("width", 512),
            "height": kwargs.get("height", 512),
            "num_frames": kwargs.get("num_frames", 16),
            "num_inference_steps": kwargs.get("num_inference_steps", 25),
            "guidance_scale": kwargs.get("guidance_scale", 7.5),
            "seed": kwargs.get("seed", None),
            "lora_scale": kwargs.get("lora_scale", 1.0)
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        if result["success"]:
            # Decode base64 frames
            frames = []
            for frame_data in result["data"]["frames"]:
                frame_bytes = base64.b64decode(frame_data)
                frame = Image.open(io.BytesIO(frame_bytes))
                frames.append(frame)
            return frames, result["data"]
        else:
            raise Exception(f"Generation failed: {result['message']}")
    
    def health_check(self):
        """Check service health"""
        url = f"{self.base_url}/health"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_models_info(self):
        """Get information about loaded models"""
        url = f"{self.base_url}/models/info"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

# Example usage
if __name__ == "__main__":
    # Initialize client
    client = AIGeneratorClient("http://your-server:8000")
    
    try:
        # Check health
        health = client.health_check()
        print(f"Service health: {health}")
        
        # Generate image
        print("Generating image...")
        image, data = client.generate_image(
            prompt="beautiful landscape, detailed, high quality",
            negative_prompt="blurry, low quality",
            width=1024,
            height=1024,
            seed=42
        )
        
        # Save image
        image.save("generated_image.png")
        print(f"Image saved as generated_image.png")
        
        # Generate video
        print("Generating video...")
        frames, data = client.generate_video(
            prompt="dancing person, smooth motion",
            negative_prompt="jumpy, unstable",
            width=512,
            height=512,
            num_frames=16,
            seed=42
        )
        
        # Save first frame as example
        frames[0].save("generated_video_frame.png")
        print(f"Video frame saved as generated_video_frame.png")
        
    except Exception as e:
        print(f"Error: {e}")
