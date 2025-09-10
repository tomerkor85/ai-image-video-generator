# AI Image & Video Generator

🚀 AI-powered image and video generation using FLUX.1-dev and Stable Video Diffusion with custom LORA models.

## Features

- 🎨 **Image Generation**: High-quality images using FLUX.1-dev with custom LORA
- 🎬 **Video Generation**: Text-to-video using Stable Video Diffusion with custom LORA
- 🌐 **Web Interface**: Beautiful web UI for easy generation
- 🔧 **API**: RESTful API for programmatic access
- 📱 **Responsive**: Works on desktop and mobile

## Installation

### Automatic Model Download
The LORA models will be downloaded automatically on first run from:
- [High LORA](https://huggingface.co/tomerkor1985/test/resolve/main/lora_t2v_A14B_separate_high.safetensors)
- [Low LORA](https://huggingface.co/tomerkor1985/test/resolve/main/lora_t2v_A14B_separate_low.safetensors)
- [Flux LORA](https://huggingface.co/tomerkor1985/test/resolve/main/naya2.safetensors)

### Manual Download
```bash
# Download models manually
bash download_models.sh
```

### Python Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

## Usage

### Web Interface
1. Start the server: `python main.py`
2. Open your browser: `http://localhost:8000/ui`
3. Enter your prompt and generate!

### API Usage

#### Generate Image
```bash
curl -X POST "http://localhost:8000/generate/image" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful sunset over mountains, highly detailed digital art",
    "negative_prompt": "blurry, low quality",
    "width": 1024,
    "height": 1024,
    "steps": 25,
    "guidance": 7.5,
    "model_type": "flux",
    "lora_scale": 1.0
  }'
```

#### Generate Video
```bash
curl -X POST "http://localhost:8000/generate/video" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cat playing with a ball",
    "negative_prompt": "blurry, low quality",
    "width": 512,
    "height": 512,
    "num_frames": 16,
    "steps": 25,
    "guidance": 7.5,
    "lora_scale": 1.0,
    "lora_type": "high"
  }'
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /generate/image` - Generate image
- `POST /generate/video` - Generate video
- `GET /ui` - Web interface
- `GET /outputs` - List generated files
- `GET /docs` - API documentation

## Models

### FLUX.1-dev
- **Model**: `black-forest-labs/FLUX.1-dev`
- **LORA**: Custom Naya LORA for enhanced generation
- **Use Case**: High-quality image generation

### Stable Video Diffusion
- **Model**: `stabilityai/stable-video-diffusion-img2vid-xt`
- **LORA**: Custom Naya WAN LORA (High/Low noise variants)
- **Use Case**: Text-to-video generation

## Configuration

### Environment Variables
```bash
# Optional: Hugging Face token for private models
export HUGGINGFACE_TOKEN="your_token_here"

# Cache directory
export HF_HOME="/workspace/cache"
export TRANSFORMERS_CACHE="/workspace/cache"
export DIFFUSERS_CACHE="/workspace/cache"
```

### GPU Requirements
- **Minimum**: 8GB VRAM
- **Recommended**: 16GB+ VRAM
- **CUDA**: Required for GPU acceleration

## Docker Support

```bash
# Build and run with Docker
docker-compose up --build
```

## Troubleshooting

### Model Download Issues
If automatic download fails, try manual download:
```bash
bash download_models.sh
```

### Memory Issues
- Reduce image/video dimensions
- Lower number of inference steps
- Use CPU offloading (enabled by default)

### Performance Tips
- Use GPU for faster generation
- Enable xformers for memory efficiency
- Adjust batch sizes based on available memory

## License

This project uses various open-source models and libraries. Please check individual licenses for commercial use.

## Support

For issues and questions, please check the API documentation at `/docs` when running the server.
