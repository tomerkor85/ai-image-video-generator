# 🔥 Uncensored AI Generator for RunPod

Professional AI image and video generation system designed for adult content creation with no censorship restrictions.

## 🚀 Features

- **🎨 FLUX Image Generation**: High-quality uncensored images with custom LORA
- **🎬 WAN2.2 Video Generation**: Smooth video generation with advanced models
- **🚫 No Censorship**: Complete removal of safety checkers and content filters
- **💻 Professional UI**: Beautiful tabbed interface with advanced controls
- **⚡ RunPod Optimized**: Designed specifically for RunPod GPU instances
- **🔧 Easy Setup**: Automatic model downloading and configuration

## 🛠️ RunPod Setup

### Quick Deploy
1. Create new RunPod instance with GPU (RTX 3090/4090 recommended)
2. Use this Docker image or clone repository
3. Run: `python main.py`
4. Access UI at: `http://your-runpod-ip:8000/ui`

### Manual Setup
```bash
# Clone repository
git clone <your-repo-url>
cd ai-generator

# Install dependencies
pip install -r requirements.txt

# Run server
python main.py
```

## 🎯 Usage

### Web Interface
- Navigate to `/ui` for the complete interface
- **Image Tab**: Generate images using FLUX + LORA
- **Video Tab**: Generate videos using WAN2.2
- **Gallery**: View and download all creations

### API Endpoints
- `POST /generate/image` - Generate uncensored images
- `POST /generate/video` - Generate uncensored videos
- `GET /health` - System health check
- `GET /outputs/list` - List generated files

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Hugging Face token for FLUX model access
export HUGGINGFACE_TOKEN="your_token_here"

# GPU memory optimization
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
```

### Model Requirements
- **FLUX LORA**: `naya2.safetensors` (auto-downloaded)
- **WAN2.2 LORA**: High/Low noise variants (auto-downloaded)
- **GPU Memory**: 8GB+ VRAM recommended

## ⚠️ Important Notes

### Adult Content Warning
This system is designed for professional adult content creation and has all safety mechanisms disabled. Use responsibly and in accordance with local laws.

### Performance Tips
- Use RTX 3090/4090 for best performance
- Enable xformers for memory efficiency
- Adjust batch sizes based on available VRAM
- Use lower resolutions for faster generation

## 🐳 Docker Support

```bash
# Build and run
docker build -t uncensored-ai .
docker run --gpus all -p 8000:8000 uncensored-ai
```

## 📊 System Requirements

- **GPU**: NVIDIA RTX 3090/4090 (8GB+ VRAM)
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ free space
- **CUDA**: 11.8+

## 🔒 Security & Privacy

- All generation happens locally on your RunPod instance
- No data is sent to external services (except model downloads)
- Complete privacy for your content creation

## 📞 Support

For issues specific to this uncensored version:
1. Check GPU memory usage
2. Verify model downloads completed
3. Check RunPod instance specifications
4. Review logs for detailed error information

---

**⚠️ This software is intended for adult content creators and removes all content safety mechanisms. Use responsibly.**