#!/bin/bash

echo "🚀 RunPod Fixed Installation Script"
echo "===================================="

# Navigate to workspace
cd /workspace

# Clone project if not exists
if [ ! -d "ai-image-video-generator" ]; then
    echo "📥 Cloning project..."
    git clone https://github.com/tomerkor85/ai-image-video-generator
fi
cd ai-image-video-generator

# Create and activate virtual environment
echo "🔧 Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install NumPy first (correct version)
echo "📦 Installing NumPy..."
pip install numpy==1.26.4

# Install PyTorch with correct versions
echo "📦 Installing PyTorch suite..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install AI packages
echo "📦 Installing AI packages..."
pip install diffusers==0.27.2
pip install transformers==4.40.0
pip install accelerate==0.29.3
pip install safetensors==0.4.3

# Install xformers (for memory optimization)
echo "📦 Installing xformers..."
pip install xformers==0.0.22.post7 --no-deps

# Install FastAPI and web server
echo "📦 Installing FastAPI..."
pip install fastapi==0.110.0
pip install uvicorn[standard]==0.29.0
pip install python-multipart==0.0.9

# Install additional utilities
echo "📦 Installing utilities..."
pip install pillow
pip install opencv-python-headless  # Use headless version for server
pip install imageio
pip install imageio-ffmpeg
pip install scipy
pip install omegaconf

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p "flux-lora"
mkdir -p "naya wan lora"
mkdir -p outputs
mkdir -p cache
mkdir -p models

# Set environment variables
echo "🔧 Setting environment variables..."
cat >> ~/.bashrc << 'EOL'
export HF_HOME=/workspace/cache
export TRANSFORMERS_CACHE=/workspace/cache
export DIFFUSERS_CACHE=/workspace/cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0
EOL

source ~/.bashrc

echo "✅ Installation complete!"
echo "===================================="
python3 -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
"