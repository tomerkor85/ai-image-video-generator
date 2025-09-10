#!/bin/bash

# RunPod Deployment Script
# This script sets up the AI generation system on RunPod

echo "🚀 Starting RunPod deployment..."

# Update system
echo "📦 Updating system packages..."
apt update && apt upgrade -y

# Install required packages
echo "🔧 Installing required packages..."
apt install -y python3 python3-pip git wget curl

# Install Docker
echo "🐳 Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
rm get-docker.sh

# Install NVIDIA Container Toolkit
echo "🎮 Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update && apt-get install -y nvidia-docker2
systemctl restart docker

# Clone or copy project files
echo "📁 Setting up project files..."
# Note: You need to upload your project files to RunPod first
# This assumes the files are already in the current directory

# Create necessary directories
mkdir -p outputs temp

# Set permissions
chmod +x main.py

# Build Docker image
echo "🏗️ Building Docker image..."
docker build -t ai-generator .

# Run the container
echo "🚀 Starting AI generation service..."
docker run --gpus all -d \
  --name ai-generator \
  -p 8000:8000 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/temp:/app/temp \
  -v $(pwd)/flux-lora:/app/flux-lora \
  -v "$(pwd)/naya_wan_lora":/app/naya\ wan\ lora \
  ai-generator

# Wait for service to start
echo "⏳ Waiting for service to start..."
sleep 30

# Check if service is running
echo "🔍 Checking service status..."
if curl -f http://localhost:8000/health; then
    echo "✅ Service is running successfully!"
    echo "🌐 API available at: http://$(curl -s ifconfig.me):8000"
    echo "📊 Health check: http://$(curl -s ifconfig.me):8000/health"
    echo "📖 API docs: http://$(curl -s ifconfig.me):8000/docs"
else
    echo "❌ Service failed to start. Checking logs..."
    docker logs ai-generator
fi

echo "🎉 Deployment complete!"
echo ""
echo "📋 Useful commands:"
echo "  View logs: docker logs ai-generator"
echo "  Restart: docker restart ai-generator"
echo "  Stop: docker stop ai-generator"
echo "  Remove: docker rm ai-generator"
