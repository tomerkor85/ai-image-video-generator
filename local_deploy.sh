#!/bin/bash

# Local Deployment Script
echo "🚀 Starting local deployment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Stop and remove existing container if it exists
echo "🧹 Cleaning up existing container..."
docker stop ai-generator 2>/dev/null || true
docker rm ai-generator 2>/dev/null || true

# Build Docker image
echo "🏗️ Building Docker image..."
docker build -t ai-generator .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

# Create necessary directories
mkdir -p outputs temp

# Run the container
echo "🚀 Starting AI generation service..."
docker run --gpus all -d \
  --name ai-generator \
  -p 8000:8000 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/temp:/app/temp \
  -v $(pwd)/models:/app/models \
  -v "$(pwd)/naya_wan_lora":/app/naya_wan_lora \
  ai-generator

# Wait for service to start
echo "⏳ Waiting for service to start..."
sleep 30

# Check if service is running
echo "🔍 Checking service status..."
if curl -f http://localhost:8000/health; then
    echo "✅ Service is running successfully!"
    echo "🌐 Web UI: http://localhost:8000/ui"
    echo "📊 Health check: http://localhost:8000/health"
    echo "📖 API docs: http://localhost:8000/docs"
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
