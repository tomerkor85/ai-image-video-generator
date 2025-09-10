#!/bin/bash

echo "📥 Downloading LORA models..."

# Create directories
mkdir -p naya_wan_lora
mkdir -p flux-lora

# Download files
echo "Downloading High LORA..."
wget -c -O naya_wan_lora/lora_t2v_A14B_separate_high.safetensors \
  https://huggingface.co/tomerkor1985/test/resolve/main/lora_t2v_A14B_separate_high.safetensors

echo "Downloading Low LORA..."
wget -c -O naya_wan_lora/lora_t2v_A14B_separate_low.safetensors \
  https://huggingface.co/tomerkor1985/test/resolve/main/lora_t2v_A14B_separate_low.safetensors

echo "Downloading Flux LORA..."
wget -c -O flux-lora/naya2.safetensors \
  https://huggingface.co/tomerkor1985/test/resolve/main/naya2.safetensors

echo "✅ All models downloaded successfully!"

# Check file sizes
echo "📊 File sizes:"
ls -lh naya_wan_lora/
ls -lh flux-lora/
