# 🚀 RunPod Setup Guide

## Template Selection

### **Recommended Template:**
```
runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
```

### **GPU Recommendations:**
1. **RTX 4090** (24GB) - Best performance/price
2. **RTX 3090** (24GB) - Great alternative  
3. **A6000** (48GB) - For heavy workloads

## Quick Setup Steps

### 1. Create RunPod Instance
- Choose PyTorch template above
- Select GPU (RTX 4090 recommended)
- Set container disk to 50GB+
- Set volume to 100GB+ (for models)

### 2. Upload Project
```bash
# In RunPod terminal
git clone <your-repo-url>
cd ai-generator
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Application
```bash
python main.py
```

### 5. Setup Hugging Face Token (Important!)
```bash
# Method 1: Environment variable (recommended)
export HUGGINGFACE_TOKEN="hf_your_token_here"

# Method 2: Using huggingface-cli
pip install huggingface_hub
huggingface-cli login

# Method 3: Via API (in the web interface)
# Go to /setup/hf-token endpoint
```

### 6. Access Interface
- **Web UI**: `https://<pod-id>-8000.proxy.runpod.net/ui`
- **API**: `https://<pod-id>-8000.proxy.runpod.net/docs`

## Environment Variables

```bash
export HUGGINGFACE_TOKEN="hf_your_token_here"  # Required for most models
export CUDA_VISIBLE_DEVICES=0
```

## Performance Tips

1. **Use Volume Storage** for models (persistent)
2. **Enable TCP Port 8000** in RunPod settings
3. **Use Spot Instances** for cost savings
4. **Monitor GPU Memory** in the UI

## Troubleshooting

### Common Issues:
- **Out of Memory**: Reduce batch size or image resolution
- **Model Download Fails**: Check internet connection
- **Port Not Accessible**: Enable TCP port 8000 in RunPod

### Memory Optimization:
```python
# These are already included in the code
torch.cuda.empty_cache()
pipeline.enable_attention_slicing()
pipeline.enable_model_cpu_offload()
```

## Cost Optimization

1. **Use Spot Instances** (50-80% cheaper)
2. **Stop Pod** when not in use
3. **Use Smaller Models** for testing
4. **Batch Multiple Generations** together

---

**Ready to generate uncensored AI content! 🔥**