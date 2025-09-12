# 🔥 Uncensored AI Generator for RunPod

Professional AI image and video generation system designed for adult content creation with **ZERO censorship**.

## 🚀 Features

- **🎨 Multiple Model Support**: FLUX, SDXL, Realistic Vision, and CivitAI models
- **🎬 WAN2.2 Video Generation**: Smooth video generation with advanced models
- **🚫 ZERO Censorship**: Complete removal of ALL safety mechanisms
- **💻 Professional UI**: Beautiful interface with model selection
- **⚡ RunPod Optimized**: Designed specifically for RunPod GPU instances
- **🔧 Easy Setup**: Automatic model downloading and configuration

## 🎯 Choosing the Right Model for Adult Content

### **Recommended Models by Use Case:**

#### 🔥 **For Realistic Adult Content:**
1. **Realistic Vision V6** - Most popular for photorealistic adult content
2. **SDXL Base** - Very permissive, great quality
3. **CivitAI Custom Models** - Specialized adult models

#### ⚡ **For Speed:**
1. **FLUX Schnell** - Fastest generation, minimal censorship
2. **SDXL Turbo** - Quick results

#### 🎨 **For Artistic Style:**
1. **Custom CivitAI Models** - Anime, artistic styles
2. **Community Fine-tunes** - Specialized styles

### **How to Use CivitAI Models:**

1. **Browse CivitAI**: Go to [civitai.com](https://civitai.com)
2. **Filter by Adult Content**: Use NSFW filters
3. **Download Model**: Get the `.safetensors` file
4. **Place in Project**: Put in `models/civitai_model.safetensors`
5. **Select in UI**: Choose "Custom CivitAI model" in the interface

### **Top CivitAI Models for Adults (Examples):**
- **Realistic Vision V6.0** - Photorealistic people
- **ChilloutMix** - Asian-focused realistic model  
- **Deliberate** - Versatile for various styles
- **DreamShaper** - Fantasy and realistic mix
- **AbyssOrangeMix** - Anime style adult content

## 🛠️ RunPod Setup

### Quick Deploy
```bash
# Clone and run
git clone <your-repo>
cd ai-generator
python main.py
```

### Access Points
- **Web UI**: `http://your-runpod-ip:8000/ui`
- **API Docs**: `http://your-runpod-ip:8000/docs`
- **Model Selection**: Built into the UI

## 🎯 Usage

### Model Selection in UI
1. Open the web interface
2. In the Image tab, select your preferred model
3. Each model has different strengths for adult content
4. Experiment to find what works best for your needs

### API Usage
```python
# Generate with specific model
{
    "prompt": "your adult content prompt",
    "model": "realistic_vision",  # or flux_schnell, sdxl_base, civitai_custom
    "width": 1024,
    "height": 1024
}
```

## ⚠️ Adult Content Guidelines

### **This system is COMPLETELY UNCENSORED:**
- ✅ All safety checkers disabled
- ✅ No content filtering
- ✅ No prompt restrictions
- ✅ Full creative freedom

### **Recommended Settings for Adult Content:**
- **Steps**: 25-35 for quality
- **Guidance**: 7-12 for realism
- **Resolution**: 1024x1024 or higher
- **LORA Scale**: 0.8-1.2 for enhancement

## 🔧 Adding Your Own CivitAI Models

1. **Download from CivitAI**:
   - Visit model page
   - Download `.safetensors` file
   - Note the recommended settings

2. **Install in Project**:
   ```bash
   # Place your model file
   cp your-model.safetensors models/civitai_model.safetensors
   ```

3. **Use in Interface**:
   - Select "Custom CivitAI model" 
   - Use recommended prompts and settings
   - Experiment with different parameters

## 📊 System Requirements

- **GPU**: RTX 3090/4090 (12GB+ VRAM recommended)
- **RAM**: 32GB+ for multiple models
- **Storage**: 100GB+ for model collection
- **CUDA**: 11.8+

## 🔒 Privacy & Legal

- **Complete Privacy**: All generation happens on your RunPod
- **No External Calls**: Except for model downloads
- **Adult Content**: Use responsibly and legally
- **No Logging**: Your prompts and images are private

---

**⚠️ This software removes ALL content restrictions. Designed for professional adult content creators. Use responsibly and in accordance with local laws.**