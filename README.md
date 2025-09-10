# AI Image & Video Generation System

מערכת ליצירת תמונות ווידאו ב-AI עם FLUX 1 DEV LORA ו-WAN2.2 LORA ללא צנזורה.

## תכונות

- **יצירת תמונות**: FLUX 1 DEV עם LORA מותאם אישית
- **יצירת וידאו**: WAN 2.2 עם LORA מותאם אישית  
- **ללא צנזורה**: 100% חופש אומנותי
- **API RESTful**: ממשק API מלא עם FastAPI
- **תמיכה ב-GPU**: אופטימיזציה מלאה ל-CUDA
- **Docker**: פריסה קלה על שרתים חיצוניים

## דרישות מערכת

- **GPU**: NVIDIA GPU עם CUDA 11.8+
- **זיכרון**: 16GB+ RAM, 8GB+ VRAM
- **מקום אחסון**: 50GB+ מקום פנוי
- **מערכת הפעלה**: Ubuntu 22.04+ או Windows עם WSL2

## התקנה מקומית

### 1. הכנת הסביבה

```bash
# Clone the repository
git clone <your-repo-url>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# או
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. הכנת קבצי LORA

ודא שקבצי ה-LORA נמצאים במיקומים הנכונים:
- `flux-lora/naya2.safetensors`
- `naya_wan_lora/high_lora.safetensors`

### 3. הרצת השרת

```bash
python main.py
```

השרת יהיה זמין ב: `http://localhost:8000`

## פריסה על שרת חיצוני (RUNPOD/AWS/GCP)

### 1. הכנת השרת

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 2. העלאת הפרויקט

```bash
# Upload your project files to the server
scp -r . user@server-ip:/path/to/project
```

### 3. בניית והרצת Docker

```bash
cd /path/to/project

# Build the image
docker build -t ai-generator .

# Run with GPU support
docker run --gpus all -p 8000:8000 -v $(pwd)/outputs:/app/outputs ai-generator
```

### 4. עם Docker Compose

```bash
docker-compose up -d
```

## שימוש ב-API

### יצירת תמונה

```bash
curl -X POST "http://your-server:8000/generate/image" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "beautiful landscape, detailed, high quality",
    "negative_prompt": "blurry, low quality",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 20,
    "guidance_scale": 7.5,
    "seed": 42
  }'
```

### יצירת וידאו

```bash
curl -X POST "http://your-server:8000/generate/video" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "dancing person, smooth motion",
    "negative_prompt": "jumpy, unstable",
    "width": 512,
    "height": 512,
    "num_frames": 16,
    "num_inference_steps": 25,
    "guidance_scale": 7.5,
    "seed": 42
  }'
```

### בדיקת סטטוס

```bash
curl http://your-server:8000/health
```

## הגדרות ללא צנזורה

המערכת מוגדרת מראש ללא צנזורה:

- `safety_checker=None`
- `requires_safety_checker=False`
- אין סינון תוכן
- חופש אומנותי מלא

## אופטימיזציה

### זיכרון GPU

```python
# Enable memory optimizations
pipeline.enable_attention_slicing()
pipeline.enable_model_cpu_offload()
pipeline.enable_memory_efficient_attention()
```

### Batch Processing

```python
# Process multiple requests
async def batch_generate(requests):
    tasks = [generate_image(req) for req in requests]
    return await asyncio.gather(*tasks)
```

## פתרון בעיות

### שגיאות זיכרון

```bash
# Reduce batch size
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### שגיאות LORA

```bash
# Check LORA file paths
ls -la flux-lora/
ls -la "naya_wan_lora/"
```

### שגיאות CUDA

```bash
# Check GPU availability
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## רישיון

פרויקט זה מיועד למטרות חינוכיות ומחקריות בלבד.

## תמיכה

לשאלות ותמיכה, צור issue ב-GitHub או פנה ישירות.
