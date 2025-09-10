# מדריך פריסה לשרתים חיצוניים

## שלב 1: העלאת הפרויקט ל-GitHub

### 1.1 יצירת Repository ב-GitHub
1. לך ל-[GitHub.com](https://github.com)
2. לחץ על "New repository"
3. שם: `ai-image-video-generator`
4. תיאור: `AI Image & Video Generation System with FLUX and WAN LORA`
5. בחר "Private" או "Public" לפי הצורך
6. **אל תסמן** "Add a README file" (כבר יש לנו)
7. לחץ "Create repository"

### 1.2 חיבור הפרויקט המקומי ל-GitHub
```bash
# הוסף את ה-remote repository
git remote add origin https://github.com/YOUR_USERNAME/ai-image-video-generator.git

# העלה את הקוד
git branch -M main
git push -u origin main
```

## שלב 2: בחירת שרת

### אפשרויות מומלצות:

#### 🚀 **RunPod** (מומלץ)
- **יתרונות**: זול, GPU חזק, קל להגדרה
- **מחיר**: $0.2-0.5/שעה
- **GPU**: RTX 4090, A100
- **זיכרון**: 24GB+ VRAM

#### ☁️ **AWS EC2**
- **יתרונות**: אמין, גמיש
- **מחיר**: $1-3/שעה
- **GPU**: V100, A100
- **זיכרון**: 16GB+ VRAM

#### 🔥 **Google Cloud**
- **יתרונות**: ביצועים טובים
- **מחיר**: $0.8-2/שעה
- **GPU**: V100, A100
- **זיכרון**: 16GB+ VRAM

## שלב 3: פריסה על RunPod

### 3.1 יצירת Instance
1. לך ל-[RunPod.io](https://runpod.io)
2. בחר "Deploy" → "RunPod Template"
3. בחר "PyTorch" template
4. בחר GPU: **RTX 4090** או **A100**
5. בחר "Start from this template"

### 3.2 הגדרת השרת
```bash
# התחבר לשרת דרך SSH
ssh root@YOUR_SERVER_IP

# Clone הפרויקט
git clone https://github.com/YOUR_USERNAME/ai-image-video-generator.git
cd ai-image-video-generator

# הרץ את סקריפט הפריסה
chmod +x runpod_deploy.sh
./runpod_deploy.sh
```

### 3.3 העלאת קבצי LORA
```bash
# העלה את קבצי ה-LORA שלך
scp -r "flux-lora/" root@YOUR_SERVER_IP:/root/ai-image-video-generator/
scp -r "naya_wan_lora/" root@YOUR_SERVER_IP:/root/ai-image-video-generator/
```

## שלב 4: פריסה על AWS EC2

### 4.1 יצירת Instance
1. לך ל-[AWS Console](https://console.aws.amazon.com)
2. בחר "EC2" → "Launch Instance"
3. בחר "Deep Learning AMI (Ubuntu 22.04)"
4. בחר Instance Type: **g4dn.xlarge** או **p3.2xlarge**
5. בחר "Launch"

### 4.2 הגדרת השרת
```bash
# התחבר לשרת
ssh -i your-key.pem ubuntu@YOUR_SERVER_IP

# התקן Docker
sudo apt update
sudo apt install docker.io docker-compose
sudo usermod -aG docker ubuntu
sudo systemctl start docker
sudo systemctl enable docker

# Clone הפרויקט
git clone https://github.com/YOUR_USERNAME/ai-image-video-generator.git
cd ai-image-video-generator

# הרץ עם Docker
docker-compose up -d
```

## שלב 5: בדיקת הפריסה

### 5.1 בדיקת סטטוס
```bash
# בדוק שהשרת רץ
curl http://YOUR_SERVER_IP:8000/health

# בדוק את ה-API docs
curl http://YOUR_SERVER_IP:8000/docs
```

### 5.2 בדיקת יצירת תמונה
```bash
curl -X POST "http://YOUR_SERVER_IP:8000/generate/image" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "beautiful landscape, detailed, high quality",
    "width": 1024,
    "height": 1024,
    "seed": 42
  }'
```

## שלב 6: אופטימיזציה

### 6.1 הגדרות זיכרון
```bash
# הגדר משתני סביבה
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
```

### 6.2 הגדרות Docker
```yaml
# docker-compose.yml
services:
  ai-generator:
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    deploy:
      resources:
        limits:
          memory: 32G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## שלב 7: ניטור ותחזוקה

### 7.1 ניטור ביצועים
```bash
# בדוק שימוש ב-GPU
nvidia-smi

# בדוק זיכרון
free -h

# בדוק לוגים
docker logs ai-generator
```

### 7.2 גיבוי
```bash
# גבה את המודלים
tar -czf models_backup.tar.gz flux-lora/ "naya_wan_lora/"

# גבה את הפלטים
tar -czf outputs_backup.tar.gz outputs/
```

## שלב 8: אבטחה

### 8.1 Firewall
```bash
# פתח רק את הפורט הנדרש
sudo ufw allow 8000
sudo ufw enable
```

### 8.2 HTTPS (אופציונלי)
```bash
# התקן nginx ו-certbot
sudo apt install nginx certbot python3-certbot-nginx

# הגדר SSL
sudo certbot --nginx -d your-domain.com
```

## פתרון בעיות נפוצות

### שגיאת זיכרון
```bash
# הפעל swap
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### שגיאת CUDA
```bash
# בדוק התקנת CUDA
nvidia-smi
nvcc --version

# התקן מחדש אם נדרש
sudo apt install nvidia-cuda-toolkit
```

### שגיאת Docker
```bash
# הפעל מחדש Docker
sudo systemctl restart docker

# נקה cache
docker system prune -a
```

## עלויות משוערות

| שרת | GPU | זיכרון | מחיר/שעה | מחיר/יום |
|------|-----|--------|-----------|----------|
| RunPod | RTX 4090 | 24GB | $0.3 | $7.2 |
| AWS | V100 | 16GB | $1.5 | $36 |
| GCP | A100 | 40GB | $2.0 | $48 |

## תמיכה

לשאלות ותמיכה:
- צור Issue ב-GitHub
- בדוק את הלוגים: `docker logs ai-generator`
- בדוק את הבריאות: `curl http://YOUR_SERVER_IP:8000/health`
