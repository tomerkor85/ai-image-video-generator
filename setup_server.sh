#!/bin/bash

# Server Setup Script for AI Generation System
# This script prepares a fresh server for deployment

echo "🚀 Starting server setup for AI Generation System..."

# Update system
echo "📦 Updating system packages..."
apt update && apt upgrade -y

# Install essential packages
echo "🔧 Installing essential packages..."
apt install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    unzip \
    htop \
    nano \
    vim \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1

# Install Docker
echo "🐳 Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
rm get-docker.sh

# Install Docker Compose
echo "🔗 Installing Docker Compose..."
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install NVIDIA Container Toolkit (if NVIDIA GPU is present)
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
    
    apt-get update && apt-get install -y nvidia-docker2
    systemctl restart docker
    
    echo "✅ NVIDIA Container Toolkit installed"
else
    echo "⚠️  No NVIDIA GPU detected, skipping NVIDIA Container Toolkit"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p /opt/ai-generator
mkdir -p /opt/ai-generator/outputs
mkdir -p /opt/ai-generator/temp
mkdir -p /opt/ai-generator/logs

# Set up swap file for memory optimization
echo "💾 Setting up swap file..."
if [ ! -f /swapfile ]; then
    fallocate -l 16G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo "✅ Swap file created (16GB)"
else
    echo "✅ Swap file already exists"
fi

# Configure firewall
echo "🔥 Configuring firewall..."
ufw --force enable
ufw allow ssh
ufw allow 8000
ufw allow 80
ufw allow 443
echo "✅ Firewall configured"

# Set up environment variables
echo "🌍 Setting up environment variables..."
cat >> /etc/environment << EOF
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
TOKENIZERS_PARALLELISM=false
PYTHONUNBUFFERED=1
EOF

# Create systemd service for auto-start
echo "⚙️ Creating systemd service..."
cat > /etc/systemd/system/ai-generator.service << EOF
[Unit]
Description=AI Image & Video Generation Service
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/ai-generator
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ai-generator.service

# Create monitoring script
echo "📊 Creating monitoring script..."
cat > /opt/ai-generator/monitor.sh << 'EOF'
#!/bin/bash

echo "=== AI Generator System Status ==="
echo "Date: $(date)"
echo ""

echo "=== Docker Status ==="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

echo "=== GPU Status ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
else
    echo "No NVIDIA GPU detected"
fi
echo ""

echo "=== Memory Status ==="
free -h
echo ""

echo "=== Disk Usage ==="
df -h /opt/ai-generator
echo ""

echo "=== Service Health ==="
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Service is healthy"
else
    echo "❌ Service is not responding"
fi
echo ""

echo "=== Recent Logs ==="
docker logs ai-generator --tail 10
EOF

chmod +x /opt/ai-generator/monitor.sh

# Create backup script
echo "💾 Creating backup script..."
cat > /opt/ai-generator/backup.sh << 'EOF'
#!/bin/bash

BACKUP_DIR="/opt/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

echo "Creating backup: $DATE"

# Backup outputs
tar -czf $BACKUP_DIR/outputs_$DATE.tar.gz -C /opt/ai-generator outputs/

# Backup models (if they exist)
if [ -d "/opt/ai-generator/flux-lora" ]; then
    tar -czf $BACKUP_DIR/flux_lora_$DATE.tar.gz -C /opt/ai-generator flux-lora/
fi

if [ -d "/opt/ai-generator/naya wan lora" ]; then
    tar -czf $BACKUP_DIR/wan_lora_$DATE.tar.gz -C /opt/ai-generator "naya wan lora/"
fi

# Clean old backups (keep last 7 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_DIR"
EOF

chmod +x /opt/ai-generator/backup.sh

# Set up cron jobs
echo "⏰ Setting up cron jobs..."
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/ai-generator/backup.sh") | crontab -
(crontab -l 2>/dev/null; echo "*/5 * * * * /opt/ai-generator/monitor.sh >> /opt/ai-generator/logs/monitor.log") | crontab -

# Create log rotation
echo "📝 Setting up log rotation..."
cat > /etc/logrotate.d/ai-generator << EOF
/opt/ai-generator/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 root root
}
EOF

echo ""
echo "🎉 Server setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Clone your project: git clone https://github.com/YOUR_USERNAME/ai-image-video-generator.git /opt/ai-generator"
echo "2. Upload your LORA files to the appropriate directories"
echo "3. Run: cd /opt/ai-generator && docker-compose up -d"
echo "4. Check status: /opt/ai-generator/monitor.sh"
echo ""
echo "🔧 Useful commands:"
echo "  Monitor: /opt/ai-generator/monitor.sh"
echo "  Backup: /opt/ai-generator/backup.sh"
echo "  Logs: docker logs ai-generator"
echo "  Restart: systemctl restart ai-generator"
echo ""
echo "🌐 Service will be available at: http://$(curl -s ifconfig.me):8000"
