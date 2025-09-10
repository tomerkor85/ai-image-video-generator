#!/usr/bin/env python3
"""בדיקת התקנה מלאה של הפרויקט"""

import sys
import os
from pathlib import Path

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_package(name, import_name=None):
    if import_name is None:
        import_name = name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'N/A')
        print(f"✅ {name:20} {version}")
        return True
    except ImportError as e:
        print(f"❌ {name:20} Error: {e}")
        return False

def main():
    print_section("🔍 System Check for AI Image/Video Generator")
    
    # Python version
    print(f"\n📌 Python Version: {sys.version}")
    
    # Check CUDA and PyTorch
    print_section("🎮 GPU & CUDA Check")
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"  Compute Capability: {props.major}.{props.minor}")
        
        # Test CUDA operation
        if torch.cuda.is_available():
            test_tensor = torch.randn(100, 100).cuda()
            result = torch.matmul(test_tensor, test_tensor)
            print("✅ CUDA Operations: Working")
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ PyTorch/CUDA Error: {e}")
    
    # Check required packages
    print_section("📦 Required Packages")
    packages = [
        ("torch", None),
        ("torchvision", None),
        ("diffusers", None),
        ("transformers", None),
        ("accelerate", None),
        ("safetensors", None),
        ("xformers", None),
        ("fastapi", None),
        ("uvicorn", None),
        ("PIL", "PIL"),
        ("cv2", "cv2"),
        ("imageio", None),
        ("numpy", None),
    ]
    
    all_good = True
    for package in packages:
        if not check_package(*package):
            all_good = False
    
    # Check directories
    print_section("📁 Directory Structure")
    dirs = {
        "flux-lora": "FLUX LORA models",
        "naya_wan_lora": "WAN LORA models",
        "outputs": "Output files",
        "cache": "Model cache",
        "models": "Base models"
    }
    
    for dir_name, description in dirs.items():
        path = Path(dir_name)
        if path.exists():
            print(f"✅ {dir_name:20} - {description}")
        else:
            print(f"⚠️  {dir_name:20} - {description} (Creating...)")
            path.mkdir(parents=True, exist_ok=True)
    
    # Check LORA files
    print_section("📄 Model Files")
    model_files = [
        ("flux-lora/naya2.safetensors", "FLUX LORA"),
        ("naya_wan_lora/high_lora.safetensors", "WAN LORA")
    ]
    
    for file_path, description in model_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size / (1024**2)
            print(f"✅ {description:15} - {file_path} ({size:.2f} MB)")
        else:
            print(f"⚠️  {description:15} - {file_path} (Not found)")
    
    # Memory check
    print_section("💾 Memory Status")
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    import psutil
    mem = psutil.virtual_memory()
    print(f"System RAM: {mem.total / 1024**3:.2f} GB")
    print(f"Available RAM: {mem.available / 1024**3:.2f} GB")
    print(f"Used RAM: {mem.percent}%")
    
    # Final status
    print_section("📊 Installation Status")
    if all_good:
        print("✅ All core packages installed successfully!")
        print("✅ System is ready for AI Image/Video generation!")
    else:
        print("⚠️  Some packages are missing. Please run install_fixed.sh")
    
    print("\n" + "="*60)
    print("  Next steps:")
    print("  1. Upload LORA model files if not present")
    print("  2. Run: python main.py")
    print("  3. Access API at: http://localhost:8000")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Install psutil if not present
    try:
        import psutil
    except ImportError:
        print("Installing psutil for system monitoring...")
        os.system("pip install psutil")
        import psutil
    
    main()