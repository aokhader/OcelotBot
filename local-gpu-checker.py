import torch
import subprocess

print("="*70)
print("GPU DIAGNOSTIC CHECK")
print("="*70)

# Check if CUDA is available
print("\n[1] CUDA Availability:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n[2] GPU {i} Details:")
        print(f"   Name: {torch.cuda.get_device_name(i)}")
        print(f"   Compute Capability: {torch.cuda.get_device_capability(i)}")
        print(f"   Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        
        # Check current memory usage
        if torch.cuda.is_initialized():
            print(f"   Allocated Memory: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"   Cached Memory: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    
    # Test CUDA with a simple operation
    print("\n[3] CUDA Functionality Test:")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("   ✅ CUDA tensor operations working")
    except Exception as e:
        print(f"   ❌ CUDA test failed: {e}")
else:
    print("\n❌ CUDA is NOT available")
    print("\nPossible reasons:")
    print("1. No NVIDIA GPU installed")
    print("2. NVIDIA drivers not installed")
    print("3. PyTorch installed without CUDA support")
    print("4. CUDA toolkit version mismatch")

# Check NVIDIA driver
print("\n[4] NVIDIA Driver Check:")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("   ✅ NVIDIA driver installed")
        print("\nGPU Information from nvidia-smi:")
        print("-" * 70)
        print(result.stdout)
    else:
        print("   ❌ nvidia-smi command failed")
except FileNotFoundError:
    print("   ❌ nvidia-smi not found (driver not installed)")

# Recommendations
print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

if not torch.cuda.is_available():
    print("\n⚠️  CUDA is not available. To enable GPU training:")
    print("\n1. Check if you have an NVIDIA GPU:")
    print("   - Run: lspci | grep -i nvidia")
    print("\n2. Install NVIDIA drivers:")
    print("   - Ubuntu: sudo apt install nvidia-driver-XXX")
    print("   - Check available versions: ubuntu-drivers devices")
    print("\n3. Install PyTorch with CUDA support:")
    print("   - Visit: https://pytorch.org/get-started/locally/")
    print("   - Choose your CUDA version and install command")
    print("   - Example: pip install torch --index-url https://download.pytorch.org/whl/cu118")
else:
    gpu_name = torch.cuda.get_device_name(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print("\n✅ CUDA is available and working!")
    print(f"\nYour GPU: {gpu_name}")
    print(f"Memory: {memory_gb:.2f} GB")
    
    # Memory recommendations
    if memory_gb < 6:
        print("\n⚠️  WARNING: Your GPU has limited memory (<6GB)")
        print("   Recommendations:")
        print("   - Reduce batch size to 1 or 2")
        print("   - Enable gradient checkpointing")
        print("   - Consider 4-bit quantization")
        print("   - Consider using cloud services like Google Colab for training")
    elif memory_gb < 12:
        print("\n✅ Your GPU should handle training with some adjustments")
        print("   Recommendations:")
        print("   - Use batch size 2-4")
        print("   - Enable gradient checkpointing if needed")
        print("   - Consider using cloud services like Google Colab for training")
    else:
        print("\n✅ Your GPU has plenty of memory for training!")
        print("   You can use the default settings")