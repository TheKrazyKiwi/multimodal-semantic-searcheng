import torch
import sys

def check_gpu():
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print("\n[SUCCESS] CUDA is available!")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        
        # Check VRAM
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        free_mem = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)
        
        print(f"Total VRAM: {total_mem:.2f} GB")
        print(f"Free VRAM: {free_mem:.2f} GB")
        
        if total_mem < 4.5:
            print("\n[WARNING] VRAM is <= 4.5 GB. Strict memory optimization is REQUIRED.")
        
    else:
        print("\n[ERROR] CUDA is NOT available. Training will be extremely slow.")
        sys.exit(1)

if __name__ == "__main__":
    check_gpu()
