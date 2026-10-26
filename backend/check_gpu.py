import sys
import torch

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
    a = torch.rand(128, 128, device="cuda")
    b = a @ a
    torch.cuda.synchronize()
    print("CUDA matrix multiplication passed:", tuple(b.shape))
else:
    print("CPU fallback is available. For GPU training, install a CUDA-enabled PyTorch build.")
