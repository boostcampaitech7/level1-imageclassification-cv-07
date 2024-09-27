import platform
import sys
import torch
import subprocess

# OS 정보
os_info = platform.platform()
print(f"Operating System: {os_info}")

# Python 버전
python_version = sys.version
print(f"Python Version: {python_version}")

# GPU 정보 (PyTorch가 설치되어 있을 경우)
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_capability = torch.cuda.get_device_capability(i)
        print(f"GPU {i}: {gpu_name}, Compute Capability: {gpu_capability}")
else:
    print("No GPU available")

# NVIDIA GPU 상세 정보 (nvidia-smi 명령어를 사용하는 경우)
try:
    nvidia_info = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
    print("NVIDIA GPU Info:\n", nvidia_info)
except FileNotFoundError:
    print("nvidia-smi command not found. Are you sure you have an NVIDIA GPU?")