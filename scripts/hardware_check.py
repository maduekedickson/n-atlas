"""Hardware and environment check script for NAIC 2026 N-ATLaS Innovation Challenge.

Detects:
- Operating System & Architecture
- Python version
- CPU name, logical cores, physical cores
- System RAM (Total, Available, Used)
- GPU availability, CUDA status, VRAM
- Storage space on the current drive
- PyTorch and Hugging Face library status (if installed)
"""

import sys
import platform
import shutil
import psutil
import json
from datetime import datetime

def get_hardware_info():
    info = {
        "timestamp": datetime.now().isoformat(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
        },
        "ram": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
            "percent_used": psutil.virtual_memory().percent,
        },
        "disk": {},
        "gpu": {
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
        },
        "libraries": {},
    }

    # Disk usage
    total, used, free = shutil.disk_usage(".")
    info["disk"]["total_gb"] = round(total / (1024**3), 2)
    info["disk"]["used_gb"] = round(used / (1024**3), 2)
    info["disk"]["free_gb"] = round(free / (1024**3), 2)

    # Check PyTorch & CUDA
    try:
        import torch
        info["libraries"]["torch_version"] = torch.__version__
        cuda_avail = torch.cuda.is_available()
        info["gpu"]["cuda_available"] = cuda_avail
        if cuda_avail:
            info["gpu"]["device_count"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["gpu"]["devices"].append({
                    "id": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "major_cuda_capability": props.major,
                    "minor_cuda_capability": props.minor,
                })
        else:
            info["gpu"]["note"] = "No CUDA-capable GPU detected by PyTorch"
    except ImportError:
        info["libraries"]["torch_version"] = "Not installed"

    # Transformers check
    try:
        import transformers
        info["libraries"]["transformers_version"] = transformers.__version__
    except ImportError:
        info["libraries"]["transformers_version"] = "Not installed"

    return info

def print_summary(info):
    print("=" * 60)
    print("NAIC 2026 — N-ATLaS System Hardware & Environment Check")
    print("=" * 60)
    print(f"Timestamp        : {info['timestamp']}")
    print(f"OS               : {info['os']['system']} {info['os']['release']} ({info['os']['machine']})")
    print(f"Python Version   : {info['python']['version'].split()[0]}")
    print(f"CPU              : {info['os']['processor']} ({info['cpu']['physical_cores']} physical / {info['cpu']['logical_cores']} logical cores)")
    print(f"RAM Total        : {info['ram']['total_gb']} GB")
    print(f"RAM Available    : {info['ram']['available_gb']} GB ({info['ram']['percent_used']}% used)")
    print(f"Disk Free        : {info['disk']['free_gb']} GB (of {info['disk']['total_gb']} GB)")
    print("-" * 60)
    print(f"PyTorch Version  : {info['libraries'].get('torch_version', 'Not installed')}")
    print(f"CUDA Available   : {info['gpu']['cuda_available']}")

    if info['gpu']['cuda_available']:
        for dev in info['gpu']['devices']:
            print(f"GPU {dev['id']}            : {dev['name']} ({dev['total_memory_gb']} GB VRAM)")
    else:
        print(f"GPU Status       : No NVIDIA CUDA GPU available.")
    
    print("-" * 60)
    print("FEASIBILITY ASSESSMENT FOR N-ATLaS (8B Model):")
    if not info['gpu']['cuda_available']:
        print("[!] Local CUDA GPU: MISSING.")
        print("    Running an 8B LLaMA model locally in full or 4-bit precision requires an NVIDIA GPU.")
        print("    Local CPU execution requires >= 24GB free RAM and will suffer severe latency.")
        print("    RECOMMENDATION: Use Google Colab (T4 / A100 GPU) for model training,")
        print("    baseline benchmarking, and inference testing.")
    else:
        vram = info['gpu']['devices'][0]['total_memory_gb']
        if vram >= 16:
            print("[+] VRAM >= 16GB: Capable of full FP16 inference and QLoRA fine-tuning.")
        elif vram >= 8:
            print("[+] VRAM >= 8GB: Capable of 4-bit (bitsandbytes) inference and QLoRA.")
        else:
            print("[!] VRAM < 8GB: Insufficient for comfortable 8B model operations.")
    print("=" * 60)

if __name__ == "__main__":
    data = get_hardware_info()
    print_summary(data)
    with open("scripts/hardware_info.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Hardware profile saved to scripts/hardware_info.json")
