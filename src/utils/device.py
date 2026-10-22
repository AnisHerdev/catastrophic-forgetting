"""
Hardware and Acceleration Device Detection & Diagnostics.
Supports automatic resolution of CUDA (NVIDIA), MPS (Apple Silicon), and CPU devices,
with physical GPU hardware inspection and actionable acceleration recommendations.
"""

import os
import platform
import subprocess
from typing import Any, Dict, List, Optional, Tuple
import torch


def detect_physical_gpus() -> List[str]:
    """
    Queries the host operating system directly for physical graphics adapters.
    Works independently of PyTorch or CUDA installation.
    """
    names: List[str] = []
    sys_name = platform.system()
    try:
        if sys_name == "Windows":
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if res.returncode == 0:
                names = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        elif sys_name == "Linux":
            res = subprocess.run(["lspci"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if any(kw in line.upper() for kw in ("VGA", "3D", "DISPLAY", "NVIDIA")):
                        names.append(line.strip())
    except Exception:
        pass
    return names


def get_device_info(device: torch.device) -> Dict[str, Any]:
    """
    Inspects and returns a structured dictionary of hardware and acceleration specifications.
    """
    info: Dict[str, Any] = {
        "device": device,
        "device_type": device.type,
        "is_cuda": device.type == "cuda",
        "is_mps": device.type == "mps",
        "is_cpu": device.type == "cpu",
    }

    if device.type == "cuda" and torch.cuda.is_available():
        device_idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_idx)

        info["gpu_name"] = torch.cuda.get_device_name(device_idx)
        info["device_index"] = device_idx
        info["device_count"] = torch.cuda.device_count()
        info["cuda_version"] = torch.version.cuda
        info["compute_capability"] = f"{props.major}.{props.minor}"

        total_bytes = props.total_memory
        info["total_vram_bytes"] = total_bytes
        info["total_vram_gb"] = total_bytes / (1024 ** 3)
        info["total_vram_mb"] = total_bytes / (1024 ** 2)

        try:
            free_bytes, _ = torch.cuda.mem_get_info(device_idx)
            info["free_vram_bytes"] = free_bytes
            info["free_vram_gb"] = free_bytes / (1024 ** 3)
            info["free_vram_mb"] = free_bytes / (1024 ** 2)
        except Exception:
            info["free_vram_gb"] = None
            info["free_vram_mb"] = None

    elif device.type == "mps":
        info["gpu_name"] = "Apple Silicon Metal Acceleration (MPS)"
        info["device_count"] = 1

    else:
        info["cpu_arch"] = platform.machine()
        info["cpu_processor"] = platform.processor()
        info["cpu_threads"] = torch.get_num_threads()
        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"] = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )

        # Inspect physical system hardware for an unutilized NVIDIA GPU
        physical_gpus = detect_physical_gpus()
        info["physical_gpus"] = physical_gpus
        nvidia_keywords = ("NVIDIA", "GEFORCE", "RTX", "GTX", "QUADRO", "TESLA", "TITAN")
        nvidia_gpus = [
            g for g in physical_gpus
            if any(kw in g.upper() for kw in nvidia_keywords)
        ]
        info["nvidia_physical_gpus"] = nvidia_gpus
        info["has_unutilized_nvidia_gpu"] = len(nvidia_gpus) > 0 and not torch.cuda.is_available()

    return info


def resolve_device(requested_device: Optional[str] = None) -> Tuple[torch.device, Dict[str, Any]]:
    """
    Resolves the computation device based on user preference or automatic hardware detection.
    Priority order for 'auto': CUDA GPU -> Apple Silicon MPS -> CPU.

    Raises:
        RuntimeError: If user explicitly requests 'cuda' or 'mps' when unavailable.
    """
    if requested_device is None or str(requested_device).lower() in ("auto", "none", ""):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        req = str(requested_device).lower().strip()
        if req == "cuda" or req.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA device requested ('--device {requested_device}'), but torch.cuda.is_available() is False.\n"
                    "Possible causes:\n"
                    " 1. No NVIDIA GPU is installed in this machine.\n"
                    " 2. NVIDIA drivers are missing or not updated.\n"
                    " 3. PyTorch was installed with CPU-only binaries.\n"
                    "To install the CUDA build: uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
                    "To run on CPU instead, use '--device cpu' or omit the flag for auto-detection."
                )
            device = torch.device(requested_device)
        elif req == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError(
                    "MPS (Apple Silicon GPU) requested ('--device mps'), but torch.backends.mps is not available.\n"
                    "Use '--device cpu' or '--device auto'."
                )
            device = torch.device("mps")
        elif req == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device(requested_device)

    info = get_device_info(device)
    return device, info


def print_device_banner(device_info: Dict[str, Any]) -> None:
    """
    Prints a clear, informative summary of the active acceleration hardware.
    Uses ASCII characters to ensure 100% compatibility across all terminal encodings (e.g. Windows cp1252).
    """
    device: torch.device = device_info["device"]
    border = "=" * 80
    print("\n" + border)
    print("  HARDWARE ACCELERATION & COMPUTE DEVICE")
    print(border)

    if device_info["is_cuda"]:
        print(f"  Active Device        : GPU (CUDA - {device})")
        print(f"  GPU Model            : {device_info.get('gpu_name', 'NVIDIA GPU')}")
        print(f"  Device Index         : {device_info.get('device_index', 0)} of {device_info.get('device_count', 1)} GPU(s)")
        print(f"  CUDA Runtime Version : {device_info.get('cuda_version', 'N/A')}")
        print(f"  Compute Capability   : {device_info.get('compute_capability', 'N/A')}")

        total_gb = device_info.get("total_vram_gb")
        free_gb = device_info.get("free_vram_gb")
        if total_gb is not None:
            print(f"  Total VRAM           : {total_gb:.2f} GB ({device_info.get('total_vram_mb', 0):,.0f} MB)")
        if free_gb is not None:
            print(f"  Available / Free VRAM: {free_gb:.2f} GB ({device_info.get('free_vram_mb', 0):,.0f} MB)")
        print(f"  Status               : GPU acceleration active - high-speed training enabled.")

    elif device_info["is_mps"]:
        print(f"  Active Device        : GPU (Apple Silicon Metal / MPS)")
        print(f"  GPU Model            : {device_info.get('gpu_name', 'Apple Silicon GPU')}")
        print(f"  Status               : GPU acceleration active via Apple Metal.")

    else:
        # Check if an unutilized physical NVIDIA GPU was detected on this machine
        if device_info.get("has_unutilized_nvidia_gpu"):
            nvidia_card = device_info["nvidia_physical_gpus"][0]
            print(f"  Active Device        : CPU (Current PyTorch build: {torch.__version__})")
            print(f"  Physical GPU Found   : {nvidia_card}")
            print(f"")
            print(f"  ----------------------------------------------------------------------------")
            print(f"  * NOTICE: PHYSICAL NVIDIA GPU DETECTED, BUT PYTORCH IS CPU-ONLY")
            print(f"  ----------------------------------------------------------------------------")
            print(f"  A dedicated NVIDIA GPU was found on this system, but your environment is")
            print(f"  currently running the CPU build of PyTorch without CUDA acceleration.")
            print(f"")
            print(f"  To unlock 15x-50x faster GPU training on this PC, copy and run:")
            print(f"    uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124")
            print(f"")
            print(f"  Continuing execution on CPU for now so your run does not halt...")
            print(f"  ----------------------------------------------------------------------------")
        else:
            print(f"  Active Device        : CPU")
            arch = device_info.get("cpu_arch") or platform.machine()
            threads = device_info.get("cpu_threads", "N/A")
            print(f"  CPU Architecture     : {arch} ({platform.system()} {platform.release()})")
            print(f"  CPU Worker Threads   : {threads}")
            print(f"  GPU Detected         : No CUDA or MPS GPU available to PyTorch.")
            print(f"  Status               : Running on CPU.")

    print(border + "\n")
