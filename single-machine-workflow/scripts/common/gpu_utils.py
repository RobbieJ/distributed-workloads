"""GPU and CUDA setup utilities for DGX Spark (GB10)."""

import os
import warnings


def setup_cuda_env():
    """Configure environment variables for DGX Spark GPU compatibility."""
    os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    # Suppress the incorrect CUDA capability warning on GB10
    warnings.filterwarnings("ignore", message=".*CUDA capability.*")


def get_device():
    """Return the best available torch device."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_gpu_info():
    """Print GPU information if available."""
    import torch

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_mem
        print(f"GPU Memory: {total / 1e9:.1f} GB")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    else:
        print("No GPU available, running on CPU")
