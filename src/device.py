"""Shared PyTorch device selection for the vision models."""
import os


def select_torch_device(requested=None):
    requested = (requested if requested is not None else os.getenv("RAVEN_DEVICE", "auto")).strip().lower()
    if requested not in {"auto", "cpu", "cuda", "mps"} and not requested.isdigit():
        raise ValueError("RAVEN_DEVICE must be auto, cpu, cuda, mps, or a CUDA device index")
    try:
        import torch
    except ImportError:
        if requested not in {"auto", "cpu"}:
            raise RuntimeError("A non-CPU RAVEN_DEVICE requires PyTorch")
        return "cpu", False
    cuda = torch.cuda.is_available()
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if requested == "auto":
        device = "0" if cuda else "mps" if mps else "cpu"
    elif requested.isdigit():
        if not cuda:
            raise RuntimeError("A numeric RAVEN_DEVICE requires CUDA")
        device = requested
    elif requested == "cuda":
        if not cuda:
            raise RuntimeError("RAVEN_DEVICE=cuda but CUDA is unavailable")
        device = "0"
    elif requested == "mps":
        if not mps:
            raise RuntimeError("RAVEN_DEVICE=mps but MPS is unavailable")
        device = "mps"
    else:
        device = "cpu"
    return device, bool(cuda and device != "cpu")


def select_onnx_device(available, requested=None):
    requested = (requested if requested is not None else os.getenv("RAVEN_ONNX_PROVIDER", "auto")).strip()
    providers = {"CPUExecutionProvider", "CUDAExecutionProvider"}
    if requested != "auto" and requested not in providers:
        raise ValueError("RAVEN_ONNX_PROVIDER must be auto, CPUExecutionProvider, or CUDAExecutionProvider")
    if requested == "auto":
        requested = "CUDAExecutionProvider" if "CUDAExecutionProvider" in available else "CPUExecutionProvider"
    if requested not in available:
        raise RuntimeError(f"Requested ONNX provider {requested} is unavailable; available: {available}")
    if requested == "CUDAExecutionProvider":
        return os.getenv("RAVEN_CUDA_DEVICE", "0"), requested
    return "cpu", requested
