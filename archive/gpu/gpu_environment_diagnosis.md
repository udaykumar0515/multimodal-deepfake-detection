# GPU Environment Diagnosis Report

## 1. Environment Information
- **Python Version:** 3.14.5 (64-bit)
- **InsightFace Version:** 2.0
- **OpenCV Version:** 4.13.0.92

## 2. GPU Hardware Information (via nvidia-smi)
- **GPU Name:** NVIDIA GeForce RTX 4050 Laptop GPU
- **NVIDIA Driver Version:** 546.18
- **Maximum Supported CUDA Version:** 12.3
- **Total VRAM:** 6141 MiB (6 GB)
- **Status:** The GPU is correctly recognized by the Windows operating system and the NVIDIA drivers are functioning properly.

## 3. PyTorch Configuration
- **Installed Package:** `torch==2.14.0+cpu`
- **Installed Vision Package:** `torchvision==0.29.0+cpu`
- **`torch.cuda.is_available()`:** `False`
- **`torch.cuda.device_count()`:** 0
- **Diagnosis:** The Python environment is currently using the **CPU-only** distribution of PyTorch. It does not contain the necessary CUDA libraries to communicate with the NVIDIA GPU.

## 4. ONNX Runtime Configuration
- **Installed Package:** `onnxruntime==1.30.0` (CPU version)
- **Missing Package:** `onnxruntime-gpu` is **Not Installed**.
- **Available Providers:** `['AzureExecutionProvider', 'CPUExecutionProvider']`
- **Diagnosis:** `CUDAExecutionProvider` is entirely missing because the GPU-enabled version of ONNX Runtime is not installed.

## 5. Root Cause of GPU Unavailability
The RTX 4050 GPU is physically present and recognized by the OS, but the Python environment is equipped exclusively with CPU-only libraries. 
1. `torch` and `torchvision` were installed from the default CPU wheel repository (indicated by the `+cpu` tag).
2. `onnxruntime` was installed instead of `onnxruntime-gpu`.

## 6. Proposed Fix
To enable GPU acceleration for InsightFace (RetinaFace) and subsequent model training, the following precise package replacements are required:

1. **Uninstall CPU packages:**
   ```bash
   pip uninstall -y torch torchvision onnxruntime
   ```

2. **Install PyTorch with CUDA 12.1 (or compatible) support:**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install ONNX Runtime GPU:**
   ```bash
   pip install onnxruntime-gpu
   ```

**Risks/Compatibility Concerns:**
- The current NVIDIA driver (546.18) supports up to CUDA 12.3, meaning PyTorch CUDA 12.1 binaries will be fully backward-compatible.
- `insightface` automatically selects `CUDAExecutionProvider` if `onnxruntime-gpu` is installed.
- No changes to the project's source code or dataset splits will be required; this is purely an environment dependency swap.
