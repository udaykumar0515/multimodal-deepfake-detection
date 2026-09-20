import sys
import os
import json
import pkg_resources

def get_package_version(package_name):
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return "Not Installed"

def diagnose():
    info = {
        "python_version": sys.version,
        "packages": {
            "torch": get_package_version("torch"),
            "torchvision": get_package_version("torchvision"),
            "onnxruntime": get_package_version("onnxruntime"),
            "onnxruntime-gpu": get_package_version("onnxruntime-gpu"),
            "insightface": get_package_version("insightface"),
            "opencv-python": get_package_version("opencv-python")
        },
        "pytorch_cuda": {},
        "onnx_providers": []
    }
    
    try:
        import torch
        info["pytorch_cuda"] = {
            "is_available": torch.cuda.is_available(),
            "version": torch.version.cuda,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, 'cudnn') else None
        }
    except ImportError:
        info["pytorch_cuda"] = "PyTorch not installed or failed to import"

    try:
        import onnxruntime as ort
        info["onnx_providers"] = ort.get_available_providers()
    except ImportError:
        info["onnx_providers"] = "ONNX Runtime not installed"

    # Print nicely formatted output
    print(json.dumps(info, indent=4))

if __name__ == "__main__":
    diagnose()
