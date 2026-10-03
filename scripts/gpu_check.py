import torch, onnxruntime as ort
print("Torch version:", torch.__version__)
print("Torch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
print("ONNX providers:", ort.get_available_providers())
