import torch
import torch.nn as nn
from torchvision import models

# 1. Load the architecture exactly as before
NUM_CLASSES = 11
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load("weather_classifier.pth", map_location="cpu", weights_only=True))
model.eval()

# 2. Create a dummy input tensor representing a 224x224 RGB image
dummy_input = torch.randn(1, 3, 224, 224)

# 3. Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "weather_classifier.onnx",
    export_params=True,
    opset_version=14,          # Standard compatibility level
    do_constant_folding=True,  # Optimizes the graph for faster inference
    input_names=['input'],
    output_names=['output']
)

print("Export complete: weather_classifier.onnx")