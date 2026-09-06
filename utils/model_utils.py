import torch
import torch.nn as nn
from torchvision import models
from typing import List, Tuple

def build_model(model_name: str = "mobilenetv2", num_classes: int = 2) -> nn.Module:
    """
    Membangun dan mengembalikan arsitektur model (MobileNetV2, ResNet18, atau SqueezeNet).
    Hanya classifier head yang disesuaikan dengan jumlah class target, sedangkan base layer di-freeze.
    
    Args:
        model_name (str): Nama arsitektur model ('mobilenetv2', 'resnet18', 'squeezenet').
        num_classes (int): Jumlah kategori yang akan diklasifikasikan.
        
    Returns:
        nn.Module: Model PyTorch.
    """
    name = model_name.lower().replace("_", "").replace("-", "")
    
    if name == "resnet18":
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        # Deep Unfreezing (Gacor Feature)
        for param in model.layer4.parameters():
            param.requires_grad = True
            
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif name == "squeezenet":
        model = models.squeezenet1_0(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        # Deep Unfreezing
        for param in model.features[10:].parameters():
            param.requires_grad = True
            
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1))
        model.num_classes = num_classes
        
    else:  # mobilenetv2 / default
        model = models.mobilenet_v2(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        # Deep Unfreezing
        for param in model.features[-2:].parameters():
            param.requires_grad = True
            
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    return model


def get_device() -> torch.device:
    """
    Mendapatkan device komputasi (CPU/GPU).
    Saat ini di-hardcode ke CPU agar aman di semua environment.
    """
    return torch.device("cpu")
