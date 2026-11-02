"""Branch 1: Body Feature Extraction backbones (ResNet-18, ResNet-50, Swin Transformer V2 Tiny)."""

import torch
import torchvision.models as models


def build_body_backbone(backbone_type: str = "swin_v2_t", checkpoint_path: str = None):
    """
    Build the Body Feature Extraction branch and return (model_body, num_body_features, is_swin).

    Supported `backbone_type` values:
    - 'resnet18': ResNet-18 pretrained on ImageNet1K
    - 'resnet50': ResNet-50 pretrained on ImageNet1K (or custom EMOTIC checkpoint)
    - 'swin_v2_t': Swin Transformer V2 Tiny pretrained on ImageNet1K (or custom EMOTIC checkpoint)
    """
    is_swin = False

    if checkpoint_path is not None:
        model_body = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        if "swin" in backbone_type.lower():
            is_swin = True
    elif backbone_type == "resnet18":
        model_body = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif backbone_type == "resnet50":
        model_body = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif backbone_type == "swin_v2_t":
        model_body = models.swin_v2_t(weights="DEFAULT")
        is_swin = True
    else:
        raise ValueError(f"Unsupported body backbone_type: {backbone_type}")

    num_body_features = list(model_body.children())[-1].in_features
    return model_body, num_body_features, is_swin
