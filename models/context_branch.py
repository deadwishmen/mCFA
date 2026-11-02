"""Branch 2: Context Feature Extraction backbones (ResNet-18 / ResNet-50 trained on Places365)."""

import os
import torch
import torchvision.models as models


def convert_places365_checkpoint(arch: str, tar_path: str, save_state_dict_path: str):
    """
    Load a Places365 `.pth.tar` checkpoint, strip 'module.' prefixes from DataParallel keys,
    and save a clean state_dict `.pth` file.
    """
    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(tar_path, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, "module.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    model.cpu()
    os.makedirs(os.path.dirname(save_state_dict_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), save_state_dict_path)
    return model


def build_context_backbone(arch: str = "resnet50", state_dict_path: str = None):
    """
    Build the Context Feature Extraction branch (ResNet-18 or ResNet-50 for 365 Places classes).
    Returns (model_context, num_context_features).
    """
    if arch not in ("resnet18", "resnet50"):
        raise ValueError(f"Unsupported context architecture: {arch}. Choose 'resnet18' or 'resnet50'.")

    model_context = models.__dict__[arch](num_classes=365)
    if state_dict_path is not None and os.path.exists(state_dict_path):
        context_state_dict = torch.load(state_dict_path, map_location=lambda storage, loc: storage)
        if "state_dict" in context_state_dict:
            context_state_dict = {
                str.replace(k, "module.", ""): v for k, v in context_state_dict["state_dict"].items()
            }
        model_context.load_state_dict(context_state_dict)

    num_context_features = list(model_context.children())[-1].in_features
    return model_context, num_context_features
