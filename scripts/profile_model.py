"""Compute GFLOPs (via fvcore) and parameter counts (total vs. trainable) for all branches and fusion."""

import argparse
import os
import torch
from fvcore.nn import FlopCountAnalysis


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def inspect_model(model_name: str, model: torch.nn.Module):
    """Print total and trainable parameter counts for a given module."""
    trainable_params = count_trainable_parameters(model)
    all_params = count_all_parameters(model)
    print(f"--- Inspecting: {model_name} ---")
    print(f"Total parameters (all):         {all_params / 1e6:.2f}M")
    print(f"Trainable parameters (counted): {trainable_params / 1e6:.2f}M")
    if all_params > 0:
        print(f"First parameter requires_grad:  {next(model.parameters()).requires_grad}")
    print("-" * 32)


def profile_checkpoints(model_dir: str, face_channels: int = 1):
    """Load saved branch & fusion models and compute FLOPs and parameter statistics."""
    model_context = torch.load(os.path.join(model_dir, "model_context.pth"), map_location="cpu", weights_only=False).eval()
    model_body = torch.load(os.path.join(model_dir, "model_body.pth"), map_location="cpu", weights_only=False).eval()
    model_face = torch.load(os.path.join(model_dir, "model_face.pth"), map_location="cpu", weights_only=False).eval()
    fusion_model = torch.load(os.path.join(model_dir, "model_fusion.pth"), map_location="cpu", weights_only=False).eval()

    dummy_input_context = torch.randn(1, 3, 224, 224)
    dummy_input_body = torch.randn(1, 3, 128, 128)
    dummy_input_face = torch.randn(1, face_channels, 48, 48)

    flops_context = FlopCountAnalysis(model_context, dummy_input_context).total()
    flops_body = FlopCountAnalysis(model_body, dummy_input_body).total()
    flops_face = FlopCountAnalysis(model_face, dummy_input_face).total()

    with torch.no_grad():
        features_context = model_context(dummy_input_context)
        features_body = model_body(dummy_input_body)
        features_face = model_face(dummy_input_face)

    dummy_input_fusion = (features_context, features_body, features_face)
    flops_fusion = FlopCountAnalysis(fusion_model, dummy_input_fusion).total()
    total_flops = flops_context + flops_body + flops_face + flops_fusion

    print(f"Context FLOPs: {flops_context / 1e9:.4f} GFLOPs")
    print(f"Body FLOPs:    {flops_body / 1e9:.4f} GFLOPs")
    print(f"Face FLOPs:    {flops_face / 1e9:.4f} GFLOPs")
    print(f"Fusion FLOPs:  {flops_fusion / 1e9:.6f} GFLOPs")
    print(f"TOTAL FLOPs:   {total_flops / 1e9:.2f} GFLOPs\n")

    inspect_model("Context Model", model_context)
    inspect_model("Body Model", model_body)
    inspect_model("Face Model", model_face)
    inspect_model("Fusion Model", fusion_model)

    for m in (model_context, model_body, model_face, fusion_model):
        for p in m.parameters():
            p.requires_grad = True

    params_context = count_trainable_parameters(model_context)
    params_body = count_trainable_parameters(model_body)
    params_face = count_trainable_parameters(model_face)
    params_fusion = count_trainable_parameters(fusion_model)
    total_params = params_context + params_body + params_face + params_fusion

    print("\n--- Recalculating After Unfreezing All Branches ---")
    print(f"Context Model: {params_context / 1e6:.2f}M")
    print(f"Body Model:    {params_body / 1e6:.2f}M")
    print(f"Face Model:    {params_face / 1e6:.2f}M")
    print(f"Fusion Model:  {params_fusion / 1e6:.2f}M")
    print("------------------------")
    print(f"TOTAL PARAMS:  {total_params / 1e6:.2f}M")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile FLOPs and Parameters of Multi-Branch Emotion Network")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model_*.pth checkpoints")
    parser.add_argument("--face_channels", type=int, default=1, help="Number of input channels for Face branch (1 for S-FER, 3 for B-FER)")
    args = parser.parse_args()
    profile_checkpoints(args.model_dir, face_channels=args.face_channels)
