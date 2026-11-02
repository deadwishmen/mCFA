"""Single-sample inference and visualization on EMOTIC test samples."""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from configs.config import EMOTIC_CATEGORIES, get_norms
from datasets.emotic_dataset import get_transforms, load_split_arrays


def test_disc_pre(
    models_list,
    device: torch.device,
    image_context,
    image_body,
    image_face,
    thresholds: torch.Tensor,
    context_norm,
    body_norm,
    face_norm,
    test_transform,
    face_test_transform,
    is_bfer: bool = False,
):
    """Predict discrete emotion categories for a single (context, body, face) sample."""
    context_norm_tran = transforms.Normalize(context_norm[0], context_norm[1])
    body_norm_tran = transforms.Normalize(body_norm[0], body_norm[1])
    face_norm_tran = transforms.Normalize(face_norm[0], face_norm[1])

    image_context = context_norm_tran(test_transform(image_context)).unsqueeze(0)
    image_body = body_norm_tran(test_transform(image_body)).unsqueeze(0)
    image_face = face_norm_tran(face_test_transform(image_face)).unsqueeze(0)

    model_context, model_body, model_face, fusion_model = models_list

    with torch.no_grad():
        model_context.to(device).eval()
        model_body.to(device).eval()
        model_face.to(device).eval()
        fusion_model.to(device).eval()

        image_context = image_context.to(device)
        image_body = image_body.to(device)
        image_face = image_face.to(device)
        if not is_bfer:
            image_face = torch.mean(image_face, dim=1, keepdim=True).to(device)

        pred_context = model_context(image_context)
        pred_body = model_body(image_body)
        pred_face = model_face(image_face)
        pred_cat = fusion_model(pred_context, pred_body, pred_face)

        pred_cat = pred_cat.squeeze(0)
        bool_cat_pred = torch.gt(pred_cat, thresholds)

    result = [EMOTIC_CATEGORIES[i] for i, value in enumerate(bool_cat_pred) if value]
    print("Predicted Emotions:", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run single-sample emotion prediction on EMOTIC test set")
    parser.add_argument("--data_src", type=str, required=True, help="Directory containing preprocessed EMOTIC .npy arrays")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model_*.pth and thresholds.npy")
    parser.add_argument("--sample_id", type=int, default=1122, help="Index of test image to predict")
    parser.add_argument("--is_swin", action="store_true", default=True, help="Use SwinT normalization and transforms")
    parser.add_argument("--is_bfer", action="store_true", default=False, help="Use 3-channel face input")
    parser.add_argument("--save_vis", type=str, default=None, help="Optional path to save context/body visualization")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    context_norm, body_norm, face_norm = get_norms(is_swin=args.is_swin)
    _, test_transform, _, face_test_transform = get_transforms(is_swin=args.is_swin)

    test_context, test_body, test_face, test_cat, _ = load_split_arrays(args.data_src, "test")

    model_context = torch.load(os.path.join(args.model_dir, "model_context.pth"), map_location=device, weights_only=False)
    model_body = torch.load(os.path.join(args.model_dir, "model_body.pth"), map_location=device, weights_only=False)
    model_face = torch.load(os.path.join(args.model_dir, "model_face.pth"), map_location=device, weights_only=False)
    fusion_model = torch.load(os.path.join(args.model_dir, "model_fusion.pth"), map_location=device, weights_only=False)
    thresholds = torch.FloatTensor(np.load(os.path.join(args.model_dir, "thresholds.npy"))).to(device)

    idx = args.sample_id
    image_context = test_context[idx]
    image_body = test_body[idx]
    image_face = test_face[idx]
    label = test_cat[idx]

    gt_emotions = [EMOTIC_CATEGORIES[i] for i, val in enumerate(label) if val == 1.0]
    print("Ground Truth Emotions:", gt_emotions)

    pred_emotions = test_disc_pre(
        [model_context, model_body, model_face, fusion_model],
        device=device,
        image_context=image_context,
        image_body=image_body,
        image_face=image_face,
        thresholds=thresholds,
        context_norm=context_norm,
        body_norm=body_norm,
        face_norm=face_norm,
        test_transform=test_transform,
        face_test_transform=face_test_transform,
        is_bfer=args.is_bfer,
    )

    if args.save_vis:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(image_context)
        axes[0].set_title("Context")
        axes[0].axis("off")
        axes[1].imshow(image_body)
        axes[1].set_title("Body")
        axes[1].axis("off")
        fig.suptitle(f"GT: {', '.join(gt_emotions)}\nPred: {', '.join(pred_emotions)}", fontsize=9)
        plt.tight_layout()
        plt.savefig(args.save_vis, dpi=150)
        print(f"Saved visualization to {args.save_vis}")


if __name__ == "__main__":
    main()
