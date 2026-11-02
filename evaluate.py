"""Evaluate the Multi-Branch Emotion Prediction model on the EMOTIC test dataset."""

import argparse
import os
import numpy as np
import torch

from datasets.emotic_dataset import build_dataloaders
from utils.metrics import test_scikit_ap, get_thresholds


def test_disc(
    models_list,
    device: torch.device,
    data_loader,
    num_images: int,
    is_bfer: bool = False,
    thresholds_save_path: str = "./thresholds.npy",
) -> float:
    """Run inference over `data_loader` and compute mean Average Precision (mAP)."""
    model_context, model_body, model_face, fusion_model = models_list
    cat_preds = np.zeros((num_images, 26))
    cat_labels = np.zeros((num_images, 26))

    with torch.no_grad():
        model_context.to(device).eval()
        model_body.to(device).eval()
        model_face.to(device).eval()
        fusion_model.to(device).eval()

        indx = 0
        print("Starting testing...")
        for images_context, images_body, images_face, labels_cat, labels_cont in iter(data_loader):
            images_context = images_context.to(device)
            images_body = images_body.to(device)
            images_face = images_face.to(device)
            if not is_bfer:
                images_face = torch.mean(images_face, dim=1, keepdim=True).to(device)

            labels_cat = labels_cat.to(device)

            pred_context = model_context(images_context)
            pred_body = model_body(images_body)
            pred_face = model_face(images_face)
            pred_cat = fusion_model(pred_context, pred_body, pred_face)

            batch_sz = pred_cat.shape[0]
            cat_preds[indx : (indx + batch_sz), :] = pred_cat.to("cpu").data.numpy()
            cat_labels[indx : (indx + labels_cat.shape[0]), :] = labels_cat.to("cpu").data.numpy()
            indx += batch_sz

    cat_preds = cat_preds.transpose()
    cat_labels = cat_labels.transpose()
    print("Completed testing.")
    ap_mean = test_scikit_ap(cat_preds, cat_labels)
    get_thresholds(cat_preds, cat_labels, save_path=thresholds_save_path)
    return ap_mean


def main():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Branch Emotion Prediction on EMOTIC")
    parser.add_argument("--data_src", type=str, required=True, help="Directory containing preprocessed EMOTIC .npy arrays")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model_context.pth, model_body.pth, model_face.pth, model_fusion.pth")
    parser.add_argument("--batch_size", type=int, default=26, help="Evaluation batch size")
    parser.add_argument("--is_swin", action="store_true", default=True, help="Use SwinT normalization and transforms")
    parser.add_argument("--is_bfer", action="store_true", default=False, help="Keep 3 channels for face input (B-FER / VGG-FER)")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    (_, _, test_dataset), (_, _, test_loader) = build_dataloaders(
        args.data_src, batch_size=args.batch_size, is_swin=args.is_swin
    )

    model_context = torch.load(os.path.join(args.model_dir, "model_context.pth"), map_location=device, weights_only=False)
    model_body = torch.load(os.path.join(args.model_dir, "model_body.pth"), map_location=device, weights_only=False)
    model_face = torch.load(os.path.join(args.model_dir, "model_face.pth"), map_location=device, weights_only=False)
    fusion_model = torch.load(os.path.join(args.model_dir, "model_fusion.pth"), map_location=device, weights_only=False)

    test_map = test_disc(
        [model_context, model_body, model_face, fusion_model],
        device=device,
        data_loader=test_loader,
        num_images=len(test_dataset),
        is_bfer=args.is_bfer,
    )
    print(f"Testing mAP = {test_map:.4f}")


if __name__ == "__main__":
    main()
