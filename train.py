"""Train the Multi-Branch Network for Imagery Emotion Prediction on the EMOTIC dataset."""

import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from datasets.emotic_dataset import build_dataloaders
from models.body_branch import build_body_backbone
from models.context_branch import build_context_backbone
from models.face_branch import build_face_backbone
from models.fusion import build_fusion_model
from utils.losses import DiscreteLoss, ContinuousLoss_L2, ContinuousLoss_SL1
from utils.metrics import test_scikit_ap, test_emotic_vad, get_thresholds
from evaluate import test_disc


def evaluate_vad(models_list, device: torch.device, data_loader, num_images: int, is_bfer: bool = False) -> float:
    """Run inference over `data_loader` for VAD prediction and compute mean MAE."""
    model_context, model_body, model_face, fusion_model = models_list
    cont_preds = np.zeros((num_images, 3))
    cont_labels = np.zeros((num_images, 3))

    with torch.no_grad():
        model_context.to(device).eval()
        model_body.to(device).eval()
        model_face.to(device).eval()
        fusion_model.to(device).eval()

        indx = 0
        for images_context, images_body, images_face, _, labels_cont in iter(data_loader):
            images_context = images_context.to(device)
            images_body = images_body.to(device)
            images_face = images_face.to(device)
            if not is_bfer:
                images_face = torch.mean(images_face, dim=1, keepdim=True).to(device)

            labels_cont = labels_cont.to(device)

            pred_context = model_context(images_context)
            pred_body = model_body(images_body)
            pred_face = model_face(images_face)
            pred_cont = fusion_model(pred_context, pred_body, pred_face)

            batch_sz = pred_cont.shape[0]
            cont_preds[indx : (indx + batch_sz), :] = pred_cont.to("cpu").data.numpy()
            cont_labels[indx : (indx + labels_cont.shape[0]), :] = labels_cont.to("cpu").data.numpy()
            indx += batch_sz

    cont_preds = cont_preds.transpose()
    cont_labels = cont_labels.transpose()
    return test_emotic_vad(cont_preds, cont_labels)


def train_emotic(
    epochs: int,
    models_list,
    device: torch.device,
    train_loader,
    val_loader,
    num_val_images: int,
    criterion,
    optimizer,
    scheduler,
    save_dir: str,
    is_vad_prediction: bool = False,
    is_bfer: bool = False,
    unfreeze_backbones: bool = False,
):
    """Main training and validation loop for EMOTIC Multi-Branch Network."""
    os.makedirs(save_dir, exist_ok=True)
    model_context, model_body, model_face, fusion_model = models_list

    model_context.to(device)
    model_body.to(device)
    model_face.to(device)
    fusion_model.to(device)

    # Freeze or unfreeze feature extraction branches
    for param in model_context.parameters():
        param.requires_grad = unfreeze_backbones
    for param in model_body.parameters():
        param.requires_grad = unfreeze_backbones
    for param in model_face.parameters():
        param.requires_grad = unfreeze_backbones
    for param in fusion_model.parameters():
        param.requires_grad = True

    min_val_loss = float("inf")
    best_metric = 0.0 if not is_vad_prediction else float("inf")

    print(f"Starting training for {epochs} epochs on device: {device}")
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        fusion_model.train()
        if unfreeze_backbones:
            model_context.train()
            model_body.train()
            model_face.train()
        else:
            model_context.eval()
            model_body.eval()
            model_face.eval()

        for images_context, images_body, images_face, labels_cat, labels_cont in iter(train_loader):
            images_context = images_context.to(device)
            images_body = images_body.to(device)
            images_face = images_face.to(device)
            if not is_bfer:
                images_face = torch.mean(images_face, dim=1, keepdim=True).to(device)

            labels_cat = labels_cat.to(device)
            labels_cont = labels_cont.to(device)

            optimizer.zero_grad()

            if unfreeze_backbones:
                pred_context = model_context(images_context)
                pred_body = model_body(images_body)
                pred_face = model_face(images_face)
            else:
                with torch.no_grad():
                    pred_context = model_context(images_context)
                    pred_body = model_body(images_body)
                    pred_face = model_face(images_face)

            pred = fusion_model(pred_context, pred_body, pred_face)
            target = labels_cont if is_vad_prediction else labels_cat
            loss = criterion(pred, target)

            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation phase
        val_loss = 0.0
        model_context.eval()
        model_body.eval()
        model_face.eval()
        fusion_model.eval()

        with torch.no_grad():
            for images_context, images_body, images_face, labels_cat, labels_cont in iter(val_loader):
                images_context = images_context.to(device)
                images_body = images_body.to(device)
                images_face = images_face.to(device)
                if not is_bfer:
                    images_face = torch.mean(images_face, dim=1, keepdim=True).to(device)

                labels_cat = labels_cat.to(device)
                labels_cont = labels_cont.to(device)

                pred_context = model_context(images_context)
                pred_body = model_body(images_body)
                pred_face = model_face(images_face)
                pred = fusion_model(pred_context, pred_body, pred_face)

                target = labels_cont if is_vad_prediction else labels_cat
                loss = criterion(pred, target)
                val_loss += loss.item()

        train_loss_avg = running_loss / max(len(train_loader), 1)
        val_loss_avg = val_loss / max(len(val_loader), 1)

        if not is_vad_prediction:
            val_metric = test_disc(
                [model_context, model_body, model_face, fusion_model],
                device=device,
                data_loader=val_loader,
                num_images=num_val_images,
                is_bfer=is_bfer,
                thresholds_save_path=os.path.join(save_dir, "thresholds_latest.npy"),
            )
            metric_str = f"val_mAP = {val_metric:.4f}"
            is_best = (val_loss < min_val_loss) or (val_metric > best_metric)
        else:
            val_metric = evaluate_vad(
                [model_context, model_body, model_face, fusion_model],
                device=device,
                data_loader=val_loader,
                num_images=num_val_images,
                is_bfer=is_bfer,
            )
            metric_str = f"val_VAD_MAE = {val_metric:.4f}"
            is_best = val_loss < min_val_loss

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"train_loss = {train_loss_avg:.4f} | val_loss = {val_loss_avg:.4f} | {metric_str}"
        )

        if is_best:
            min_val_loss = min(min_val_loss, val_loss)
            best_metric = max(best_metric, val_metric) if not is_vad_prediction else min(best_metric, val_metric)
            print(f"  -> Saving best model checkpoints to {save_dir}")
            torch.save(model_context, os.path.join(save_dir, "model_context.pth"))
            torch.save(model_body, os.path.join(save_dir, "model_body.pth"))
            torch.save(model_face, os.path.join(save_dir, "model_face.pth"))
            torch.save(fusion_model, os.path.join(save_dir, "model_fusion.pth"))
            if not is_vad_prediction:
                latest_thresh = os.path.join(save_dir, "thresholds_latest.npy")
                if os.path.exists(latest_thresh):
                    np.save(os.path.join(save_dir, "thresholds.npy"), np.load(latest_thresh))


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Branch Emotion Prediction Network on EMOTIC")
    parser.add_argument("--data_src", type=str, required=True, help="Directory containing preprocessed EMOTIC .npy arrays")
    parser.add_argument("--save_dir", type=str, default="./weights/model_swin_attn", help="Directory to save trained model_*.pth and thresholds.npy")
    parser.add_argument("--body_backbone", type=str, default="swin_v2_t", choices=["resnet18", "resnet50", "swin_v2_t"])
    parser.add_argument("--body_ckpt", type=str, default=None, help="Optional pretrained EMOTIC body checkpoint (.pth)")
    parser.add_argument("--context_arch", type=str, default="resnet50", choices=["resnet18", "resnet50"])
    parser.add_argument("--context_ckpt", type=str, default="./places/resnet50_state_dict.pth", help="Places365 state_dict path")
    parser.add_argument("--face_backbone", type=str, default="sfer", choices=["sfer", "bfer", "vgg_fer"])
    parser.add_argument("--face_ckpt", type=str, default=None, help="Pretrained FER checkpoint path (.pt / .pth)")
    parser.add_argument("--fusion_type", type=str, default="cross_self_attention", choices=["concat", "spatial_attention", "cross_self_attention"])
    parser.add_argument("--is_vad", action="store_true", default=False, help="Train for continuous VAD prediction instead of 26 discrete emotions")
    parser.add_argument("--discrete_weight_type", type=str, default="dynamic", choices=["mean", "static", "dynamic"])
    parser.add_argument("--cont_loss_type", type=str, default="sl1", choices=["sl1", "l2"])
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=26, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Adam weight decay")
    parser.add_argument("--step_size", type=int, default=7, help="StepLR decay period")
    parser.add_argument("--gamma", type=float, default=0.1, help="StepLR multiplicative factor")
    parser.add_argument("--unfreeze_backbones", action="store_true", default=False, help="Fine-tune all 3 backbones along with Fusion module")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Build 3 feature extraction branches
    model_body, num_body_features, is_swin = build_body_backbone(
        backbone_type=args.body_backbone,
        checkpoint_path=args.body_ckpt,
    )
    # Cắt bỏ lớp phân loại cuối của nhánh Body để lấy vector đặc trưng
    if is_swin and hasattr(model_body, "head"):
      model_body.head = torch.nn.Identity()
    elif hasattr(model_body, "fc"):
      model_body = torch.nn.Sequential(*(list(model_body.children())[:-1]))

    spatial_ctx = args.fusion_type == "spatial_attention"
    model_context, num_context_features = build_context_backbone(
        arch=args.context_arch,
        state_dict_path=args.context_ckpt,
    )
    # Cắt bỏ lớp phân loại cuối của nhánh Context
    if spatial_ctx:
      model_context = torch.nn.Sequential(
          *(list(model_context.children())[:-2])
      )
    else:
      model_context = torch.nn.Sequential(
          *(list(model_context.children())[:-1])
      )
    model_face, num_face_features, is_bfer = build_face_backbone(
        backbone_type=args.face_backbone,
        checkpoint_path=args.face_ckpt,
    )

    # 2. Build Fusion module
    shape_output_context = (args.batch_size, num_context_features, 7, 7) if spatial_ctx else None
    fusion_model = build_fusion_model(
        fusion_type=args.fusion_type,
        num_context_features=num_context_features,
        num_body_features=num_body_features,
        num_face_features=num_face_features,
        is_swin=is_swin,
        is_vad_prediction=args.is_vad,
        shape_output_context=shape_output_context,
    )

    # 3. Build DataLoaders
    (_, val_dataset, test_dataset), (train_loader, val_loader, test_loader) = build_dataloaders(
        args.data_src, batch_size=args.batch_size, is_swin=is_swin
    )

    # 4. Loss, Optimizer & Scheduler
    if not args.is_vad:
        criterion = DiscreteLoss(weight_type=args.discrete_weight_type, device=device)
    else:
        criterion = ContinuousLoss_SL1() if args.cont_loss_type == "sl1" else ContinuousLoss_L2()

    if args.unfreeze_backbones:
        params_to_optimize = (
            list(fusion_model.parameters())
            + list(model_context.parameters())
            + list(model_body.parameters())
            + list(model_face.parameters())
        )
    else:
        params_to_optimize = list(fusion_model.parameters())

    optimizer = optim.Adam(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # 5. Train
    train_emotic(
        epochs=args.epochs,
        models_list=[model_context, model_body, model_face, fusion_model],
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        num_val_images=len(val_dataset),
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=args.save_dir,
        is_vad_prediction=args.is_vad,
        is_bfer=is_bfer,
        unfreeze_backbones=args.unfreeze_backbones,
    )

    # 6. Final Evaluation on Test Split using Best Saved Checkpoints
    print("\nLoading best saved checkpoints for final test evaluation...")
    best_context = torch.load(os.path.join(args.save_dir, "model_context.pth"), map_location=device, weights_only=False)
    best_body = torch.load(os.path.join(args.save_dir, "model_body.pth"), map_location=device, weights_only=False)
    best_face = torch.load(os.path.join(args.save_dir, "model_face.pth"), map_location=device, weights_only=False)
    best_fusion = torch.load(os.path.join(args.save_dir, "model_fusion.pth"), map_location=device, weights_only=False)

    if not args.is_vad:
        test_map = test_disc(
            [best_context, best_body, best_face, best_fusion],
            device=device,
            data_loader=test_loader,
            num_images=len(test_dataset),
            is_bfer=is_bfer,
            thresholds_save_path=os.path.join(args.save_dir, "thresholds.npy"),
        )
        print(f"Final Test mAP = {test_map:.4f}")
    else:
        test_mae = evaluate_vad(
            [best_context, best_body, best_face, best_fusion],
            device=device,
            data_loader=test_loader,
            num_images=len(test_dataset),
            is_bfer=is_bfer,
        )
        print(f"Final Test VAD MAE = {test_mae:.4f}")


if __name__ == "__main__":
    main()