"""EMOTIC preprocessed dataset and DataLoader builder."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from configs.config import get_norms


def get_transforms(is_swin: bool = True):
    """Return (train_transform, test_transform, face_train_transform, face_test_transform)."""
    face_train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
    ])

    face_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    if is_swin:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=[232], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=[224]),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=[232], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=[224]),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

    return train_transform, test_transform, face_train_transform, face_test_transform


class Emotic_PreDataset(Dataset):
    """Custom EMOTIC dataset class using preprocessed data stored in .npy files."""

    def __init__(
        self,
        x_context,
        x_body,
        x_face,
        y_cat,
        y_cont,
        context_transform,
        body_transform,
        face_transform,
        context_norm,
        body_norm,
        face_norm,
    ):
        super(Emotic_PreDataset, self).__init__()
        self.x_context = x_context
        self.x_body = x_body
        self.x_face = x_face
        self.y_cat = y_cat
        self.y_cont = y_cont
        self.context_transform = context_transform
        self.body_transform = body_transform
        self.face_transform = face_transform
        self.context_norm = transforms.Normalize(context_norm[0], context_norm[1])
        self.body_norm = transforms.Normalize(body_norm[0], body_norm[1])
        self.face_norm = transforms.Normalize(face_norm[0], face_norm[1])

    def __len__(self):
        return len(self.y_cont)

    def __getitem__(self, index):
        image_context = self.x_context[index]
        image_body = self.x_body[index]
        image_face = self.x_face[index]
        cat_label = self.y_cat[index]
        cont_label = self.y_cont[index]

        return (
            self.context_norm(self.context_transform(image_context)),
            self.body_norm(self.body_transform(image_body)),
            self.face_norm(self.face_transform(image_face)),
            torch.tensor(cat_label, dtype=torch.float32),
            torch.tensor(cont_label, dtype=torch.float32) / 10.0,
        )


def load_split_arrays(data_src: str, split: str):
    """Load context, body, face, cat, and cont numpy arrays for a given split ('train', 'val', 'test')."""
    context_arr = np.load(os.path.join(data_src, f"{split}_context_arr.npy"))
    body_arr = np.load(os.path.join(data_src, f"{split}_body_arr.npy"))
    cat_arr = np.load(os.path.join(data_src, f"{split}_cat_arr.npy"))
    cont_arr = np.load(os.path.join(data_src, f"{split}_cont_arr.npy"))
    face_raw = np.load(os.path.join(data_src, f"{split}_face_arr.npy"))
    face_arr = np.stack((face_raw,) * 3, axis=-1) if face_raw.ndim == 3 else face_raw
    return context_arr, body_arr, face_arr, cat_arr, cont_arr


def build_dataloaders(data_src: str, batch_size: int = 26, is_swin: bool = True):
    """Build train, val, and test datasets and DataLoaders from preprocessed .npy files."""
    context_norm, body_norm, face_norm = get_norms(is_swin=is_swin)
    train_tf, test_tf, face_train_tf, face_test_tf = get_transforms(is_swin=is_swin)

    train_ctx, train_body, train_face, train_cat, train_cont = load_split_arrays(data_src, "train")
    val_ctx, val_body, val_face, val_cat, val_cont = load_split_arrays(data_src, "val")
    test_ctx, test_body, test_face, test_cat, test_cont = load_split_arrays(data_src, "test")

    train_dataset = Emotic_PreDataset(
        train_ctx, train_body, train_face, train_cat, train_cont,
        train_tf, train_tf, face_train_tf, context_norm, body_norm, face_norm,
    )
    val_dataset = Emotic_PreDataset(
        val_ctx, val_body, val_face, val_cat, val_cont,
        train_tf, train_tf, face_train_tf, context_norm, body_norm, face_norm,
    )
    test_dataset = Emotic_PreDataset(
        test_ctx, test_body, test_face, test_cat, test_cont,
        test_tf, test_tf, face_test_tf, context_norm, body_norm, face_norm,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (train_dataset, val_dataset, test_dataset), (train_loader, val_loader, test_loader)
