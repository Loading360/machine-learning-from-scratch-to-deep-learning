"""U-Net baseline for Kaggle Ultrasound Nerve Segmentation.

Download the labeled training images separately. Do not commit the dataset.
Run: python 04_deep_learning/unet.py --data-dir /path/to/train --epochs 3
"""
from pathlib import Path
import argparse
import copy
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit


class UltrasoundDataset(Dataset):
    def __init__(self, images, masks, image_size=256):
        assert len(images) == len(masks)
        self.images = images
        self.masks = masks
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        with Image.open(self.images[idx]) as im:
            image = im.convert("L").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            x = np.asarray(image, dtype=np.float32).copy() / 255.0
        with Image.open(self.masks[idx]) as im:
            mask = im.convert("L").resize(
                (self.image_size, self.image_size), Image.Resampling.NEAREST
            )
            y = (np.asarray(mask) > 0).astype(np.float32)
        return torch.from_numpy(x).unsqueeze(0), torch.from_numpy(y).unsqueeze(0)


def patient_id(path):
    return path.stem.split("_")[0]


def split_by_patient(images, masks, seed=42):
    groups = [patient_id(p) for p in images]
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    dev_idx, test_idx = next(split.split(images, groups=groups))
    dev_images = [images[i] for i in dev_idx]
    dev_masks = [masks[i] for i in dev_idx]
    test_images = [images[i] for i in test_idx]
    test_masks = [masks[i] for i in test_idx]

    dev_groups = [patient_id(p) for p in dev_images]
    split_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + 1)
    train_idx, val_idx = next(split_val.split(dev_images, groups=dev_groups))
    train = ([dev_images[i] for i in train_idx], [dev_masks[i] for i in train_idx])
    val = ([dev_images[i] for i in val_idx], [dev_masks[i] for i in val_idx])
    test = (test_images, test_masks)
    ids = [{patient_id(p) for p in subset[0]} for subset in (train, val, test)]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    return train, val, test


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(1, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.bottleneck = DoubleConv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.output = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat((self.up3(b), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        return self.output(d1)


def dice_per_image(logits, targets, eps=1e-7):
    preds = (torch.sigmoid(logits) > 0.5).float()
    dims = (1, 2, 3)
    intersection = (preds * targets).sum(dim=dims)
    denominator = preds.sum(dim=dims) + targets.sum(dim=dims)
    return (2 * intersection + eps) / (denominator + eps)


def combined_loss(logits, targets):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    soft_dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    return bce + (1 - soft_dice.mean())


def balanced_sampler(mask_paths):
    flags = []
    for path in mask_paths:
        with Image.open(path) as im:
            flags.append(bool(np.any(np.asarray(im) > 0)))
    flags = np.asarray(flags, dtype=bool)
    n_pos = int(flags.sum())
    n_neg = len(flags) - n_pos
    if not n_pos or not n_neg:
        raise ValueError("Training set needs both positive and empty masks")
    print("Training masks:", n_pos, "positive,", n_neg, "empty")
    weights = np.where(flags, 1 / n_pos, 1 / n_neg)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(flags),
        replacement=True,
    )


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_n = 0.0, 0
    all_dice, positive_dice = [], []
    positive_n = detected_n = empty_n = correctly_empty_n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.sigmoid(logits) > 0.5
            real_pos = y.sum(dim=(1, 2, 3)) > 0
            pred_pos = preds.sum(dim=(1, 2, 3)) > 0
            dice = dice_per_image(logits, y)
            all_dice.extend(dice.cpu().tolist())
            positive_dice.extend(dice[real_pos].cpu().tolist())
            positive_n += real_pos.sum().item()
            detected_n += (real_pos & pred_pos).sum().item()
            empty_n += (~real_pos).sum().item()
            correctly_empty_n += ((~real_pos) & (~pred_pos)).sum().item()
            total_loss += loss_fn(logits, y).item() * len(x)
            total_n += len(x)
    return {
        "loss": total_loss / total_n,
        "dice_all": float(np.mean(all_dice)),
        "dice_positive": float(np.mean(positive_dice)) if positive_dice else float("nan"),
        "positive_detection": detected_n / positive_n if positive_n else float("nan"),
        "empty_correct": correctly_empty_n / empty_n if empty_n else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing image.tif and image_mask.tif pairs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("unet_outputs"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    images = sorted(p for p in args.data_dir.glob("*.tif")
                    if not p.stem.endswith("_mask"))
    if not images:
        raise FileNotFoundError(f"No .tif ultrasound images in {args.data_dir}")
    masks = [p.with_name(p.stem + "_mask.tif") for p in images]
    missing = [p for p in masks if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing masks, e.g. {missing[0]}")

    train, val, test = split_by_patient(images, masks, args.seed)
    print("Image counts (train/val/test):", *(len(s[0]) for s in (train, val, test)))
    print("Patient counts (train/val/test):",
          *(len({patient_id(p) for p in s[0]}) for s in (train, val, test)))

    train_ds, val_ds, test_ds = [
        UltrasoundDataset(*subset, image_size=args.image_size)
        for subset in (train, val, test)
    ]
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=balanced_sampler(train[1]),
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0)

    model = UNet().to(device)
    loss_fn = combined_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_positive_dice": []}
    best_loss, best_weights = float("inf"), None

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(x)
        train_loss = train_loss_sum / len(train_ds)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        val_loss = val_metrics["loss"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_metrics["dice_all"])
        history["val_positive_dice"].append(val_metrics["dice_positive"])
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch+1}/{args.epochs} | train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} | val Dice {val_metrics['dice_all']:.4f} | "
              f"positive Dice {val_metrics['dice_positive']:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), args.output_dir / "unet_best.pth")
    test_metrics = evaluate(model, test_loader, loss_fn, device)
    print("Held-out test metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    fig, ax = plt.subplots()
    ax.plot(history["train_loss"], label="Train BCE + Dice")
    ax.plot(history["val_loss"], label="Validation BCE + Dice")
    ax.set(xlabel="Epoch", ylabel="BCE + Dice loss")
    ax.legend()
    fig.savefig(args.output_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(history["val_dice"], label="Validation Dice (all)")
    ax.plot(history["val_positive_dice"], label="Validation Dice (positive)")
    ax.set(xlabel="Epoch", ylabel="Dice")
    ax.legend()
    fig.savefig(args.output_dir / "dice_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    positive_indices = []
    for i in range(len(test_ds)):
        _, mask = test_ds[i]
        if mask.sum() > 0:
            positive_indices.append(i)
        if len(positive_indices) == 3:
            break
    n_show = len(positive_indices)
    fig, axes = plt.subplots(n_show, 3, figsize=(10, 3 * n_show), squeeze=False)
    model.eval()
    with torch.no_grad():
        for i in range(n_show):
            x, y = test_ds[positive_indices[i]]
            logits = model(x.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy()
            for ax, arr, title in zip(
                axes[i], (x[0].numpy(), y[0].numpy(), pred),
                ("Ultrasound", "Ground truth", "Prediction")
            ):
                ax.imshow(arr, cmap="gray")
                ax.set_title(title)
                ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.output_dir / "positive_predictions.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
