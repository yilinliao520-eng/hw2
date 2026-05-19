from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from task3_unet.dataset import CLASS_NAMES, StanfordBackgroundDataset
from task3_unet.losses import build_loss
from task3_unet.metrics import SegmentationMeter
from task3_unet.model import UNet
from task3_unet.viz import save_prediction_grid


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    meter = SegmentationMeter(len(CLASS_NAMES))
    with torch.set_grad_enabled(training):
        for images, masks in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            meter.update(logits, masks)
    metrics = meter.compute()
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def write_metrics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["epoch", "train_loss", "train_miou", "train_pixel_acc", "val_loss", "val_miou", "val_pixel_acc"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def plot_curves(path: Path, rows: list[dict]) -> None:
    epochs = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(epochs, [r["train_loss"] for r in rows], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in rows], label="val")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, [r["train_miou"] for r in rows], label="train")
    axes[1].plot(epochs, [r["val_miou"] for r in rows], label="val")
    axes[1].set_title("mIoU")
    axes[2].plot(epochs, [r["train_pixel_acc"] for r in rows], label="train")
    axes[2].plot(epochs, [r["val_pixel_acc"] for r in rows], label="val")
    axes[2].set_title("Pixel Accuracy")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
        ax.set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/stanford_background")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--loss", choices=["ce", "dice", "ce_dice"], required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    run_name = args.name or f"unet_{args.loss}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    train_set = StanfordBackgroundDataset(
        args.data_root, Path(args.splits_dir) / "train.txt", image_size=args.image_size, augment=True
    )
    val_set = StanfordBackgroundDataset(
        args.data_root, Path(args.splits_dir) / "val.txt", image_size=args.image_size, augment=False
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = UNet(num_classes=len(CLASS_NAMES), base_channels=args.base_channels).to(device)
    criterion = build_loss(args.loss, len(CLASS_NAMES))
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_miou = -1.0
    rows = []
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, train_loader, criterion, device, optimizer)
        val = run_epoch(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train["loss"],
            "train_miou": train["miou"],
            "train_pixel_acc": train["pixel_acc"],
            "val_loss": val["loss"],
            "val_miou": val["miou"],
            "val_pixel_acc": val["pixel_acc"],
        }
        rows.append(row)
        write_metrics(out_dir / "metrics.csv", rows)
        plot_curves(out_dir / "curves.png", rows)
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": row, "class_names": CLASS_NAMES}, out_dir / "last.pt")
        if val["miou"] > best_miou:
            best_miou = val["miou"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": row, "class_names": CLASS_NAMES}, out_dir / "best.pt")
            batch = next(iter(val_loader))
            with torch.no_grad():
                logits = model(batch[0].to(device))
            save_prediction_grid(batch[0], batch[1], logits.cpu(), out_dir / "predictions.png")
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"train_miou={row['train_miou']:.4f} val_miou={row['val_miou']:.4f}"
        )

    summary = {"best_val_miou": best_miou, "best_epoch": max(rows, key=lambda r: r["val_miou"])["epoch"], "final": rows[-1]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
