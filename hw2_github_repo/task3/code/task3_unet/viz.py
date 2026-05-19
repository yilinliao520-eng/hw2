from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


PALETTE = np.array(
    [
        [128, 192, 255],
        [34, 139, 34],
        [128, 128, 128],
        [124, 252, 0],
        [65, 105, 225],
        [178, 34, 34],
        [160, 82, 45],
        [255, 215, 0],
    ],
    dtype=np.uint8,
)


def colorize(mask: np.ndarray) -> np.ndarray:
    clipped = np.clip(mask.astype(np.int64), 0, len(PALETTE) - 1)
    return PALETTE[clipped]


def denormalize(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (image.cpu() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def save_prediction_grid(images: torch.Tensor, masks: torch.Tensor, logits: torch.Tensor, out_path: str | Path) -> None:
    preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
    masks_np = masks.detach().cpu().numpy()
    n = min(4, images.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    for i in range(n):
        axes[i, 0].imshow(denormalize(images[i]))
        axes[i, 0].set_title("Image")
        axes[i, 1].imshow(colorize(masks_np[i]))
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(colorize(preds[i]))
        axes[i, 2].set_title("Prediction")
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
