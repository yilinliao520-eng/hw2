from __future__ import annotations

import numpy as np
import torch


class SegmentationMeter:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy().reshape(-1)
        gt = target.detach().cpu().numpy().reshape(-1)
        mask = (gt >= 0) & (gt < self.num_classes)
        hist = np.bincount(
            self.num_classes * gt[mask].astype(np.int64) + pred[mask].astype(np.int64),
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)
        self.confusion += hist

    def compute(self) -> dict[str, float | list[float]]:
        hist = self.confusion.astype(np.float64)
        diag = np.diag(hist)
        union = hist.sum(axis=1) + hist.sum(axis=0) - diag
        iou = np.divide(diag, union, out=np.full_like(diag, np.nan), where=union > 0)
        acc = diag.sum() / max(hist.sum(), 1.0)
        return {
            "pixel_acc": float(acc),
            "miou": float(np.nanmean(iou)),
            "class_iou": [float(x) if not np.isnan(x) else float("nan") for x in iou],
        }
