from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = [
    "sky",
    "tree",
    "road",
    "grass",
    "water",
    "building",
    "mountain",
    "foreground",
]


def find_dataset_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    for candidate in [root, root / "iccv09Data", root / "stanford_background"]:
        if (candidate / "images").exists() and (candidate / "labels").exists():
            return candidate
    raise FileNotFoundError(f"Cannot find Stanford Background images/labels under {root}")


def build_splits(data_root: str | Path, out_dir: str | Path, val_ratio: float = 0.2, seed: int = 42) -> None:
    root = find_dataset_root(data_root)
    names = sorted(p.stem for p in (root / "images").glob("*.jpg"))
    if not names:
        names = sorted(p.stem for p in (root / "images").glob("*.png"))
    if not names:
        raise FileNotFoundError(f"No images found in {root / 'images'}")
    rng = random.Random(seed)
    rng.shuffle(names)
    val_count = max(1, int(round(len(names) * val_ratio)))
    splits = {"train": names[val_count:], "val": names[:val_count]}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, split_names in splits.items():
        (out / f"{split}.txt").write_text("\n".join(split_names) + "\n", encoding="utf-8")
    meta = {"dataset_root": str(root), "num_images": len(names), "val_ratio": val_ratio, "seed": seed}
    (out / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


class StanfordBackgroundDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split_file: str | Path,
        image_size: int = 256,
        augment: bool = False,
    ) -> None:
        self.root = find_dataset_root(data_root)
        self.names = [x.strip() for x in Path(split_file).read_text(encoding="utf-8").splitlines() if x.strip()]
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.names)

    def _image_path(self, name: str) -> Path:
        for ext in [".jpg", ".png", ".jpeg"]:
            p = self.root / "images" / f"{name}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(name)

    def _mask_path(self, name: str) -> Path:
        for suffix in [".regions.txt", ".txt", ".png"]:
            p = self.root / "labels" / f"{name}{suffix}"
            if p.exists():
                return p
        raise FileNotFoundError(name)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.names[idx]
        image = Image.open(self._image_path(name)).convert("RGB")
        mask_path = self._mask_path(name)
        if mask_path.suffix == ".png":
            mask = Image.open(mask_path)
        else:
            arr = np.loadtxt(mask_path, dtype=np.int64)
            if arr.min() >= 1:
                arr = arr - 1
            arr = np.where(arr < 0, 255, arr)
            mask = Image.fromarray(arr.astype(np.uint8), mode="L")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        image_arr = (image_arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        mask_arr = np.asarray(mask, dtype=np.int64)
        mask_arr = np.where(mask_arr == 255, -1, mask_arr)
        return torch.from_numpy(image_arr).permute(2, 0, 1), torch.from_numpy(mask_arr)
