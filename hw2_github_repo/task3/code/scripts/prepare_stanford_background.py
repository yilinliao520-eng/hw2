from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

from task3_unet.dataset import build_splits, find_dataset_root


URL = "http://dags.stanford.edu/data/iccv09Data.tar.gz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/stanford_background")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--url", default=URL)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "iccv09Data.tar.gz"
    if not archive.exists():
        print(f"Downloading {args.url} -> {archive}")
        urllib.request.urlretrieve(args.url, archive)
    try:
        root = find_dataset_root(data_dir)
        print(f"Dataset already extracted at {root}")
    except FileNotFoundError:
        print(f"Extracting {archive}")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(data_dir)
        root = find_dataset_root(data_dir)
    build_splits(root, args.splits_dir, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Dataset root: {root}")
    print(f"Splits written to: {args.splits_dir}")


if __name__ == "__main__":
    main()
