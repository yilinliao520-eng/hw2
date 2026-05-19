from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_first(root: Path, name: str) -> Path | None:
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None


def count_images(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def as_rel(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def load_source_config(raw: Path) -> dict | None:
    yaml_path = find_first(raw, "data.yaml")
    if yaml_path is None:
        yaml_files = sorted(raw.rglob("*.yaml"))
        yaml_path = yaml_files[0] if yaml_files else None
    if yaml_path is None:
        return None
    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_yaml_path"] = str(yaml_path)
    return cfg


def infer_split(raw: Path, names: list[str]) -> dict:
    candidates = {
        "train": ["train/images", "images/train"],
        "val": ["valid/images", "val/images", "images/valid", "images/val"],
        "test": ["test/images", "images/test"],
    }
    result: dict[str, str | list[str] | int] = {"names": names, "nc": len(names)}
    for split, rels in candidates.items():
        for rel in rels:
            path = raw / rel
            if path.exists():
                result[split] = as_rel(path, raw)
                break
    if "train" not in result or "val" not in result:
        raise FileNotFoundError(
            "Could not infer YOLO train/val image folders. Expected data.yaml or "
            "train/images plus valid/images/val/images under the raw dataset root."
        )
    result["path"] = str(raw.resolve())
    return result


def normalize_from_config(raw: Path, cfg: dict) -> dict:
    names = cfg.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if not names:
        names = [f"class_{i}" for i in range(int(cfg.get("nc", 0) or 0))]
    if not names:
        raise ValueError("The source data.yaml has no names/nc class definition.")

    yaml_dir = Path(cfg["_yaml_path"]).parent
    source_path = Path(cfg.get("path", "."))
    base = source_path if source_path.is_absolute() else (yaml_dir / source_path)
    base = base.resolve()

    out: dict[str, str | list[str] | int] = {"path": str(base), "names": names, "nc": len(names)}
    for split in ["train", "val", "test"]:
        src = cfg.get(split)
        if src is None:
            continue
        if isinstance(src, list):
            out[split] = [str((base / item).resolve()) if not Path(item).is_absolute() else item for item in src]
        else:
            src_path = Path(src)
            out[split] = str(src_path if src_path.is_absolute() else src)
    if "train" not in out or "val" not in out:
        return infer_split(raw, names)
    return out


def split_count(cfg: dict, split: str) -> int:
    base = Path(str(cfg["path"]))
    split_value = cfg.get(split)
    if split_value is None:
        return 0
    paths = split_value if isinstance(split_value, list) else [split_value]
    total = 0
    for item in paths:
        p = Path(item)
        if not p.is_absolute():
            p = base / p
        if p.exists():
            total += count_images(p)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Road Vehicle YOLO data config.")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/road_vehicle.yaml")
    args = parser.parse_args()

    raw = Path(args.raw).resolve()
    if not raw.exists():
        raise FileNotFoundError(f"Raw dataset path does not exist: {raw}")

    source = load_source_config(raw)
    if source is not None:
        cfg = normalize_from_config(raw, source)
        if split_count(cfg, "train") == 0 or split_count(cfg, "val") == 0:
            names = cfg["names"]
            cfg = infer_split(Path(source["_yaml_path"]).parent, names)
    else:
        cfg = infer_split(raw, [f"vehicle_{i}" for i in range(21)])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    summary = {
        "data_yaml": str(out.resolve()),
        "classes": cfg["names"],
        "train_images": split_count(cfg, "train"),
        "val_images": split_count(cfg, "val"),
        "test_images": split_count(cfg, "test"),
    }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
