from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--names", nargs="+", default=["unet_ce", "unet_dice", "unet_ce_dice"])
    parser.add_argument("--out", default="runs/comparison.md")
    args = parser.parse_args()

    rows = []
    for name in args.names:
        metrics_path = Path(args.runs_dir) / name / "metrics.csv"
        with metrics_path.open(newline="", encoding="utf-8") as f:
            metrics = [{k: float(v) if k != "epoch" else int(v) for k, v in r.items()} for r in csv.DictReader(f)]
        best = max(metrics, key=lambda r: r["val_miou"])
        summary_path = Path(args.runs_dir) / name / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        rows.append(
            {
                "run": name,
                "best_epoch": int(best["epoch"]),
                "best_val_miou": best["val_miou"],
                "best_val_pixel_acc": best["val_pixel_acc"],
                "final_val_miou": metrics[-1]["val_miou"],
                "summary": summary,
            }
        )

    lines = ["# Loss Comparison", "", "| Run | Best Epoch | Best Val mIoU | Best Val Pixel Acc | Final Val mIoU |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['best_epoch']} | {r['best_val_miou']:.4f} | "
            f"{r['best_val_pixel_acc']:.4f} | {r['final_val_miou']:.4f} |"
        )
    best_run = max(rows, key=lambda r: r["best_val_miou"])
    lines += ["", f"Best run by validation mIoU: **{best_run['run']}**."]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
