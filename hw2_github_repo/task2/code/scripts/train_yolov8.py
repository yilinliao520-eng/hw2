from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Road Vehicle Images Dataset.")
    parser.add_argument("--data", default="data/road_vehicle.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="road_vehicle_yolov8n")
    args = parser.parse_args()

    project_dir = Path("runs/detect").resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        amp=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
