from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


def side_of_line(point: tuple[float, float], line: tuple[float, float, float, float]) -> float:
    x, y = point
    x1, y1, x2, y2 = line
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLOv8 tracking and export bboxes, IDs, and line counts.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", default="runs/track/video")
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--line",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Counting line in pixel coordinates. Default: vertical midline of the video.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Could not read video metadata from {args.source}")

    line = tuple(args.line) if args.line else (width / 2, 0.0, width / 2, float(height))
    writer = cv2.VideoWriter(
        str(out_dir / "annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    model = YOLO(args.weights)
    rows: list[dict] = []
    crossings: list[dict] = []
    last_side: dict[int, float] = {}
    class_counts: Counter[str] = Counter()
    active_ids: set[int] = set()

    results = model.track(
        source=args.source,
        stream=True,
        persist=True,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )

    for frame_idx, result in enumerate(results):
        annotated = result.plot()
        cv2.line(
            annotated,
            (int(line[0]), int(line[1])),
            (int(line[2]), int(line[3])),
            (0, 255, 255),
            2,
        )
        writer.write(annotated)

        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        names = result.names

        for bbox, track_id, class_id, score in zip(xyxy, ids, cls, conf):
            x1, y1, x2, y2 = [float(v) for v in bbox]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            class_name = names.get(int(class_id), str(class_id))
            class_counts[class_name] += 1
            active_ids.add(int(track_id))

            current_side = side_of_line((cx, cy), line)
            previous_side = last_side.get(int(track_id))
            if previous_side is not None and previous_side * current_side < 0:
                crossings.append(
                    {
                        "frame": frame_idx,
                        "track_id": int(track_id),
                        "class": class_name,
                        "from_side": "positive" if previous_side > 0 else "negative",
                        "to_side": "positive" if current_side > 0 else "negative",
                    }
                )
            if current_side != 0:
                last_side[int(track_id)] = current_side

            rows.append(
                {
                    "frame": frame_idx,
                    "track_id": int(track_id),
                    "class_id": int(class_id),
                    "class": class_name,
                    "confidence": float(score),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": cx,
                    "cy": cy,
                }
            )

    writer.release()

    detections_path = out_dir / "detections.csv"
    with detections_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["frame", "track_id", "class_id", "class", "confidence", "x1", "y1", "x2", "y2", "cx", "cy"]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    crossings_path = out_dir / "crossings.csv"
    with crossings_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["frame", "track_id", "class", "from_side", "to_side"]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(crossings)

    summary = {
        "source": args.source,
        "weights": args.weights,
        "frames_with_detections": len({r["frame"] for r in rows}),
        "detections": len(rows),
        "unique_track_ids": len(active_ids),
        "line": line,
        "line_crossings": len(crossings),
        "class_observations": dict(class_counts),
        "outputs": {
            "annotated_video": str((out_dir / "annotated.mp4").resolve()),
            "detections_csv": str(detections_path.resolve()),
            "crossings_csv": str(crossings_path.resolve()),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
