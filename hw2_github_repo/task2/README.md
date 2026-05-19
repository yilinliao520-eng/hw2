# Task 2: YOLOv8 Vehicle Detection and Tracking

## Goal

Train a modern one-stage detector on the Road Vehicle Images Dataset, then apply the trained model to a 10-30 second road video for object detection, multi-object tracking, and line-crossing counting.

## Method

- Detector: YOLOv8s
- Training strategy: supervised fine-tuning from pretrained YOLOv8s weights
- Dataset: Road Vehicle Images Dataset
- Tracking: YOLOv8 tracking module
- Counting: virtual vertical line crossing based on Tracking IDs

## Repository Structure

```text
task2/
  code/
    scripts/
      prepare_dataset.py
      train_yolov8.py
      track_video.py
      analyze_tracks.py
  config/
    road_vehicle.yaml
  results/
    training/
    tracking/
```

## Training

```bash
cd task2/code
pip install -r requirements.txt

python scripts/train_yolov8.py \
  --data ../config/road_vehicle.yaml \
  --model weights/yolov8s.pt \
  --epochs 50 \
  --batch 16 \
  --imgsz 640 \
  --workers 0 \
  --device 0 \
  --name road_vehicle_yolov8s_b16
```

## Video Tracking

```bash
python scripts/track_video.py \
  --weights path/to/yolov8s_road_vehicle_best.pt \
  --source path/to/input_video_30s.mp4 \
  --out runs/track/user_road_video_new_yolov8s_b16 \
  --device 0 \
  --imgsz 640 \
  --conf 0.25
```

## Outputs

- `results/training/results.csv`: training metrics.
- `results/training/results.png`: training curves.
- `results/training/BoxPR_curve.png`: precision-recall curve.
- `results/training/confusion_matrix.png`: class-level confusion matrix.
- `results/tracking/detections.csv`: frame-level bbox, class, confidence, and Tracking ID.
- `results/tracking/crossings.csv`: line-crossing records.
- `results/tracking/summary.json`: tracking summary.
- `results/tracking/track_analysis.md`: ID and trajectory analysis.

## Main Tracking Result

- Frames: `900`
- Detections: `2857`
- Unique Tracking IDs: `78`
- Virtual line: vertical center line, `x = 640`
- Line crossings: `5`

Large files, including the trained YOLOv8s weight and annotated tracking video, are stored in Google Drive.
