# HW2: Deep Learning and Spatial Intelligence

This repository contains the code and selected results for HW2.

## Contents

- `task2/`: YOLOv8 vehicle detection, multi-object tracking, and line-crossing counting.
- `task3/`: U-Net semantic segmentation on the Stanford Background Dataset with loss comparison.

Large files such as trained model weights and videos are not stored in this GitHub repository. They should be uploaded to Google Drive or another cloud drive and linked in the PDF report.

## Environment

The experiments were run with Python, PyTorch, Ultralytics YOLOv8, OpenCV, NumPy, Pillow, Matplotlib, and tqdm.

Task-specific dependencies are listed in:

- `task2/code/requirements.txt`
- `task3/code/requirements.txt`

## Task 2 Summary

Task 2 fine-tunes YOLOv8s on the Road Vehicle Images Dataset and applies the trained detector to a 30-second road video. YOLOv8 tracking is used to assign Tracking IDs, and a vertical virtual line is used for line-crossing counting.

Main results:

- Model: YOLOv8s fine-tuned from pretrained weights
- Dataset: Road Vehicle Images Dataset
- Best validation mAP50: about `0.495`
- Video frames: `900`
- Total detections: `2857`
- Unique Tracking IDs: `78`
- Line crossings: `5`

See `task2/README.md` for details.

## Task 3 Summary

Task 3 implements U-Net from basic PyTorch layers and trains it on the Stanford Background Dataset. Three loss functions are compared: Cross-Entropy, Dice, and Cross-Entropy + Dice.

Main results:

| Loss | Best Epoch | Best Val mIoU | Best Val Pixel Acc | Final Val mIoU |
|---|---:|---:|---:|---:|
| Cross-Entropy | 46 | 0.5188 | 0.7557 | 0.5073 |
| Dice | 46 | 0.5190 | 0.7522 | 0.5056 |
| Cross-Entropy + Dice | 38 | 0.5224 | 0.7552 | 0.5021 |

The best validation mIoU is achieved by Cross-Entropy + Dice.

See `task3/README.md` for details.

## Model Weights and Large Outputs

The trained model weights and large video outputs are provided via cloud storage.

Google Drive link:

`TODO: paste Google Drive sharing link here`

The PDF report should include both this GitHub repository link and the Google Drive link.
