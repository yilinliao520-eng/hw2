# HW2: Deep Learning and Spatial Intelligence

This repository contains the code and selected results for HW2.

## Contents

- `task1/`: ResNet18 Flowers102 image classification with learning-rate, pretraining, and attention-module comparison.
- `task2/`: YOLOv8 vehicle detection, multi-object tracking, and line-crossing counting.
- `task3/`: U-Net semantic segmentation on the Stanford Background Dataset with loss comparison.

Large files such as trained model weights and videos are not stored in this GitHub repository. They should be uploaded to Google Drive or another cloud drive and linked in the PDF report.

## Work Division

| Task | Contributor | Main Work |
|---|---|---|
| Task 1 | Wenjie | Flowers102 classification with ResNet18, SE, and CBAM |
| Task 2 | Liaoyi Lin | YOLOv8 vehicle detection, tracking, and line-crossing counting |
| Task 3 | Wenjie and Liaoyi Lin | U-Net semantic segmentation and loss comparison |

## Environment

The experiments were run with Python, PyTorch, torchvision, Ultralytics YOLOv8, OpenCV, NumPy, Pillow, Matplotlib, and tqdm.

Task-specific dependencies are listed in:

- `task1/README.md`
- `task2/code/requirements.txt`
- `task3/code/requirements.txt`

## Task 1 Summary

Task 1 fine-tunes ResNet18 on the Oxford Flowers102 dataset and compares learning rates, ImageNet pretraining, training epochs, and attention modules.

Main results:

| ID | Model | Setting | Best Val Acc | Test Acc |
|---|---|---|---:|---:|
| H1 | ResNet18 | baseline, pretrained | 96.08% | 94.80% |
| H4 | ResNet18 | higher LR: 0.005 / 0.05 | 96.96% | 96.18% |
| H6 | ResNet18 | no pretraining | 45.20% | 43.92% |
| H7 | ResNet18 + SE | pretrained, higher LR | 96.96% | 96.57% |
| H8 | ResNet18 + CBAM | pretrained, higher LR | 97.06% | 96.67% |

The best result is achieved by ResNet18 + CBAM with `96.67%` test accuracy. See `task1/README.md` for the full result table.

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

https://drive.google.com/drive/u/0/folders/1HcGtyP6CGNqLppbp7kNhJWUJ10tHl-Ad

The PDF report should include both this GitHub repository link and the Google Drive link.
