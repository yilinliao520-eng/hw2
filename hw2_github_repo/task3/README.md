# Task 3: U-Net Semantic Segmentation

## Goal

Implement U-Net from basic PyTorch layers and train it on the Stanford Background Dataset. Compare Cross-Entropy Loss, Dice Loss, and Cross-Entropy + Dice Loss using validation mIoU.

## Dataset

- Dataset: Stanford Background Dataset
- Total images: `715`
- Training set: `572`
- Validation set: `143`
- Split ratio: 80/20
- Split files: `data_splits/train.txt`, `data_splits/val.txt`

The original dataset is not included in this repository. Use `iccv09Data.tar.gz` from the Stanford Background Dataset.

## Model

The U-Net implementation is in `code/task3_unet/model.py` and includes:

- encoder/downsampling path
- decoder/upsampling path
- skip connections
- final `1x1` convolution classifier

## Training

```bash
cd task3/code
pip install -r requirements.txt

PYTHONPATH=. python scripts/prepare_stanford_background.py \
  --data-dir ../data/stanford_background \
  --splits-dir ../data_splits

PYTHONPATH=. python scripts/train_unet.py \
  --data-root ../data/stanford_background \
  --splits-dir ../data_splits \
  --loss ce \
  --epochs 50 \
  --batch-size 8 \
  --lr 3e-4 \
  --image-size 256 \
  --base-channels 32 \
  --workers 0 \
  --device cuda:0 \
  --name unet_ce
```

To run all three losses:

```bash
PYTHONPATH=. bash scripts/run_all_losses.sh
```

## Results

| Loss | Best Epoch | Best Val mIoU | Best Val Pixel Acc | Final Val mIoU |
|---|---:|---:|---:|---:|
| Cross-Entropy | 46 | 0.5188 | 0.7557 | 0.5073 |
| Dice | 46 | 0.5190 | 0.7522 | 0.5056 |
| Cross-Entropy + Dice | 38 | 0.5224 | 0.7552 | 0.5021 |

The best validation mIoU is achieved by Cross-Entropy + Dice.

## Outputs

- `results/loss_comparison.md`: final comparison table.
- `results/*/metrics.csv`: epoch-level loss, mIoU, and pixel accuracy.
- `results/*/curves.png`: loss, mIoU, and pixel accuracy curves.
- `results/*/predictions.png`: image / ground truth / prediction visualization.
- `results/*/summary.json`: best epoch and final metrics.

Large files, including trained U-Net weights, are stored in Google Drive.
