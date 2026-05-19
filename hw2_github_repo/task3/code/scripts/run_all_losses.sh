#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.

python scripts/train_unet.py \
  --data-root data/stanford_background \
  --splits-dir data/splits \
  --loss ce \
  --epochs 50 \
  --batch-size 8 \
  --image-size 256 \
  --base-channels 32 \
  --workers 0 \
  --device cuda:0 \
  --name unet_ce

python scripts/train_unet.py \
  --data-root data/stanford_background \
  --splits-dir data/splits \
  --loss dice \
  --epochs 50 \
  --batch-size 8 \
  --image-size 256 \
  --base-channels 32 \
  --workers 0 \
  --device cuda:0 \
  --name unet_dice

python scripts/train_unet.py \
  --data-root data/stanford_background \
  --splits-dir data/splits \
  --loss ce_dice \
  --epochs 50 \
  --batch-size 8 \
  --image-size 256 \
  --base-channels 32 \
  --workers 0 \
  --device cuda:0 \
  --name unet_ce_dice

python scripts/compare_runs.py --runs-dir runs
