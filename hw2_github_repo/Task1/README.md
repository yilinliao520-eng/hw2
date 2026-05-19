# HW2 Task 1: Flowers102 Classification

This task fine-tunes ResNet18 on the Oxford Flowers102 dataset and compares different training settings, including learning rate, number of epochs, ImageNet pretraining, and attention modules.

## Method

The dataset is loaded with `torchvision.datasets.Flowers102`. Images are resized to 224 x 224 and normalized with ImageNet statistics. Training uses random resized crop and horizontal flip as data augmentation.

Three models are tested:

- `resnet18`: standard ResNet18 with a new 102-class classifier.
- `resnet18_se`: ResNet18 with SE attention blocks.
- `resnet18_cbam`: ResNet18 with CBAM attention blocks.

The optimizer is SGD with momentum 0.9 and weight decay 1e-4. A cosine annealing learning rate scheduler is used. For pretrained models, the backbone uses a smaller learning rate than the final fully connected layer.

## Results

| ID | Model | Setting | Best Val Acc | Test Acc |
|---|---|---|---:|---:|
| H1 | ResNet18 | baseline, pretrained | 96.08% | 94.80% |
| H2 | ResNet18 | lower LR: 0.0005 / 0.005 | 94.71% | 93.92% |
| H3 | ResNet18 | very low LR: 0.0001 / 0.001 | 76.57% | 75.00% |
| H4 | ResNet18 | higher LR: 0.005 / 0.05 | 96.96% | 96.18% |
| H5 | ResNet18 | 40 epochs | 97.06% | 95.88% |
| H6 | ResNet18 | no pretraining | 45.20% | 43.92% |
| H7 | ResNet18 + SE | pretrained, higher LR | 96.96% | 96.57% |
| H8 | ResNet18 + CBAM | pretrained, higher LR | 97.06% | 96.67% |
