import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Flowers102


def build_transforms(image_size=224):
    """
    数据集预处理。
    """

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_transform, eval_transform


def build_datasets(data_root="./data", image_size=224):
    """
    划分数据集
    """
    train_transform, eval_transform = build_transforms(image_size)

    train_dataset = Flowers102(root=data_root,split="test",transform=train_transform,download=False)
    val_dataset = Flowers102(root=data_root,split="val",transform=eval_transform,download=False)
    test_dataset = Flowers102(root=data_root,split="train",transform=eval_transform,download=False)

    return train_dataset, val_dataset, test_dataset


def build_dataloaders(data_root="./data",image_size=224,batch_size=32,num_workers=4):
    """
    加载数据集
    """
    train_dataset, val_dataset, test_dataset = build_datasets(data_root=data_root,image_size=image_size)

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=use_cuda)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=use_cuda)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=use_cuda)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_dataloaders(data_root="./data",image_size=224,batch_size=32,num_workers=4)

    print("Flowers102 dataloaders are ready.")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")