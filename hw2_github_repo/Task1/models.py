import torch
import torch.nn as nn
from torchvision import models


# SE Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden_channels = max(channels // reduction, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        weight = self.pool(x).view(batch_size, channels)
        weight = self.fc(weight).view(batch_size, channels, 1, 1)

        return x * weight


# CBAM Block
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden_channels = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))

        weight = self.sigmoid(avg_out + max_out)
        return x * weight


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(2,1,kernel_size=kernel_size,padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        weight = torch.cat([avg_out, max_out], dim=1)
        weight = self.sigmoid(self.conv(weight))
        return x * weight


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel_size=7):
        super().__init__()

        self.channel_attention = ChannelAttention(channels=channels,reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# Attention Wrapper
class AttentionWrapper(nn.Module):
    """
    用一个 wrapper 包住 torchvision 原来的 ResNet BasicBlock。
    好处：
    1. 不需要自己重写 ResNet18；
    2. 可以直接加载 ImageNet 预训练权重；
    3. 只额外添加 SE / CBAM 模块。
    """

    def __init__(self, block, attention):
        super().__init__()

        self.block = block
        self.attention = attention

    def forward(self, x):
        x = self.block(x)
        x = self.attention(x)
        return x


# Build ResNet
def build_resnet18(num_classes=102, pretrained=True):
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def add_attention_to_resnet18(model, attention_type="se"):
    """
    给 ResNet18 的每个 BasicBlock 后面加注意力模块。
    """
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name)

        for i, block in enumerate(layer):
            channels = block.conv2.out_channels
            if attention_type == "se":
                attention = SEBlock(channels)
            elif attention_type == "cbam":
                attention = CBAMBlock(channels)
            else:
                raise ValueError(f"Unknown attention type: {attention_type}")

            layer[i] = AttentionWrapper(block, attention)
    return model


def build_resnet18_se(num_classes=102, pretrained=True):
    model = build_resnet18(num_classes=num_classes,pretrained=pretrained)
    model = add_attention_to_resnet18(model,attention_type="se")
    return model


def build_resnet18_cbam(num_classes=102, pretrained=True):
    model = build_resnet18(num_classes=num_classes,pretrained=pretrained)
    model = add_attention_to_resnet18(model,attention_type="cbam")
    return model


def build_model(model_name, num_classes=102, pretrained=True):
    if model_name == "resnet18":
        return build_resnet18(num_classes=num_classes,pretrained=pretrained)
    if model_name == "resnet18_se":
        return build_resnet18_se(num_classes=num_classes,pretrained=pretrained)
    if model_name == "resnet18_cbam":
        return build_resnet18_cbam(num_classes=num_classes,pretrained=pretrained)

    raise ValueError(f"Unknown model name: {model_name}")