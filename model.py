"""
AnguyiNet: a compact, attention-augmented CNN for CIFAR-10.

This module defines a custom architecture (not ResNet/VGG/etc.) with
depthwise-separable convolutions for efficiency and squeeze-excitation-style
channel attention for adaptive feature recalibration.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnguyiNetBlock1(nn.Module):
    """
    First convolutional macro-block.

    Architectural role: extract low-level edges, color blobs, and local textures
    from raw RGB pixels; batch normalization keeps activations in a stable range
    for deep stacks; max pooling downsamples once so deeper layers see a wider
    receptive field without stacking many stride-1 convolutions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AnguyiNetBlock2(nn.Module):
    """
    Second convolutional macro-block.

    Architectural role: increase channel width to encode richer part combinations
    while preserving moderate spatial resolution; the second 3x3 convolution
    refines local mixing before pooling; this stage feeds the attention module
    with semantically richer maps than the first block.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention (Hu et al., 2018).

    Architectural role: each channel gets a scalar gate computed from global
    context, so the network can emphasize informative feature maps and suppress
    redundant ones without changing spatial resolution.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DepthwiseSeparableBlock(nn.Module):
    """
    Third convolutional macro-block using depthwise separable convolutions.

    Architectural role: depthwise 3x3 convolutions apply spatial filtering per
    channel with shared geometric prior but independent scales; pointwise 1x1
    convolutions mix channels at each spatial location. This factorization
    sharply reduces parameters versus a dense 3x3 of comparable width, which
    acts as structural regularization on small datasets.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1, groups=96, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1, groups=192, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GlobalPoolClassifierHead(nn.Module):
    """
    Global pooling, dropout, and linear classifier.

    Architectural role: global average pooling averages each feature map to a
    scalar, producing a compact 192-dimensional descriptor without a huge
    flattened vector that would require an enormous fully connected weight
    matrix (a major overfitting risk on CIFAR-scale data). Dropout perturbs
    this descriptor during training; the linear layer maps to class logits.
    Probabilities are softmax(logits); training uses CrossEntropyLoss on logits.
    """

    def __init__(self, in_channels: int, num_classes: int, dropout_p: float) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gap(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)


class AnguyiNet(nn.Module):
    """
    AnguyiNet — compact CNN for 32x32 RGB inputs (CIFAR-10).

    Composes two standard pooled convolutional stages, channel attention,
    a depthwise separable stage, and a global-pooling classifier head.
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout_p: float = 0.4,
        se_reduction: int = 8,
    ) -> None:
        super().__init__()
        self.block1 = AnguyiNetBlock1()
        self.block2 = AnguyiNetBlock2()
        self.channel_attn = ChannelAttention(96, reduction=se_reduction)
        self.block3 = DepthwiseSeparableBlock()
        self.head = GlobalPoolClassifierHead(192, num_classes, dropout_p)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.channel_attn(x)
        x = self.block3(x)
        return self.head(x)

    def class_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax class probabilities (inference-style output)."""
        return F.softmax(self.forward(x), dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, device: torch.device) -> None:
    """Print layer names, output shapes, and total parameter count."""
    model = model.to(device)
    dummy = torch.zeros(1, 3, 32, 32, device=device)
    print("\n=== AnguyiNet summary (input 1x3x32x32) ===\n")
    hooks = []

    def hook(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        class_name = module.__class__.__name__
        if isinstance(out, torch.Tensor):
            print(f"{class_name:28s} output: {tuple(out.shape)}")
        else:
            print(f"{class_name:28s} output: (non-tensor)")

    for module in model.children():
        hooks.append(module.register_forward_hook(hook))

    with torch.no_grad():
        _ = model(dummy)

    for h in hooks:
        h.remove()

    total = count_parameters(model)
    print(f"\nTotal trainable parameters: {total:,}\n")


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AnguyiNet()
    print_model_summary(net, dev)
