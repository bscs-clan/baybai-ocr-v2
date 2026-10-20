"""Small CNN + temporal convolution OCR model designed for fixed-shape mobile inference."""
from typing import Optional
import torch
from torch import nn


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=1):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class TemporalConvBlock(nn.Module):
    """Residual depthwise-separable temporal convolution; no recurrent ops."""
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=5, padding=2 * dilation,
            dilation=dilation, groups=channels, bias=False,
        )
        self.bn = nn.BatchNorm1d(channels)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.act(self.bn(x))
        x = self.pointwise_bn(self.pointwise(x))
        return self.act(x + residual)


class MobileBaybayinOCR(nn.Module):
    """Mobile-first CTC OCR.

    Input:  (B, 1, 48, W)
    Output: (B, ceil(W/2), num_classes), with blank class at index 0.

    The feature-map height is collapsed by a learned (6x1) convolution so that
    above/below kudlit evidence is combined learnedly rather than averaged away.
    Keep W fixed for export and benchmark on the actual target mobile device.
    """
    def __init__(self, num_classes: int, feature_channels: int = 128, temporal_blocks: int = 4):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feature_channels = int(feature_channels)
        self.temporal_blocks = int(temporal_blocks)

        self.backbone = nn.Sequential(
            ConvBNAct(1, 32, stride=(2, 2)),     # H:48->24, W:W->ceil(W/2)
            DepthwiseSeparableConv2d(32, 48),
            nn.MaxPool2d(kernel_size=(2, 1)),    # H:24->12
            DepthwiseSeparableConv2d(48, 64),
            nn.MaxPool2d(kernel_size=(2, 1)),    # H:12->6
            DepthwiseSeparableConv2d(64, 96),
        )
        self.height_projection = nn.Sequential(
            nn.Conv2d(96, feature_channels, kernel_size=(6, 1), bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU6(inplace=True),
        )
        dilations = [2 ** (i % 4) for i in range(temporal_blocks)]
        self.temporal = nn.Sequential(*[
            TemporalConvBlock(feature_channels, dilation=d) for d in dilations
        ])
        self.classifier = nn.Linear(feature_channels, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        features = self.height_projection(features)
        # [B, C, 1, T] -> [B, C, T]
        features = features.squeeze(2)
        features = self.temporal(features)
        # [B, C, T] -> [B, T, C]
        features = features.transpose(1, 2).contiguous()
        return self.classifier(features)

    def count_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters()
        return sum(p.numel() for p in params if (p.requires_grad or not trainable_only))


def build_model(num_classes: int, config: Optional[dict] = None) -> MobileBaybayinOCR:
    config = config or {}
    return MobileBaybayinOCR(
        num_classes=num_classes,
        feature_channels=int(config.get("feature_channels", 128)),
        temporal_blocks=int(config.get("temporal_blocks", 4)),
    )
