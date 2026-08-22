"""Lightweight Panoptic-DeepLab style U-Net."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class PanopticUNet(nn.Module):
    """Shared encoder with semantic, center, and offset prediction heads."""

    def __init__(self, in_channels: int = 3, num_classes: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.encoders = nn.ModuleList(
            [
                ConvBlock(in_channels, widths[0]),
                ConvBlock(widths[0], widths[1]),
                ConvBlock(widths[1], widths[2]),
                ConvBlock(widths[2], widths[3]),
            ]
        )
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(widths[-1], widths[-1] * 2)
        self.ups = nn.ModuleList(
            [
                nn.ConvTranspose2d(widths[-1] * 2, widths[-1], 2, 2),
                nn.ConvTranspose2d(widths[-1], widths[-2], 2, 2),
                nn.ConvTranspose2d(widths[-2], widths[-3], 2, 2),
                nn.ConvTranspose2d(widths[-3], widths[-4], 2, 2),
            ]
        )
        self.refine = nn.ModuleList(
            [
                ConvBlock(widths[-1] * 2, widths[-1]),
                ConvBlock(widths[-2] * 2, widths[-2]),
                ConvBlock(widths[-3] * 2, widths[-3]),
                ConvBlock(widths[-4] * 2, widths[-4]),
            ]
        )
        self.semantic_head = nn.Conv2d(widths[0], num_classes, 1)
        self.center_head = nn.Conv2d(widths[0], 1, 1)
        self.offset_head = nn.Conv2d(widths[0], 2, 1)
        self.checkpoint_metadata = {"architecture": "panoptic_unet_small", "head": "semantic_center_offset"}

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if image.ndim != 4 or image.shape[-2] % 16 or image.shape[-1] % 16:
            raise ValueError("PanopticUNet expects [B,C,H,W] with H and W divisible by 16")
        skips = []
        value = image
        for encoder in self.encoders:
            value = encoder(value)
            skips.append(value)
            value = self.pool(value)
        value = self.bottleneck(value)
        for up, refine, skip in zip(self.ups, self.refine, reversed(skips), strict=True):
            value = refine(torch.cat([up(value), skip], dim=1))
        return {
            "semantic": self.semantic_head(value),
            "center": self.center_head(value),
            "offset": self.offset_head(value),
        }
