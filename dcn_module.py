"""Standalone DCNv2 module for experimentation.

This module mirrors the Ultralytics-integrated DCNv2 block used in Phase 2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


class DCNv2(nn.Module):
    """Deformable Convolution v2 with learned offsets and modulation mask.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        k (int): Kernel size.
        s (int): Stride.
        p (int | None): Padding. If None, uses k//2.
        g (int): Convolution groups.
        act (bool | nn.Module): Activation function.
    """

    default_act = nn.SiLU()

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int | None = None, g: int = 1, act=True):
        super().__init__()
        if p is None:
            p = k // 2

        self.offset_conv = nn.Conv2d(c1, 2 * k * k, kernel_size=k, stride=s, padding=p, groups=g, bias=True)
        self.mask_conv = nn.Conv2d(c1, k * k, kernel_size=k, stride=s, padding=p, groups=g, bias=True)
        self.conv = DeformConv2d(c1, c2, kernel_size=k, stride=s, padding=p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        return self.act(self.bn(self.conv(x, offset, mask)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        return self.act(self.conv(x, offset, mask))
