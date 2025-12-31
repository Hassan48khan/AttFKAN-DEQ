# model/lcbam.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LCBAM(nn.Module):
    """
    Lightweight Convolutional Block Attention Module (LCBAM).
    Sequential channel and spatial attention with reduced parameters.
    """
    def __init__(self, channels, reduction_ratio=16):
        super(LCBAM, self).__init__()
        reduced_c = max(channels // reduction_ratio, 8)

        # Channel Attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_c, kernel_size=1, bias=True),
            nn.BatchNorm2d(reduced_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_c, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # Spatial Attention (depthwise 7x7 convolution)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        # Channel attention
        ca = self.channel_attn(x)
        x = x * ca

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attn(sa_input)
        x = x * sa

        return x
