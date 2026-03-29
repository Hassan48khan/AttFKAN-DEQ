"""
lcbam.py
--------
Lightweight Convolutional Block Attention Module (LCBAM).

LCBAM is a streamlined alternative to CBAM (Woo et al., ECCV 2018) designed
for inclusion inside the DEQ iteration loop, where repeated execution makes
parameter efficiency critical.

Key differences from CBAM:
  - Channel attention: MLP replaced by two 1×1 Conv layers + BatchNorm
  - Spatial attention: standard 7×7 Conv replaced by depthwise 7×7 Conv
  - Parameter count: ~2C²/r  vs  CBAM's ~2C²  (16× reduction at r=16)

Structure (sequential):
    Input M ∈ ℝ^{C×H×W}
      → Channel Attention M_c ∈ ℝ^{C×1×1}  →  M' = M ⊗ M_c
      → Spatial Attention M_s ∈ ℝ^{1×H×W}  →  M'' = M' ⊗ M_s
    Output M'' ∈ ℝ^{C×H×W}
"""

import torch
import torch.nn as nn


class LCBAM(nn.Module):
    """
    Lightweight Convolutional Block Attention Module.

    Args:
        channels (int): Number of input/output channels C.
        reduction_ratio (int): Channel reduction factor r. Default: 16.
    """

    def __init__(self, channels: int, reduction_ratio: int = 16):
        super(LCBAM, self).__init__()

        reduced_c = max(channels // reduction_ratio, 8)

        # ── Channel Attention ────────────────────────────────────────────────
        # AdaptiveAvgPool → Conv1×1 (C→C/r) → BN → ReLU → Conv1×1 (C/r→C) → Sigmoid
        # Parameters ≈ 2C²/r  (vs CBAM MLP: 2C²/r × 2 due to MaxPool branch)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                              # [B, C, 1, 1]
            nn.Conv2d(channels, reduced_c, kernel_size=1, bias=True),
            nn.BatchNorm2d(reduced_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_c, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),                                         # M_c ∈ [0,1]^{C×1×1}
        )

        # ── Spatial Attention ────────────────────────────────────────────────
        # Concat(AvgPool_C, MaxPool_C) → DWConv 7×7 → BN → Sigmoid
        # Depthwise conv operates on 2 channels; params ≈ 7²×2 = 98
        # vs standard 7×7 Conv: 7²×2×C params
        self.spatial_attn = nn.Sequential(
            # Input: 2 channels (avg + max along channel dim)
            nn.Conv2d(2, 2, kernel_size=7, padding=3, groups=2, bias=False),  # depthwise
            nn.Conv2d(2, 1, kernel_size=1, bias=False),                        # pointwise
            nn.BatchNorm2d(1),
            nn.Sigmoid(),                                          # M_s ∈ [0,1]^{1×H×W}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map [B, C, H, W].

        Returns:
            Attended feature map [B, C, H, W].
        """
        # Channel attention
        ca = self.channel_attn(x)        # [B, C, 1, 1]
        x = x * ca                       # broadcast: [B, C, H, W]

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)        # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)      # [B, 1, H, W]
        sa_input = torch.cat([avg_out, max_out], dim=1)      # [B, 2, H, W]
        sa = self.spatial_attn(sa_input)                     # [B, 1, H, W]
        x = x * sa                                           # [B, C, H, W]

        return x
