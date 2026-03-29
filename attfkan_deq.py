"""
attfkan_deq.py
--------------
AttFKAN-DEQ: Attention-Enhanced Fourier Kolmogorov-Arnold Networks
             using Deep Equilibrium for Breast Cancer Histopathology Classification.

Pipeline
--------
1.  CNN backbone  →  spatial features  [B, C_bb, H', W']
2.  Global average pool + linear projection  →  fixed injection vector p  [B, h]
3.  DEQ fixed-point iteration  →  equilibrium state z*  [B, h]
        z^{k+1} = (1-α)z^k + α · (p + f_AttFKAN(z^k + p))
4.  Linear classifier  →  logits  [B, num_classes]

The AttFKAN block (f_AttFKAN) is a residual structure:
        f_AttFKAN(u) = u + LCBAM( FKAN2( ReLU( FKAN1( LN(u) ) ) ) )
with u = z + p.

References
----------
- Bai et al., "Deep Equilibrium Models", NeurIPS 2019.
- Xu et al., "FourierKAN-GCF", arXiv:2406.01034, 2024.
- Woo et al., "CBAM", ECCV 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fourier_kan import NaiveFourierKANLayer
from lcbam import LCBAM


# ── Lightweight CNN Backbone ─────────────────────────────────────────────────

class BasicBlock(nn.Module):
    """Simple residual block used in the custom CNN backbone."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x) + self.skip(x), inplace=True)


def build_backbone(in_channels: int, base_ch: int = 32) -> tuple:
    """
    Build the lightweight custom CNN backbone described in Figure 2 of the paper.

    Architecture (matching paper diagram):
        Conv2d(in→base_ch, k=7, s=2) + BN + ReLU  →  [B, base_ch, H/2, W/2]
        BasicBlock(stride=1)                        →  [B, base_ch, H/2, W/2]
        BasicBlock(stride=2)                        →  [B, 2*base_ch, H/4, W/4]
        BasicBlock(stride=2)                        →  [B, 4*base_ch, H/8, W/8]
        BasicBlock(stride=2)                        →  [B, 4*base_ch, H/16, W/16]
        AdaptiveAvgPool2d(1)  →  Flatten            →  [B, 4*base_ch]

    Returns:
        (backbone_module, out_channels)
    """
    out_ch = base_ch * 4
    backbone = nn.Sequential(
        nn.Conv2d(in_channels, base_ch, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(base_ch),
        nn.ReLU(inplace=True),
        BasicBlock(base_ch, base_ch, stride=1),
        BasicBlock(base_ch, base_ch * 2, stride=2),
        BasicBlock(base_ch * 2, out_ch, stride=2),
        BasicBlock(out_ch, out_ch, stride=2),
    )
    return backbone, out_ch


# ── AttFKAN Block ─────────────────────────────────────────────────────────────

class AttFKANBlock(nn.Module):
    """
    Attention-Augmented Fourier KAN Residual Block  (f_AttFKAN).

    This is the core nonlinear transformation used inside the DEQ loop.

    Given u = z + p  ∈  ℝ^{B×1×h}, the block computes:
        out = u + LCBAM( FKAN2( ReLU( FKAN1( LayerNorm(u) ) ) ) )

    The LCBAM is applied by temporarily reshaping the sequence to a
    [B, C, 1, L] pseudo-image so the 2-D attention module is compatible.

    Args:
        dim (int): Hidden dimension h.
        grid_size (int): Fourier frequency terms per edge. Default: 8.
        reduction_ratio (int): LCBAM channel reduction ratio. Default: 16.
        dropout (float): Dropout probability applied after each FKAN. Default: 0.2.
    """

    def __init__(
        self,
        dim: int,
        grid_size: int = 8,
        reduction_ratio: int = 16,
        dropout: float = 0.2,
    ):
        super(AttFKANBlock, self).__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.fkan1 = NaiveFourierKANLayer(
            inputdim=dim,
            outdim=dim,
            gridsize=grid_size,
            addbias=True,
            smooth_initialization=True,
        )
        self.drop1 = nn.Dropout(p=dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.fkan2 = NaiveFourierKANLayer(
            inputdim=dim,
            outdim=dim,
            gridsize=grid_size,
            addbias=True,
            smooth_initialization=True,
        )
        self.drop2 = nn.Dropout(p=dropout)

        # LCBAM expects [B, C, H, W]; we reshape [B, L, C] → [B, C, 1, L]
        self.lcbam = LCBAM(channels=dim, reduction_ratio=reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, dim]  (L = 1 in the DEQ context)

        Returns:
            Refined tensor [B, L, dim] with residual connection.
        """
        residual = x

        # First FKAN pass
        out = self.norm1(x)
        out = self.fkan1(out)
        out = self.drop1(F.relu(out, inplace=False))

        # Second FKAN pass
        out = self.norm2(out)
        out = self.fkan2(out)
        out = self.drop2(out)

        # LCBAM: reshape [B, L, C] → [B, C, 1, L] → [B, L, C]
        B, L, C = out.shape
        out = out.permute(0, 2, 1).unsqueeze(2)   # [B, C, 1, L]
        out = self.lcbam(out)                      # [B, C, 1, L]
        out = out.squeeze(2).permute(0, 2, 1)      # [B, L, C]

        return residual + out


# ── Full AttFKAN-DEQ Model ────────────────────────────────────────────────────

class AttFKAN_DEQ(nn.Module):
    """
    AttFKAN-DEQ: Full model for histopathological breast cancer classification.

    Args:
        in_channels (int): Input image channels (3 for RGB). Default: 3.
        hidden_dim (int): DEQ hidden state dimension h. Default: 128.
        num_classes (int): Number of output classes (2 for benign/malignant). Default: 2.
        grid_size (int): Fourier grid size g. Default: 8.
        max_iters (int): Maximum DEQ fixed-point iterations. Default: 10.
        tol (float): Convergence tolerance for early stopping. Default: 1e-4.
        alpha (float): DEQ relaxation coefficient α ∈ (0, 1]. Default: 0.5.
        backbone (str): CNN backbone choice: 'custom' | 'resnet18' | 'resnet50'.
                        Default: 'custom'.
        base_ch (int): Base channel count for the custom backbone. Default: 32.
        dropout (float): Dropout inside AttFKAN block. Default: 0.2.
        reduction_ratio (int): LCBAM reduction ratio. Default: 16.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 128,
        num_classes: int = 2,
        grid_size: int = 8,
        max_iters: int = 10,
        tol: float = 1e-4,
        alpha: float = 0.5,
        backbone: str = "custom",
        base_ch: int = 32,
        dropout: float = 0.2,
        reduction_ratio: int = 16,
    ):
        super(AttFKAN_DEQ, self).__init__()

        self.hidden_dim = hidden_dim
        self.max_iters = max_iters
        self.tol = tol
        self.alpha = alpha

        # ── CNN Backbone ──────────────────────────────────────────────────
        if backbone == "resnet50":
            import torchvision.models as models
            _resnet = models.resnet50(weights="IMAGENET1K_V1")
            self.cnn_backbone = nn.Sequential(*list(_resnet.children())[:-2])
            backbone_channels = 2048
        elif backbone == "resnet18":
            import torchvision.models as models
            _resnet = models.resnet18(weights="IMAGENET1K_V1")
            self.cnn_backbone = nn.Sequential(*list(_resnet.children())[:-2])
            backbone_channels = 512
        else:
            # Custom lightweight backbone (paper Figure 2)
            self.cnn_backbone, backbone_channels = build_backbone(in_channels, base_ch)

        # ── Projection: backbone features → hidden dim ────────────────────
        self.proj = nn.Linear(backbone_channels, hidden_dim)

        # ── AttFKAN block (shared across all DEQ iterations) ──────────────
        self.attfkan_block = AttFKANBlock(
            dim=hidden_dim,
            grid_size=grid_size,
            reduction_ratio=reduction_ratio,
            dropout=dropout,
        )

        # ── Classifier ────────────────────────────────────────────────────
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Weight initialisation
        self._init_weights()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image [B, C, H, W].

        Returns:
            logits: Classification logits [B, num_classes].
        """
        B = x.shape[0]

        # 1. CNN feature extraction
        features = self.cnn_backbone(x)                        # [B, C_bb, H', W']

        # 2. Build fixed injection vector p
        p = F.adaptive_avg_pool2d(features, 1).view(B, -1)    # [B, C_bb]
        p = self.proj(p).unsqueeze(1)                          # [B, 1, hidden_dim]

        # 3. DEQ: relaxed fixed-point iteration
        #    z^{k+1} = (1-α)z^k + α·(p + f_AttFKAN(z^k + p))
        z = torch.zeros_like(p)                                 # z^{(0)} = 0

        for _ in range(self.max_iters):
            z_prev = z

            z_input = z_prev + p                               # [B, 1, hidden_dim]
            z_update = self.attfkan_block(z_input)             # [B, 1, hidden_dim]
            z_new = z_prev + p + z_update                      # F_θ(x, z^k)

            z = (1.0 - self.alpha) * z_prev + self.alpha * z_new

            # Early stopping
            delta = torch.norm(z - z_prev)
            scale = torch.norm(z_prev) + 1e-8
            if delta < self.tol * scale:
                break

        # 4. Classify from equilibrium state z*
        logits = self.classifier(z.squeeze(1))                 # [B, num_classes]
        return logits

    # ── Utility ───────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quick Sanity Check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    model = AttFKAN_DEQ(
        in_channels=3,
        hidden_dim=128,
        num_classes=2,
        grid_size=8,
        max_iters=10,
        alpha=0.5,
        backbone="custom",
    ).to(device)

    dummy = torch.randn(4, 3, 50, 50, device=device)
    logits = model(dummy)

    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {logits.shape}")
    print(f"Parameters   : {model.count_parameters():,}")
    assert logits.shape == (4, 2), "Unexpected output shape"
    print("✓ Forward pass OK")
