import torch
import torch.nn as nn
import torch.nn.functional as F
from .fourier_kan import NaiveFourierKANLayer
from .lcbam import LCBAM  # Use your full LCBAM

class AttFKANBlock(nn.Module):
    def __init__(self, dim, grid_size=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fkan1 = NaiveFourierKANLayer(dim, dim, grid_size)
        self.norm2 = nn.LayerNorm(dim)
        self.fkan2 = NaiveFourierKANLayer(dim, dim, grid_size)
        self.lcbam = LCBAM(dim)  # Your full attention

    def forward(self, x):
        residual = x
        out = self.fkan1(self.norm1(x))
        out = F.relu(out)
        out = self.fkan2(self.norm2(out))
        # Reshape for LCBAM (assuming vectorized features)
        B, L, C = out.shape
        out = out.view(B, C, 1, L)
        out = self.lcbam(out)
        out = out.view(B, L, C)
        return residual + out

class AttFKAN_DEQ(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=512, num_classes=2, grid_size=8, max_iters=10, alpha=0.5):
        super().__init__()
        self.cnn_backbone = nn.Sequential(  # Replace with ResNet if desired
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            # Add more layers as needed
        )
        self.proj = nn.Linear(64, hidden_dim)  # Adjust based on backbone output
        self.block = AttFKANBlock(hidden_dim, grid_size)
        self.max_iters = max_iters
        self.alpha = alpha
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        features = self.cnn_backbone(x)
        p = F.adaptive_avg_pool2d(features, 1).view(B, -1)
        p = self.proj(p).unsqueeze(1)  # [B, 1, hidden_dim]
        z = torch.zeros_like(p, device=x.device)

        for _ in range(self.max_iters):
            z_prev = z
            z = z_prev + p + self.block(z_prev + p)
            z = (1 - self.alpha) * z_prev + self.alpha * z

        logits = self.classifier(z.squeeze(1))
        return logits

# model/attfkan_deq.py
# Final Integrated AttFKAN-DEQ Model
# Combines:
# 1. Fourier-based KAN (FKAN) for expressive univariate transformations
# 2. Lightweight Convolutional Block Attention (LCBAM) for dynamic feature recalibration
# 3. Deep Equilibrium (DEQ) framework for memory-efficient infinite-depth refinement

import torch
import torch.nn as nn
import torch.nn.functional as F
from .fourier_kan import NaiveFourierKANLayer
from .lcbam import LCBAM

class AttFKANBlock(nn.Module):
    """
    Attention-Augmented Fourier KAN Residual Block.
    
    This is the core nonlinear transformation f_AttFKAN(z + p) used in the DEQ fixed-point equation.
    
    Architecture:
    - LayerNorm → FKAN1 → ReLU → LayerNorm → FKAN2 → Reshape to [B, C, 1, 1] → LCBAM → Reshape back → Residual
    
    The block enables:
    - Expressive multi-scale modeling via Fourier series
    - Dynamic channel and spatial focus via lightweight attention
    - Stable gradient flow via residual connection and normalization
    """
    def __init__(self, dim: int, grid_size: int = 8, reduction_ratio: int = 16):
        super(AttFKANBlock, self).__init__()
        
        # Pre-FKAN normalization
        self.norm1 = nn.LayerNorm(dim)
        self.fkan1 = NaiveFourierKANLayer(
            inputdim=dim,
            outdim=dim,
            gridsize=grid_size,
            addbias=True,
            smooth_initialization=True
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.fkan2 = NaiveFourierKANLayer(
            inputdim=dim,
            outdim=dim,
            gridsize=grid_size,
            addbias=True,
            smooth_initialization=True
        )
        
        # Lightweight attention module
        self.lcbam = LCBAM(channels=dim, reduction_ratio=reduction_ratio)

    def forward(self, x):
        """
        Args:
            x: Hidden state tensor [B, 1, hidden_dim] (seq_len=1 in DEQ context)
        
        Returns:
            Refined tensor with residual connection [B, 1, hidden_dim]
        """
        residual = x
        
        # First FKAN transformation
        out = self.norm1(x)
        out = self.fkan1(out)
        out = F.relu(out)
        
        # Second FKAN transformation
        out = self.norm2(out)
        out = self.fkan2(out)
        
        # Apply LCBAM: treat vector as 1x1 "image"
        B, L, C = out.shape
        out = out.view(B, C, 1, L)  # [B, C, 1, seq_len]
        out = self.lcbam(out)
        out = out.view(B, L, C)     # Back to [B, seq_len, C]
        
        # Residual connection
        out = residual + out
        
        return out


class AttFKAN_DEQ(nn.Module):
    """
    Complete AttFKAN-DEQ Model.
    
    Full pipeline:
    1. CNN backbone extracts spatial features from histopathological image
    2. Global average pooling + linear projection creates fixed injection vector p ∈ ℝ^hidden_dim
    3. DEQ solves the fixed-point equation:
          z* = z* + p + f_AttFKAN(z* + p)
       using relaxed fixed-point iteration with parameter α
    4. Equilibrium state z* is passed to classifier for benign/malignant prediction
    
    This design enables infinite-depth refinement with constant memory usage.
    """
    def __init__(self,
                 in_channels: int = 3,
                 hidden_dim: int = 512,
                 num_classes: int = 2,
                 grid_size: int = 8,
                 max_iters: int = 10,
                 tol: float = 1e-4,
                 alpha: float = 0.5,
                 backbone: str = 'resnet50'):
        super(AttFKAN_DEQ, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.max_iters = max_iters
        self.tol = tol
        self.alpha = alpha
        
        # CNN backbone for initial feature extraction
        if backbone == 'resnet50':
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
            backbone_channels = 2048
        elif backbone == 'resnet18':
            import torchvision.models as models
            resnet = models.resnet18(pretrained=True)
            self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
            backbone_channels = 512
        else:
            # Custom lightweight backbone
            self.cnn_backbone = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            )
            backbone_channels = hidden_dim
        
        # Projection from backbone features to hidden dimension
        self.proj = nn.Linear(backbone_channels, hidden_dim)
        
        # Core AttFKAN residual block for DEQ
        self.attfkan_block = AttFKANBlock(
            dim=hidden_dim,
            grid_size=grid_size,
            reduction_ratio=16
        )
        
        # Final classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass of AttFKAN-DEQ.
        
        Args:
            x: Input histopathological image [B, 3, H, W]
        
        Returns:
            logits: Classification logits [B, num_classes]
        """
        B = x.shape[0]
        
        # 1. CNN feature extraction
        features = self.cnn_backbone(x)  # [B, C_backbone, H', W']
        
        # 2. Create fixed injection vector p
        p = F.adaptive_avg_pool2d(features, 1)  # [B, C_backbone, 1, 1]
        p = p.view(B, -1)                        # [B, C_backbone]
        p = self.proj(p)                         # [B, hidden_dim]
        p = p.unsqueeze(1)                       # [B, 1, hidden_dim] — injected every iteration
        
        # 3. Initialize hidden state z^(0) = 0
        z = torch.zeros_like(p, device=x.device)
        
        # 4. DEQ: Solve z* = z* + p + f_AttFKAN(z* + p) using relaxed iteration
        for _ in range(self.max_iters):
            z_prev = z
            
            # Compute f_AttFKAN(z + p)
            z_input = z_prev + p
            z_update = self.attfkan_block(z_input)
            
            # Update: z + p + f_AttFKAN(z + p)
            z_new = z_prev + p + z_update
            
            # Relaxation: smooth update for better convergence
            z = (1 - self.alpha) * z_prev + self.alpha * z_new
            
            # Early stopping based on relative change
            if torch.norm(z - z_prev) < self.tol * (torch.norm(z_prev) + 1e-8):
                break
        
        # 5. Classification from equilibrium state z*
        logits = self.classifier(z.squeeze(1))  # [B, num_classes]
        
        return logits
