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
