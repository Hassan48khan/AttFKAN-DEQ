"""
fourier_kan.py
--------------
Naive Fourier KAN Layer implementation.

Based on: "FourierKAN-GCF: Fourier Kolmogorov-Arnold Network –
An Effective and Efficient Feature Transformation for Graph Collaborative Filtering"
Xu et al., arXiv:2406.01034, 2024.

Each edge in the layer has its own learnable Fourier series:
    phi_{j,i}(z) = b_{j,i} + sum_{k=1}^{g} [ a_{j,i,k} * cos(k*z) + b_{j,i,k} * sin(k*z) ]

This replaces fixed activation functions with per-edge trainable univariate functions,
giving FKANs high expressivity for multi-scale and frequency-domain pattern modeling.
"""

import torch
import torch.nn as nn
import numpy as np


class NaiveFourierKANLayer(nn.Module):
    """
    Fourier KAN Layer.

    Maps input of shape (..., inputdim) to output of shape (..., outdim)
    using per-edge Fourier series activations.

    Args:
        inputdim (int): Number of input features.
        outdim (int): Number of output features.
        gridsize (int): Number of Fourier frequency terms (harmonics). Default: 8.
        addbias (bool): Whether to add a learnable bias. Default: True.
        smooth_initialization (bool): If True, initializes higher-frequency
            coefficients with smaller magnitude for smoother start. Default: False.
    """

    def __init__(
        self,
        inputdim: int,
        outdim: int,
        gridsize: int = 8,
        addbias: bool = True,
        smooth_initialization: bool = False,
    ):
        super(NaiveFourierKANLayer, self).__init__()

        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        # Normalization: smooth init dampens high-frequency terms
        if smooth_initialization:
            # Decay by k^2 so high-frequency coefficients start near zero
            grid_norm_factor = (torch.arange(1, gridsize + 1, dtype=torch.float32)) ** 2
        else:
            grid_norm_factor = float(np.sqrt(gridsize))

        # Fourier coefficients: shape [2, outdim, inputdim, gridsize]
        #   [0, ...] -> cosine coefficients (a_{j,i,k})
        #   [1, ...] -> sine   coefficients (b_{j,i,k})
        self.fouriercoeffs = nn.Parameter(
            torch.randn(2, outdim, inputdim, gridsize) / (np.sqrt(inputdim) * grid_norm_factor)
        )

        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (..., inputdim).

        Returns:
            Output tensor of shape (..., outdim).
        """
        xshp = x.shape
        outshape = xshp[:-1] + (self.outdim,)

        # Flatten to 2D for computation: [N, inputdim]
        x = x.reshape(-1, self.inputdim)

        # Frequency indices k = 1, 2, ..., gridsize
        # Shape: [1, 1, 1, gridsize]
        k = torch.arange(1, self.gridsize + 1, device=x.device, dtype=x.dtype).reshape(1, 1, 1, self.gridsize)

        # Expand x: [N, 1, inputdim, 1]
        xrshp = x.reshape(x.shape[0], 1, x.shape[1], 1)

        # Compute cosine and sine basis: [N, 1, inputdim, gridsize]
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)

        # Weighted sum over input dims and frequencies
        # fouriercoeffs[0]: [outdim, inputdim, gridsize] -> broadcast with c: [N, 1, inputdim, gridsize]
        # Result after sum over (-2, -1): [N, outdim]
        y = torch.sum(c * self.fouriercoeffs[0:1], dim=(-2, -1))
        y = y + torch.sum(s * self.fouriercoeffs[1:2], dim=(-2, -1))

        if self.addbias:
            y = y + self.bias

        # Restore original leading dimensions
        y = y.reshape(outshape)
        return y


def demo():
    """Quick sanity check."""
    print("=" * 50)
    print("NaiveFourierKANLayer Demo")
    print("=" * 50)

    B, L, D = 4, 1, 128
    outdim = 128
    gridsize = 8

    layer = NaiveFourierKANLayer(inputdim=D, outdim=outdim, gridsize=gridsize, smooth_initialization=True)
    x = torch.randn(B, L, D)
    y = layer(x)

    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {y.shape}")
    print(f"Params       : {sum(p.numel() for p in layer.parameters()):,}")
    assert y.shape == (B, L, outdim), "Shape mismatch!"
    print("✓ Forward pass OK")


if __name__ == "__main__":
    demo()
