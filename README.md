# AttFKAN-DEQ

**Attention-Enhanced Fourier Kolmogorov–Arnold Networks Using Deep Equilibrium for Breast Cancer Histopathology Classification**

> Hassan Ali · Muhammad Asghar Khan · Gordana Barb
> Published in *IEEE Access* (2026)
> DOI: [10.1109/ACCESS.2026.3577827](https://doi.org/10.1109/ACCESS.2026.3577827)

---
## 🧠 Architecture Diagram

<p align="center">
  <img src="Graphical abstract.png" alt="AttFKAN-DEQ Architecture" width="800"/>
</p>

<p align="center">
  <em>Figure: Overview of the proposed AttFKAN-DEQ architecture integrating FKAN layers, LCBAM attention, and DEQ-based iterative refinement.</em>
</p>
## Overview

AttFKAN-DEQ is a hybrid deep-learning framework for binary benign-vs-malignant classification of breast histopathological images. It fuses three complementary ideas into a single, memory-efficient pipeline:

| Component | Role |
|-----------|------|
| **Fourier KAN (FKAN)** | Replaces fixed activations with per-edge learnable Fourier series, capturing multi-scale frequency patterns in tissue textures |
| **Lightweight CBAM (LCBAM)** | Dynamically recalibrates channel and spatial features during DEQ iterations using depthwise convolutions (≈98 params for spatial attention) |
| **Deep Equilibrium (DEQ)** | Solves for a fixed-point hidden state *z\** instead of stacking layers — infinite effective depth with constant memory |

The full pipeline is:

```
Image x
  └─ CNN Backbone ──► Global Avg Pool ──► Linear ──► p  [B, h]
                                                       │
                          ┌────────────────────────────┘
                          ▼
                     z⁽⁰⁾ = 0
                     ┌────────────────────────────────────────────┐
                     │  z⁽ᵏ⁺¹⁾ = (1−α)z⁽ᵏ⁾ + α·(p + f_AttFKAN(z⁽ᵏ⁾+p)) │  × max_iters
                     └────────────────────────────────────────────┘
                          │
                          ▼  z*  (equilibrium state)
                     Linear Classifier ──► logits  [B, 2]
```

where `f_AttFKAN(u) = u + LCBAM(FKAN₂(ReLU(FKAN₁(LN(u)))))`.

---

## Results

### BreakHis Dataset (all magnifications, 5-fold CV)

| Model | Accuracy | Precision | Recall | Specificity | F1-Score | AUC |
|-------|----------|-----------|--------|-------------|----------|-----|
| ResNet-18 | 89.45 | 88.72 | 87.91 | 90.14 | 88.31 | 93.56 |
| ResNet-50 | 91.73 | 90.88 | 90.42 | 92.61 | 90.65 | 95.18 |
| ResNet-152 | 93.67 | 93.02 | 92.58 | 94.43 | 92.80 | 96.51 |
| EfficientNet-B0 | 93.19 | 92.48 | 92.67 | 94.02 | 92.57 | 96.13 |
| Swin Transformer | 94.36 | 93.71 | 93.95 | 94.88 | 93.83 | 96.75 |
| ConvNeXt | 95.12 | 94.57 | 94.82 | 95.41 | 94.69 | 97.28 |
| DEQ-KAN | 93.66 | — | — | — | — | — |
| **AttFKAN-DEQ (ours)** | **96.60** | **96.12** | **96.28** | **96.85** | **96.20** | **98.44** |

### IDC Dataset (50×50 patches, 5-fold CV)

| Model | Accuracy | Precision | Recall | Specificity | F1-Score | AUC |
|-------|----------|-----------|--------|-------------|----------|-----|
| ResNet-152 | 92.71 | 92.14 | 91.78 | 93.52 | 91.96 | 95.67 |
| ConvNeXt | 94.12 | 93.68 | 93.51 | 94.62 | 93.59 | 96.89 |
| **AttFKAN-DEQ (ours)** | **94.52** | **94.18** | **94.37** | **94.68** | **94.27** | **97.31** |

### Model Efficiency (NVIDIA V100, 224×224, batch=1)

| Model | Parameters | GFLOPs | Inference |
|-------|-----------|--------|-----------|
| ViT-B/16 | 86.0 M | 17.60 | 45.3 ms |
| ResNet-50 | 25.6 M | 4.10 | 12.4 ms |
| EfficientNet-B0 | 5.3 M | 0.39 | 4.8 ms |
| DEQ-KAN | 5.0 M | 0.50 | 7.1 ms |
| **AttFKAN-DEQ** | **6.0 M** | **0.60** | **8.3 ms** |

## Citation

If you use this code or build upon our work, please cite:

```bibtex
@article{ali2025attfkandeq,
  title   = {{AttFKAN-DEQ}: Attention-Enhanced Fourier Kolmogorov--Arnold Networks
             Using Deep Equilibrium for Breast Cancer Histopathology Classification},
  author  = {Ali, Hassan and Khan, Muhammad Asghar and Barb, Gordana},
  journal = {IEEE Access},
  year    = {2025},
  note    = {Under review}
}
```

---

## Acknowledgements

- DEQ framework: [Bai et al., NeurIPS 2019](https://arxiv.org/abs/1909.01377)
- KAN: [Liu et al., arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
- FourierKAN: [Xu et al., arXiv:2406.01034](https://arxiv.org/abs/2406.01034)
- CBAM: [Woo et al., ECCV 2018](https://arxiv.org/abs/1807.06521)
- BreakHis dataset: [Spanhol et al., IEEE TBME 2015](https://doi.org/10.1109/TBME.2015.2496264)
- IDC dataset: [Cruz-Roa et al., SPIE 2014](https://doi.org/10.1117/12.2043872)

---

## License

This project is released under the MIT License.
