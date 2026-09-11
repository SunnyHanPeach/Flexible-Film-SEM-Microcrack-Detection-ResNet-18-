```markdown
# Quantitative Characterization of Stress Microcracks in Flexible Thin Films via ResNet-18 and Continuous Spatial Probability Modeling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Author:** Sun Han (Tongji University)  
> **Contact:** [GitHub Profile](https://github.com/SunnyHanPeach)  
> **Topic:** AI for Materials Science (AI4S) / Fatigue Failure Characterization of Flexible Optical Thin Films

---

## 1. Overview & Scientific Motivation

In the reliability assessment of flexible semiconductor devices (e.g., Quantum Dot LEDs, flexible perovskite thin films, and battery electrode collectors), cyclic mechanical bending induces progressive tensile stress concentration, eventually leading to sub-micron **hairline microcracks**.

Conventional characterization predominantly relies on qualitative visual inspection of Scanning Electron Microscopy (SEM) images, which suffers from:
1. **Subjective Sampling Bias**: Manual selection of local regions fails to reflect macroscopic fracture behavior.
2. **Lack of Continuous Quantification**: Standard bounding-box detection (e.g., YOLO) introduces severe feature dilution on filamentary cracks, while Class Activation Mapping (Grad-CAM) collapses spatially on downsampled feature layers.

To address these challenges, this repository provides a high-throughput, cross-disciplinary characterization framework. By combining high-density sliding spatial probing, residual feature extraction, and **separable 2D Gaussian spatial smoothing**, this system achieves continuous probability field mapping of mechanical stress damage while maintaining high statistical specificity on pristine control groups.

---

## 2. Theoretical & Methodological Framework

### 2.1 Mechanical & Morphological Correlation
This framework was benchmarked against the surface morphology phenomena reported in flexible optoelectronic research (e.g., ligand engineering on colloidal quantum dots):
* **High-Modulus Brittle Surface (e.g., Oleic Acid / OA Ligands)**: Weak intermolecular crosslinking and high modulus cause severe tensile stress concentration under dynamic bending, forming dense, interconnected microcrack networks.
* **Toughness-Enhanced Surface (e.g., DDTC Ligand Exchange)**: Dense chemical crosslinking significantly improves elongation at break, effectively suppressing crack initiation.

### 2.2 Dense Spatial Probing vs. Grad-CAM
Standard Grad-CAM interpolates coarse activation maps ($7 \times 7$ or $1 \times 1$ on downscaled bottleneck layers) back to input dimensions, leading to spatial resolution collapse and "blocky" artifacts. 

In this work, we implement a **dense sliding spatial probe** with small stride ($s = 8\text{ px}$), coupled with **2D separable Gaussian smoothing**:

$$G(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

Exploiting kernel separability reduces spatial convolution complexity from $\mathcal{O}(H \cdot W \cdot K^2)$ to $\mathcal{O}(H \cdot W \cdot 2K)$, yielding continuous, contour-like stress damage fields calibrated to an absolute probability scale $[0.0, 1.0]$.

---

## 3. Repository Structure

```text
├── config/
│   └── settings.yaml                      # Centralized hyperparameters & physical scales
├── data/
│   ├── train/                             # Sub-sampled balanced training patches (32x32)
│   └── val/                               # Stratified validation patches
├── models/
│   └── resnet18_crack_model.pth           # Fine-tuned ResNet-18 weights (Validation F1-checkpoint)
├── raw_large_images/                      # Full-scale uncropped SEM micrographs
│   ├── test_hairline_crack.png            # Bending fatigue sample (High crack density)
│   └── test_intact_clean.png              # Pristine negative control sample
├── results/                               # Output diagnostic visualisations and metric logs
│
├── preprocess_patches.py                  # Multiprocess spatial sliding window patch extractor
├── balance_dataset.py                     # Deterministic class-balancing downsampler (Seed: 42)
├── train_feature_extractor.py             # Transfer learning pipeline (Cosine Annealing, AdamW)
├── eval_blind_detection.py                # High-throughput batch streaming blind detection
└── generate_stress_heatmap.py             # XAI continuous Gaussian stress probability field generator

```

---

## 4. Benchmark & Experimental Validation

### 4.1 Quantitative Dual-Blind Evaluation

Evaluated across full-field SEM micrographs under identical inference thresholds ($\text{Confidence} \ge 0.80$):

| Specimen Group | Morphological Description | Probed Patches | Damage Area Ratio | Specificity / True Negative Rate | Physical Interpretation |
| --- | --- | --- | --- | --- | --- |
| **Bending Fatigued** | QDs-OA film after cyclic bending | 14,884 | **88.13%** | — | High tensile stress causes extensive interconnected fracture network |
| **Pristine Control** | QDs-DDTC ligand-exchanged film | 14,884 | **0.86%** | **99.14%** | Elastic stress dispersal; zero false alarms on background nanogranules |

*Note: Damage ratio reflects the fraction of spatial micro-domains ($32 \times 32\text{ px}$) affected by tensile strain relaxation fields, not total material mass loss.*

### 4.2 Validation Metrics During Training

* **Backbone:** ResNet-18 (Pre-trained on ImageNet-1K)
* **Optimization:** AdamW ($\text{LR} = 3\times 10^{-4}$, $\text{Weight Decay} = 10^{-4}$, Cosine Annealing)
* **Performance:**
* Validation Accuracy: **98.4%**
* Crack Detection Recall: **97.8%**
* Macro F1-Score: **98.1%**



---

## 5. Getting Started

### 5.1 Environment Setup

```bash
git clone [https://github.com/SunnyHanPeach/Flexible-SEM-Crack-Detection.git](https://github.com/SunnyHanPeach/Flexible-SEM-Crack-Detection.git)
cd Flexible-SEM-Crack-Detection

# Install standard scientific and deep learning dependencies
pip install torch torchvision opencv-python numpy

```

### 5.2 End-to-End Pipeline Execution

#### Step 1: Multiprocess Spatial Patch Extraction

Extract overlapping $32 \times 32$ regions of interest (ROI) from raw micrographs:

```bash
python preprocess_patches.py --source_dir ./raw_large_images --target_dir ./data/train --workers 4

```

#### Step 2: Class-Balancing Downsampling

Enforce strict class balance to prevent dominant background bias (deterministic seed = 42):

```bash
python balance_dataset.py --data_dir ./data/train --max_samples 1500 --seed 42

```

#### Step 3: Model Fine-Tuning & Validation Checkpoint

Train residual classifier with cosine learning rate decay and F1-score tracking:

```bash
python train_feature_extractor.py --epochs 20 --batch_size 32 --lr 3e-4

```

#### Step 4: Full-Field Blind Quantitative Testing

Stream batched patches through GPU buffer to localize damage and compute global damage ratios:

```bash
python eval_blind_detection.py --conf_thresh 0.80 --batch_size 64

```

#### Step 5: Continuous Gaussian Stress Probability Mapping

Generate calibrated continuous spatial stress distributions with integrated colorbar calibration:

```bash
python generate_stress_heatmap.py --stride 8 --kernel_size 31

```

---

## 6. Engineering Highlights & Algorithmic Optimizations

1. **Batched Inference Streaming**: Circumvents single-sample forward pass bottlenecks by queueing sliding coordinates into contiguous GPU tensor batches, achieving an inference speedup of $>15\times$.
2. **Separable Gaussian Convolution**: Implements spatial decomposition to reduce continuous smoothing complexity from quadratic $\mathcal{O}(K^2)$ to linear $\mathcal{O}(2K)$ per pixel.
3. **Absolute Scale Calibration**: Standardizes the colorimetric mapping to a strict $[0.0, 1.0]$ absolute scale, preventing pristine control groups from artificially showing false-positive thermal hotspots.

---

## 7. License & Citation

This project is released under the [MIT License](https://www.google.com/search?q=LICENSE).

```

```
