# DLC-Net: Bi-Temporal Ante-Hoc Siamese 3D Res-Axial Network for Longitudinal Alzheimer's Progression

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DLC-Net is a deep learning medical imaging framework designed for longitudinal Alzheimer's Disease (AD) progression modeling using paired 3D T1-weighted brain MRI scans ($X_1, X_2$) across an inter-scan interval ($\Delta t \approx 1.0\text{ year}$). 

Rather than relying on unverified post-hoc saliency maps (e.g., Grad-CAM, SHAP), DLC-Net integrates an **Ante-Hoc Self-Explaining Saliency Gate** directly into the forward pass, coupled with a **Factorized 3D Axial Self-Attention Siamese Backbone** and a **Time-Aware Progression Velocity Head**.

---

## Architecture Overview

```
Baseline (X1) ──┐
                ├──► [Stage B: Ante-Hoc Saliency Gate] ──► M1, M2, ΔM = |M2 - M1|
Follow-up (X2) ─┤                                         │
                │                                         ▼
                │                               Factual Gating: X̃1 = X1 ⊙ M1, X̃2 = X2 ⊙ M2
                │                                         │
                │                                         ▼
                └──────────────────────────────► [Stage C: Siamese 3D Res-Axial Backbone]
                                                 (Shared Weights + Factorized 3D Axial Attention)
                                                          │
                                                          ▼
                                                 Latent Embeddings (z1, z2: 128-D)
                                                          │
                                                          ▼
                                                 [Stage D: Time-Aware Progression Head]
                                                 (Latent Severities s1, s2; Velocity Δs = Δ/Δt; 258-D Fusion)
                                                          │
                                                          ▼
                                                 Predictions: [0: Stable, 1: Progressing, 2: Advanced]
```

### Core Innovations
1. **Ante-Hoc Saliency Gating (Stage B):** Generates spatial masks $M_1, M_2 \in [0, 1]$ and longitudinal differential atrophy map $\Delta M = |M_2 - M_1|$. Downstream feature extractors *only* process gated factual representations.
2. **Factorized 3D Axial Attention (Stage C):** Computes self-attention independently along Depth, Height, and Width axes. Reduces computational complexity from $\mathcal{O}((DHW)^2)$ to $\mathcal{O}(DHW(D + H + W))$, allowing high-resolution $128^3$ 3D transformer attention within 4.0 GB VRAM.
3. **Time-Aware Differential Fusion (Stage D):** Normalizes progression velocity by elapsed visit time ($\Delta_s = (s_2 - s_1) / \Delta t$) and fuses scalar disease severities with 128-D differential embeddings.
4. **Counterfactual Verification:** Empirically verifies causal reliance by measuring confidence drop under mask inversion ($\text{Drop}_{\text{cf}} = f(X \odot M) - f(X \odot (1 - M))$).

---

## Cohort Profile (ADNI)

Evaluated on 172 subjects (344 volumes) from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) under strict subject-level stratified splitting:

| Trajectory Class | Description | Train (70.9%) | Val (14.5%) | Test (14.5%) | Total |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Class 0: Stable** | Cognitively Normal / Stable MCI | 78 | 16 | 16 | **110** (64.0%) |
| **Class 1: Progressing** | Progressive MCI ($\Delta\text{CDR} > 0$) | 19 | 4 | 4 | **27** (15.7%) |
| **Class 2: Advanced** | Alzheimer's Disease (AD) | 25 | 5 | 5 | **35** (20.3%) |
| **Total Subject Pairs** | Zero-leakage disjoint sets | **122** | **25** | **25** | **172** |

---

## Experimental Benchmarks & Results

| Training Session | Epochs | Class Weights | Regularizations | Val Acc | Val Macro-F1 | Class 1 F1 |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **Session 01** | 15 | `[1.0, 1.0, 1.0]` | Baseline GPU implementation | 64.00% | 38.46% | 0.00% |
| **Session 02** | 50 | `[1.0, 3.5, 2.0]` | Class weighting | 72.00% | 50.29% | 33.33% |
| **Session 03 (Peak)** | **28** | `[1.0, 2.8, 1.8]` | **Coupled Jitter, Label Smooth (0.05), Axial Dropout (0.10)** | **76.00%** | **61.40%** | **66.67%** |
| **Session 04** | 67 | `[1.0, 2.8, 1.8]` | Early stopping validation (sweet spot confirmed) | 72.00% | 50.29% | 33.33% |

* **Held-out Test Performance:** 60.00% Accuracy, 93.75% Stable Recall, 76.92% Stable F1.
* **Causal Faithfulness:** $\text{Drop}_{\text{cf}} = +0.0636$ (High Causal Reliance).
* **Convergence Sweet Spot:** Codified at Epochs 25–30.

---

## Repository Structure

```
├── configs/
│   └── default_config.yaml         # Training, architecture, and loss hyperparameters
├── data/
│   ├── dataset.py                  # PyTorch Dataset with coupled 3D spatial jitter & augmentations
│   ├── prepare_adni_dataset.py     # End-to-end DICOM parsing, clinical delta extraction, and manifest builder
│   └── preprocess_mri.py           # Brain skull-stripping, intensity clipping, and isotropic resampling
├── evaluation/
│   └── metrics.py                  # Macro-F1, Cohen's Kappa, ROC-AUC, and counterfactual drop metrics
├── losses/
│   └── composite_loss.py           # Class-weighted CE + Monotonic Ranking + L1 Mask Sparsity
├── models/
│   ├── ante_hoc_gate.py            # Ante-Hoc Saliency Gate (Stage B)
│   ├── axial_attention_3d.py       # Factorized 3D Axial Self-Attention (Stage C1)
│   ├── siamese_backbone.py         # 3D Siamese Residual Backbone (Stage C2)
│   └── progression_head.py         # Time-Aware Differential Velocity Fusion Head (Stage D)
├── Plans/
│   ├── MODEL_ARCHITECTURE.md       # Detailed mathematical specification of DLC-Net
│   └── REPO_STRUCTURE.md           # Engineering guidelines and architectural contracts
├── progress/
│   ├── README.md                   # Training sessions master log
│   ├── generate_report.py          # Automated publication-grade PDF report generator
│   └── *.pdf                       # Generated PDF progress audit reports (Sessions 01-04)
├── visualizations/                 # Ante-hoc multi-planar saliency overlay figures
├── evaluate.py                     # Checkpoint evaluation, confusion matrix & XAI generation
├── requirements.txt                # Python package dependencies
├── test_pipeline.py                # Unit and integration test suite
└── train.py                        # Main distributed/local AMP training pipeline
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Data Preprocessing
Preprocess raw ADNI DICOM series into $(128, 128, 128)$ isotropic arrays:
```bash
python data/prepare_adni_dataset.py --dicom_dir /path/to/ADNI --output_dir data/preprocessed_128
```

### 2. Training
Run training using the optimal 30-epoch configuration:
```bash
python train.py --config configs/default_config.yaml --epochs 30 --patience 10
```

### 3. Evaluation & Explainability
Evaluate a trained checkpoint and generate multi-planar ante-hoc saliency overlays:
```bash
python evaluate.py --checkpoint checkpoints/best_model.pt --split test --visualize
```

---

## Citation & References
If you use this codebase or methodology in your research, please cite:
- Alzheimer's Disease Neuroimaging Initiative (ADNI): [adni.loni.usc.edu](https://adni.loni.usc.edu/)
- Zhang et al., *"An explainable 3D residual self-attention deep neural network for joint atrophy localization and Alzheimer’s disease diagnosis"*, IEEE JBHI, 2021.
- Ho et al., *"Axial Attention in Multidimensional Transformers"*, arXiv:1912.12180, 2019.

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.
