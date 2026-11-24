# Repository Structure and File Guide

## Bi-Temporal Ante-Hoc Siamese 3D Res-Axial Network (DLC-Net)

This document provides a comprehensive guide to the repository structure, detailing the architectural purpose, technical mechanisms, input/output contracts, and interdependencies of every file.

---

## 1. Directory Tree Overview

```text
Major Project/
├── ARGON_TASK_SPEC.md              # Formal technical specifications and architectural requirements
├── REPO_STRUCTURE.md               # Repository documentation and file guide (this document)
├── MODEL_ARCHITECTURE.md           # In-depth mathematical and architectural specification of DLC-Net
├── requirements.txt                # Python environment dependencies
├── default_config.yaml             # Root reference configuration
├── dataset.py                      # Root reference dataset module
│
├── configs/
│   └── default_config.yaml         # Primary YAML configuration file (hyperparameters, paths, loss weights)
│
├── data/
│   ├── __init__.py                 # Data package initializer exposing preprocessing and loader routines
│   ├── preprocess_mri.py           # Bias correction (N4ITK), skull-stripping, registration, resampling
│   └── dataset.py                  # Patient-stratified splitting, Zero-Leakage verification, PyTorch Dataset
│
├── models/
│   ├── __init__.py                 # Models package initializer exposing all DLC-Net submodules
│   ├── ante_hoc_gate.py            # Stage B: Bi-Temporal Ante-Hoc Saliency Gate (Factual & Counterfactual)
│   ├── axial_attention_3d.py       # Stage C1: Factorized 3D Axial Self-Attention (Depth, Height, Width)
│   ├── siamese_backbone.py         # Stage C2: Weight-Shared 3D Res-Axial Encoder f_θ(·)
│   ├── progression_head.py         # Stage D: Latent Severity Scorer and 258-D Differential Progression MLP Head
│   └── full_model.py               # End-to-end BiTemporalSiameseDLCNet orchestrating Stages B, C, and D
│
├── losses/
│   ├── __init__.py                 # Losses package initializer
│   └── composite_loss.py           # Stage E: Multi-objective loss (Cross-Entropy + Monotonic Ranking + L1 Sparsity)
│
├── evaluation/
│   ├── __init__.py                 # Evaluation package initializer
│   ├── metrics.py                  # Classification metrics (Acc, F1, Kappa, ROC-AUC) and Drop_cf evaluator
│   └── visualize_xai.py            # 3D NIfTI volume export and 3x3 multi-planar 2D slice overlay rendering
│
├── train.py                        # Mixed-Precision (AMP) training, validation loop, and checkpointing
├── evaluate.py                     # Standalone performance factor & XAI evaluation dashboard
└── test_pipeline.py                # 5-stage automated acceptance verification suite
```

---

## 2. Root-Level Files

### `ARGON_TASK_SPEC.md`
- **Purpose**: Ground-truth specification document detailing the research motivation, formal tensor shapes, loss equations, and acceptance criteria for DLC-Net.
- **Key Contents**:
  - Longitudinal ADNI T1-weighted MRI pair formulation.
  - Mathematical equations for Stage B Saliency Discovery, Stage C1 Axial Attention, Stage D Differential Vector Fusion, and Stage E Composite Loss.
  - Automated verification checklist implemented in `test_pipeline.py`.

### `requirements.txt`
- **Purpose**: Defines minimum version requirements for production and research environments.
- **Key Dependencies**:
  - `torch>=2.0.0`: Deep learning engine, automatic mixed precision (`torch.amp`), and gradient checkpointing.
  - `monai>=1.2.0`: Medical imaging domain algorithms.
  - `nibabel>=5.0.0`: Neuroimaging NIfTI (.nii/.nii.gz) I/O.
  - `SimpleITK>=2.2.0`: N4ITK bias correction and Mattes Mutual Information 3D registration.
  - `scikit-learn>=1.2.0`: Evaluation metrics (Macro-F1, Cohen's Kappa, OvR ROC-AUC).
  - `pyyaml>=6.0`: Configuration parsing.
  - `numpy`, `scipy`, `matplotlib`, `pytest`: Numerical operations and visualization.

---

## 3. Configuration Module (`configs/`)

### `configs/default_config.yaml`
- **Purpose**: Centralized source of truth for all experiment configurations, structured hierarchically:
  1. `project`: Project metadata, random seed (`42`), target device (`cuda` with automatic `cpu` fallback), AMP flag (`use_amp: true`), and gradient checkpointing flag (`use_checkpoint: true`).
  2. `data`: Directory paths, target isotropic dimensions `[128, 128, 128]`, split ratios (`train: 0.70`, `val: 0.15`, `test: 0.15`), batch size, and data augmentation settings.
  3. `preprocessing`: SimpleITK N4ITK iterations `[50, 50, 30, 20]`, registration metric bins (`50`), and skull-stripping morphological radii.
  4. `model`: Sub-network channel specifications across the Stem (`16`), Stage 1 (`32`), Stage 2 (`64`), Stage 3 (`128`), attention heads (`4`), and progression head hidden dimension (`64`).
  5. `loss`: Class balance weights `[1.0, 1.5, 2.0]`, ranking margin ($\epsilon=0.10$), stable tolerance ($\epsilon_{\text{stable}}=0.05$), and loss multipliers ($\lambda_1=0.5, \lambda_2=0.05$).
  6. `training`: Epoch count, AdamW learning rate (`1.0e-4`), cosine annealing schedule, gradient clipping norm (`1.0`), and checkpoint directory.
  7. `evaluation`: Counterfactual drop evaluation settings and slice visualization coordinates.

---

## 4. Data Processing Module (`data/`)

### `data/preprocess_mri.py`
- **Purpose**: Standalone, robust 3D structural MRI preprocessing pipeline converting raw NIfTI scans into standardized $128^3$ isotropic tensors in $[0, 1]$.
- **Core Functions**:
  - `load_mri_volume(source)`: Reads NIfTI (`.nii`, `.nii.gz`), NumPy (`.npy`), or memory arrays, returning float32 3D volumes, 4x4 affine matrices, and SimpleITK image representations.
  - `compute_brain_mask(volume, closing_radius)`: Computes automated foreground Otsu thresholding, extracts the largest 3D connected component, and applies morphological closing with hole filling.
  - `skull_strip_mri(volume, brain_mask)`: Eliminates non-brain skull/dura voxels.
  - `n4_bias_field_correction(volume, mask)`: Mitigates low-frequency B1 magnetic field inhomogeneity via SimpleITK's `N4BiasFieldCorrectionImageFilter` (with SciPy homomorphic Gaussian filter fallback).
  - `rigid_register_to_reference(moving, fixed)`: Performs 6-DOF 3D rigid registration (`Euler3DTransform` optimized with Mattes Mutual Information and regular step gradient descent) to align baseline $T_1$ to MNI152 and follow-up $T_2$ to $T_1$.
  - `resample_isotropic_volume(volume, target_shape, is_label)`: Performs trilinear interpolation for MRI scans and nearest-neighbor interpolation for binary masks.
  - `normalize_intensity(volume, brain_mask, p_min, p_max)`: Clips intensity outliers to the $[0.5, 99.5]$ foreground percentiles and scales intensities linearly into $[0.0, 1.0]$.
  - `preprocess_longitudinal_pair(...)`: Orchestrates the complete end-to-end paired pipeline.

### `data/dataset.py`
- **Purpose**: Handles metadata parsing, patient-level stratified splitting, paired 3D data augmentation, and PyTorch `DataLoader` instantiation.
- **Core Components**:
  - `ScanPairRecord`: Data class storing `ptid`, scan paths or in-memory arrays, inter-scan interval `delta_t` (in years), and clinical progression `label` $\in \{0, 1, 2\}$.
  - `verify_zero_data_leakage(train_ptids, val_ptids, test_ptids)`: **Strict zero-leakage validator** that asserts no patient ID appears in more than one split:
    $$\text{Train}_{\text{ptids}} \cap \text{Val}_{\text{ptids}} = \emptyset, \quad \text{Train}_{\text{ptids}} \cap \text{Test}_{\text{ptids}} = \emptyset, \quad \text{Val}_{\text{ptids}} \cap \text{Test}_{\text{ptids}} = \emptyset$$
  - `stratified_patient_split(records, ...)`: Groups scan pairs strictly by patient ID, determines each subject's maximum disease progression severity for stratification, and generates split subsets.
  - `LongitudinalMRIDataset`: PyTorch `Dataset` that loads $(X_1, X_2)$ pairs. When `augment=True`, it applies coupled 3D left-right sagittal flips and synchronized gamma/contrast variations to preserve true temporal progression.
  - `create_longitudinal_dataloaders(...)`: Factory function returning PyTorch `DataLoader` objects for `"train"`, `"val"`, and `"test"`.

---

## 5. Model Architecture Module (`models/`)

### `models/ante_hoc_gate.py` (Stage B)
- **Purpose**: Enforces causal explainability at the input resolution ($128^3$) *before* any downsampling or deep feature extraction.
- **Classes**:
  - `BiTemporalSaliencyGate`:
    - Inputs: Baseline $X_1$ and follow-up $X_2$ ($\mathbb{R}^{B \times 1 \times 128 \times 128 \times 128}$).
    - Calculates absolute difference: $D_{12} = |X_2 - X_1|$.
    - Concatenates along channels: $U_t = [X_t \,\|\, D_{12}] \in \mathbb{R}^{B \times 2 \times 128 \times 128 \times 128}$.
    - Passes $U_t$ through shared discovery layers (`Conv3d(2, 8, 3)` $\to$ `InstanceNorm3d(8)` $\to$ `ReLU` $\to$ `Conv3d(8, 1, 1)` $\to$ `Sigmoid`) to produce masks $M_1, M_2 \in (0, 1)$.
    - Produces factual gated volumes: $X_{\text{gated}, t} = X_t \odot M_t$.
    - Produces counterfactual volumes: $X_{\text{cf}, t} = X_t \odot (1 - M_t)$.
    - Produces longitudinal atrophy map: $\Delta M = |M_2 - M_1|$.

### `models/axial_attention_3d.py` (Stage C1)
- **Purpose**: Factorized 3D Multi-Head Self-Attention (MHSA) that captures global spatial context along Depth, Height, and Width axes without the quadratic memory explosion of full 3D attention.
- **Classes**:
  - `AxialAttention3D`:
    - Decomposes 3D self-attention sequentially: $\text{Depth }(D) \to \text{Height }(H) \to \text{Width }(W)$.
    - Incorporates learnable 1D positional encodings (`pos_embed_d`, `pos_embed_h`, `pos_embed_w`).
    - Employs LayerNorm and residual connections across each axis.
    - Integrates `torch.utils.checkpoint` to recompute activations during backpropagation, enabling 3D pairs to fit into GPU VRAM.

### `models/siamese_backbone.py` (Stage C2)
- **Purpose**: Weight-shared 3D Res-Axial feature extractor mapping 3D volumes into 128-dimensional latent representations.
- **Classes**:
  - `ResBlock3D`: Two $3\times 3\times 3$ convolutions with `BatchNorm3d`, `LeakyReLU(0.01)`, and a $1\times 1\times 1$ projection shortcut for channel/stride changes.
  - `ResAxial3DEncoder`: Hierarchical feature extractor:
    - **Stem**: `Conv3d(1, 16, stride=2)` $\to (B, 16, 64^3)$
    - **Stage 1**: `ResBlock3D(16, 32)` + `MaxPool3d(2)` $\to (B, 32, 32^3)$
    - **Stage 2**: `ResBlock3D(32, 64)` + `AxialAttention3D(64, spatial_dim=32)` + `MaxPool3d(2)` $\to (B, 64, 16^3)$
    - **Stage 3**: `ResBlock3D(64, 128)` + `AxialAttention3D(128, spatial_dim=16)` + `MaxPool3d(2)` $\to (B, 128, 8^3)$
    - **Global Pool**: `AdaptiveAvgPool3d((1, 1, 1))` + `Flatten()` $\to \mathbf{z}_t \in \mathbb{R}^{B \times 128}$
  - `Siamese3DBackbone`: Maintains a single shared `ResAxial3DEncoder` instance that processes both $X_{\text{gated}, 1}$ and $X_{\text{gated}, 2}$.

### `models/progression_head.py` (Stage D)
- **Purpose**: Models longitudinal progression by estimating latent severity, computing progression velocity, and classifying disease trajectory.
- **Classes**:
  - `TimeAwareProgressionHead`:
    - **Shared Latent Severity Scorer**: $s_t = \sigma(\text{Linear}_{128 \to 1}(\mathbf{z}_t)) \in [0, 1]$.
    - **Progression Delta**: $\Delta = s_2 - s_1 \in [-1, 1]$.
    - **Progression Velocity**: $\Delta s = \frac{s_2 - s_1}{\max(\Delta t, 10^{-3})} \in \mathbb{R}^{B \times 1}$.
    - **258-D Differential Fusion Vector**:
      $$\mathbf{h} = [\Delta s \,\|\, \Delta \,\|\, (\mathbf{z}_2 - \mathbf{z}_1) \,\|\, |\mathbf{z}_2 - \mathbf{z}_1|] \in \mathbb{R}^{B \times 258}$$
    - **Classifier MLP**: `Linear(258, 64)` $\to$ `ReLU` $\to$ `Dropout(0.4)` $\to$ `Linear(64, 3)` $\to$ logits $\in \mathbb{R}^{B \times 3}$.

### `models/full_model.py`
- **Purpose**: Top-level wrapper integrating Stages B, C, and D.
- **Classes**:
  - `BiTemporalSiameseDLCNet` (alias `DLCNet`):
    - `forward(X_1, X_2, delta_t)`: Runs the complete factual forward pass.
    - `forward_counterfactual(X_1, X_2, delta_t)`: Runs dual forward passes ($X_{\text{gated}}$ and $X_{\text{cf}}$) to compute counterfactual probability drops.
    - `from_config(config)`: Factory constructor from parsed YAML dictionary.

---

## 6. Losses and Evaluation Modules

### `losses/composite_loss.py` (Stage E)
- **Purpose**: Defines the multi-objective training loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_1 \mathcal{L}_{\text{ranking}} + \lambda_2 \mathcal{L}_{\text{sparsity}}$$
- **Components**:
  - `compute_monotonic_ranking_loss(s_1, s_2, targets, epsilon, epsilon_stable)`:
    - For progressing patients ($y > 0$): $\max(0, s_1 - s_2 + \epsilon)$.
    - For stable patients ($y = 0$): $\max(0, |s_2 - s_1| - \epsilon_{\text{stable}})$.
  - `compute_mask_sparsity_loss(M_1, M_2)`: Mean $L_1$ norm of saliency masks across all voxels.
  - `CompositeDLCNetLoss`: Module that encapsulates class-weighted cross-entropy, ranking loss, and sparsity loss.

### `evaluation/metrics.py`
- **Purpose**: Quantitative evaluation and causal counterfactual validation.
- **Functions**:
  - `compute_accuracy`, `compute_macro_f1`, `compute_cohens_kappa`, `compute_multiclass_roc_auc`: Standard multi-class classification metrics.
  - `compute_counterfactual_drop(probs_factual, probs_cf, targets, progression_classes)`:
    $$\text{Drop}_{\text{cf}} = P(Y = y \mid X_{\text{gated}}) - P(Y = y \mid X_{\text{cf}})$$
    Measures the drop in confidence when causal atrophy regions are masked out.
  - `evaluate_counterfactual_batch`: Runs factual and counterfactual passes under `torch.no_grad()`.

### `evaluation/visualize_xai.py`
- **Purpose**: Explainable AI visualization in 2D and 3D.
- **Functions**:
  - `export_xai_nifti_volumes`: Exports $X_1$, $X_2$, $M_1$, $M_2$, and $\Delta M = |M_2 - M_1|$ as 3D `.nii.gz` volumes.
  - `plot_longitudinal_xai_overlays`: Produces a $3\times 3$ publication-grade figure displaying Axial, Coronal, and Sagittal cross-sections:
    - Row 1: Baseline scan $X_1$ overlaid with Ante-Hoc mask $M_1$
    - Row 2: Follow-up scan $X_2$ overlaid with Ante-Hoc mask $M_2$
    - Row 3: Average anatomy overlaid with Longitudinal Atrophy Map $\Delta M$

---

## 7. Execution and Verification Scripts

### `train.py`
- **Purpose**: Production training pipeline supporting mixed precision and checkpointing.
- **Features**:
  - Deterministic random seeding across Python, NumPy, and PyTorch.
  - Automatic Mixed Precision (`torch.amp.autocast` with `GradScaler`).
  - Cosine annealing learning rate scheduling and gradient norm clipping.
  - Periodic validation computing classification metrics and $\text{Drop}_{\text{cf}}$.
  - Saves `best_model.pt` (based on Macro-F1) and `last_model.pt`.
  - CLI support: `--config`, `--epochs`, and `--synthetic-demo`.

### `evaluate.py`
- **Purpose**: Standalone CLI evaluation tool and diagnostic dashboard for computing all model performance factors and explainability metrics.
- **Features**:
  - Computes Accuracy, Macro-F1, Cohen's Kappa ($\kappa$), and Macro OvR ROC-AUC.
  - Computes Causal Counterfactual Confidence Drop ($\text{Drop}_{\text{cf}}$).
  - Generates full confusion matrix and per-class precision / recall / specificity / support.
  - Computes latent disease progression dynamics ($s_1, s_2, \Delta, \Delta s$).
  - Exports 3D NIfTI volumes and 2D multi-planar (Axial, Coronal, Sagittal) overlay images with `--visualize`.
  - Supports testing on `--checkpoint`, `--split test|val|train`, and `--synthetic-demo`.

### `test_pipeline.py`
- **Purpose**: Acceptance verification suite validating all mathematical and architectural constraints.
- **Verification Tests**:
  1. `test_tensor_shapes_full_resolution`: Validates shapes at $128^3$ resolution across all stages.
  2. `test_siamese_symmetry`: Asserts that when $X_1 = X_2$, $\mathbf{z}_1 = \mathbf{z}_2$, $s_1 = s_2$, $\Delta = 0$, $\Delta s = 0$, and $\Delta M = 0$.
  3. `test_gradient_flow_and_composite_loss`: Confirms non-zero, finite gradients across all submodules after one backward pass.
  4. `test_counterfactual_forward_and_drop_metric`: Validates `forward_counterfactual` execution and probability normalization.
  5. `test_patient_stratified_split_and_dataset_signature`: Verifies zero patient leakage and dataset return shapes.

---

## 8. Common Workflows

### Running Verification Tests
```bash
python test_pipeline.py
```

### Running Complete Evaluation & Metrics Dashboard
```bash
# Evaluate a trained checkpoint on test split with XAI overlay export
python evaluate.py --checkpoint checkpoints/best_model.pt --visualize

# Evaluate demo/synthetic split instantly without needing pre-existing weights
python evaluate.py --synthetic-demo --visualize
```

### Running Training Smoke Test (Synthetic Data)
```bash
python train.py --synthetic-demo --epochs 2
```

### Running Full Model Training
```bash
python train.py --config configs/default_config.yaml
```
