# Model Architecture Specification: Bi-Temporal Ante-Hoc Siamese 3D Res-Axial Network (DLC-Net)

## Theoretical Foundations, Mathematical Formulations, and Engineering Design

This document details the architectural design, mathematical formalisms, and computational flow of the **Bi-Temporal Ante-Hoc Siamese 3D Res-Axial Network (DLC-Net)** for longitudinal Alzheimer's disease (AD) and mild cognitive impairment (MCI) progression modeling.

---

## 1. Motivation and Problem Formulation

### 1.1 Limitations of Conventional Approaches
1. **Black-Box 3D CNNs**: Standard 3D CNNs and 3D CNN–LSTMs process volumetric MRI data as generic 3D tensors, obscuring which anatomical sub-regions (e.g., hippocampus, entorhinal cortex, ventricles) drive predictions.
2. **Post-Hoc Saliency Limitations (Grad-CAM, Saliency Maps)**: Conventional XAI methods compute explanations *after* decisions are made via gradient backpropagation. In medical neuroimaging, post-hoc methods often suffer from:
   - Gradient saturation and sensitivity to high-frequency noise.
   - Misleading attribution where gradients highlight non-causal brain boundaries or scan artifacts.
   - Lack of counterfactual verifiability.
3. **Absence of Time Normalization**: Many longitudinal models concatenate features from two timepoints without accounting for varying inter-scan intervals ($\Delta t$), failing to distinguish between rapid neurodegeneration over 6 months versus slow aging over 3 years.

### 1.2 The DLC-Net Paradigm
DLC-Net introduces three core innovations:
1. **Ante-Hoc Causal Gating**: Saliency discovery occurs **before** deep representation learning at full $128^3$ resolution. Only causally attributed voxels are passed to the deep backbone.
2. **Factorized 3D Axial Attention**: Overcomes the $\mathcal{O}(V^2)$ computational complexity of 3D self-attention by decomposing attention sequentially along Depth ($D$), Height ($H$), and Width ($W$).
3. **Time-Aware Differential Fusion**: Computes latent disease severity scores and time-normalized progression velocity ($\Delta s = \Delta / \Delta t$) combined with differential embedding vectors.

---

## 2. End-to-End Computational Pipeline

```
Raw Scans (T1, T2) -> Preprocessing (N4ITK, Rigid Reg, 128³) -> (X_1, X_2)
                                                                 │
┌────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────┐
│ STAGE B: Bi-Temporal Ante-Hoc Saliency Gate (ante_hoc_gate.py)                                                          │
│                                                                                                                         │
│   D_12 = |X_2 - X_1|                                                                                                    │
│   U_1 = [X_1 || D_12],  U_2 = [X_2 || D_12]                                                                             │
│   Shared Discovery Net: Conv3D(2->8) -> IN3D -> ReLU -> Conv3D(8->1) -> Sigmoid                                         │
│   Saliency Masks: M_1, M_2 ∈ (0, 1)^{B × 1 × 128 × 128 × 128}                                                           │
│   Atrophy Map: ΔM = |M_2 - M_1|                                                                                         │
│                                                                                                                         │
│   Factual Branch:          X_gated,1 = X_1 ⊙ M_1,        X_gated,2 = X_2 ⊙ M_2                                          │
│   Counterfactual Branch:   X_cf,1 = X_1 ⊙ (1 - M_1),     X_cf,2 = X_2 ⊙ (1 - M_2)                                       │
└───────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────┘
                                                │ (Factual Volumes)
┌───────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────┐
│ STAGES C1 & C2: Weight-Shared Siamese 3D Res-Axial Encoder f_θ(·) (siamese_backbone.py, axial_attention_3d.py)          │
│                                                                                                                         │
│   X_gated,1 ──>  [ Shared 3D Res-Axial Encoder f_θ ]  ──>  z_1 ∈ ℝ^{B × 128}                                            │
│   X_gated,2 ──>  [ Shared 3D Res-Axial Encoder f_θ ]  ──>  z_2 ∈ ℝ^{B × 128}                                            │
│                                                                                                                         │
│   Internal Encoder Architecture:                                                                                        │
│     Stem: Conv3D(1->16, k=3, s=2, p=1) + BN + LeakyReLU(0.01)               ──> (B, 16, 64³)                            │
│     Stage 1: ResBlock3D(16->32) + MaxPool3D(2)                              ──> (B, 32, 32³)                            │
│     Stage 2: ResBlock3D(32->64) + AxialAttention3D(64, dim=32) + MaxPool3D ──> (B, 64, 16³)                            │
│     Stage 3: ResBlock3D(64->128) + AxialAttention3D(128, dim=16) + MaxPool3D──> (B, 128, 8³)                           │
│     Global Pool: AdaptiveAvgPool3D((1, 1, 1)) + Flatten                     ──> (B, 128)                                │
└───────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────┘
                                                │ z_1, z_2, and Δt
┌───────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────┐
│ STAGE D: Time-Aware Differential Progression Head (progression_head.py)                                                 │
│                                                                                                                         │
│   1. Severity Scoring:       s_1 = σ(W_s z_1 + b_s) ∈ [0, 1],   s_2 = σ(W_s z_2 + b_s) ∈ [0, 1]                         │
│   2. Progression Delta:      Δ = s_2 - s_1 ∈ [-1, 1]                                                                    │
│   3. Progression Velocity:   Δs = (s_2 - s_1) / max(Δt, 10⁻³) ∈ ℝ^{B × 1}                                               │
│   4. Differential Fusion:    h = [Δs || Δ || (z_2 - z_1) || |z_2 - z_1|] ∈ ℝ^{B × 258}                                  │
│   5. Progression Classifier: Linear(258, 64) -> ReLU -> Dropout(0.4) -> Linear(64, 3) -> Logits ∈ ℝ^{B × 3}             │
└───────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────┘
                                                │ Logits, s_1, s_2, M_1, M_2
┌───────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────┐
│ STAGE E: Composite Multi-Objective Loss (composite_loss.py)                                                             │
│                                                                                                                         │
│   L_total = L_CE(y, y_hat) + λ_1 · L_ranking(s_1, s_2, y) + λ_2 · L_sparsity(M_1, M_2)                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Specifications by Module

### 3.1 Stage B: Bi-Temporal Ante-Hoc Saliency Gate (`models/ante_hoc_gate.py`)

#### Inputs
- Baseline scan: $X_1 \in \mathbb{R}^{B \times 1 \times 128 \times 128 \times 128}$
- Follow-up scan: $X_2 \in \mathbb{R}^{B \times 1 \times 128 \times 128 \times 128}$

#### Formulation
1. **Structural Difference Tensor**:
   $$D_{12} = |X_2 - X_1| \in \mathbb{R}^{B \times 1 \times 128 \times 128 \times 128}$$
   Captures localized macroscopic structural variations (e.g., ventricular enlargement, cortical thinning, hippocampal shrinkage) without assuming a specific progression direction.

2. **Channel Concatenation**:
   $$U_1 = [X_1 \,\|\, D_{12}] \in \mathbb{R}^{B \times 2 \times 128 \times 128 \times 128}$$
   $$U_2 = [X_2 \,\|\, D_{12}] \in \mathbb{R}^{B \times 2 \times 128 \times 128 \times 128}$$
   Each timepoint's anatomy is conditioned on the observed structural divergence between scans.

3. **Discovery Sub-Network**:
   A shared, lightweight 3D convolutional operator $g_\phi(\cdot)$ transforms $U_t$ into continuous saliency masks:
   $$H_t = \text{ReLU}\left(\text{InstanceNorm3D}\left(\text{Conv3D}_{2 \to 8, k=3}(U_t)\right)\right)$$
   $$M_t = \sigma\left(\text{Conv3D}_{8 \to 1, k=1}(H_t)\right) \in (0, 1)^{B \times 1 \times 128 \times 128 \times 128}$$
   *Instance Normalization* ensures stable mask generation across varying patient intensity profiles.

4. **Factual and Counterfactual Decomposition**:
   - **Factual Volume**: $X_{\text{gated}, t} = X_t \odot M_t$ (passes only disease-relevant anatomical features).
   - **Counterfactual Volume**: $X_{\text{cf}, t} = X_t \odot (1 - M_t)$ (inverts attribution, masking out identified causal regions).
   - **Conservation Property**: $X_{\text{gated}, t} + X_{\text{cf}, t} \equiv X_t$.
   - **Longitudinal Atrophy Map**: $\Delta M = |M_2 - M_1|$ (quantifies spatial expansion or contraction of salient regions across time).

---

### 3.2 Stage C1: Factorized 3D Axial Self-Attention (`models/axial_attention_3d.py`)

#### Computational Rationale
For a 3D feature volume $F \in \mathbb{R}^{B \times C \times D \times H \times W}$ with $V = D \cdot H \cdot W$:
- Full 3D Self-Attention computes an attention matrix of size $V \times V$:
  $$\mathcal{O}(V^2) = \mathcal{O}(D^2 H^2 W^2)$$
  For an intermediate feature map of size $32^3 = 32,768$ voxels, full attention requires $32,768^2 \approx 1.07 \times 10^9$ elements per attention head, exceeding standard GPU memory.
- **Factorized Axial Attention** decomposes this operation into three consecutive 1D Multi-Head Self-Attentions along the Depth, Height, and Width axes:
  $$\mathcal{O}(V(D + H + W)) = \mathcal{O}(D H W (D + H + W))$$
  For $32^3$, this requires only $32,768 \times (32 + 32 + 32) \approx 3.14 \times 10^6$ operations—a **340-fold reduction** in complexity.

#### Axis-Wise Formulation
Let $F \in \mathbb{R}^{B \times C \times D \times H \times W}$:

1. **Depth Axis ($D$)**:
   - Reshape & Permute: $F \to (B \cdot H \cdot W, D, C)$.
   - Add learnable 1D position encoding: $\mathbf{P}_D \in \mathbb{R}^{1 \times D \times C}$.
   - Apply LayerNorm and MHSA:
     $$A_D = \text{MHSA}(\text{LayerNorm}(F_D + \mathbf{P}_D))$$
   - Reshape back and add residual:
     $$F' = F + \text{Reshape}(A_D)$$

2. **Height Axis ($H$)**:
   - Reshape & Permute: $F' \to (B \cdot D \cdot W, H, C)$.
   - Add learnable 1D position encoding: $\mathbf{P}_H \in \mathbb{R}^{1 \times H \times C}$.
   - Apply LayerNorm and MHSA:
     $$A_H = \text{MHSA}(\text{LayerNorm}(F'_H + \mathbf{P}_H))$$
   - Reshape back and add residual:
     $$F'' = F' + \text{Reshape}(A_H)$$

3. **Width Axis ($W$)**:
   - Reshape & Permute: $F'' \to (B \cdot D \cdot H, W, C)$.
   - Add learnable 1D position encoding: $\mathbf{P}_W \in \mathbb{R}^{1 \times W \times C}$.
   - Apply LayerNorm and MHSA:
     $$A_W = \text{MHSA}(\text{LayerNorm}(F''_W + \mathbf{P}_W))$$
   - Reshape back and add residual:
     $$F_{\text{out}} = F'' + \text{Reshape}(A_W)$$

#### Memory Optimization
`AxialAttention3D` integrates `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`. During training, intermediate attention activations are discarded during the forward pass and recomputed during backpropagation, keeping peak GPU memory usage bounded.

---

### 3.3 Stage C2: Weight-Shared Siamese 3D Backbone (`models/siamese_backbone.py`)

A single encoder instance $f_\theta(\cdot)$ extracts features from both $X_{\text{gated}, 1}$ and $X_{\text{gated}, 2}$.

#### Sub-Module Structure

1. **Stem**:
   - `Conv3D(1, 16, kernel_size=3, stride=2, padding=1, bias=False)`
   - `BatchNorm3D(16)`
   - `LeakyReLU(negative_slope=0.01)`
   - Output: $(B, 16, 64, 64, 64)$

2. **Stage 1 (Local Feature Extraction)**:
   - `ResBlock3D(16, 32)`: Two $3\times 3\times 3$ convs with `BatchNorm3D` and `LeakyReLU`, with a $1\times 1\times 1$ conv projection shortcut.
   - `MaxPool3D(kernel_size=2, stride=2)`
   - Output: $(B, 32, 32, 32, 32)$

3. **Stage 2 (Mid-Level Context & Axial Attention)**:
   - `ResBlock3D(32, 64)`
   - `AxialAttention3D(channels=64, spatial_dim=32, num_heads=4)`
   - `MaxPool3D(kernel_size=2, stride=2)`
   - Output: $(B, 64, 16, 16, 16)$

4. **Stage 3 (High-Level Context & Axial Attention)**:
   - `ResBlock3D(64, 128)`
   - `AxialAttention3D(channels=128, spatial_dim=16, num_heads=4)`
   - `MaxPool3D(kernel_size=2, stride=2)`
   - Output: $(B, 128, 8, 8, 8)$

5. **Global Pooling & Latent Projection**:
   - `AdaptiveAvgPool3D((1, 1, 1))` collapses spatial dimensions.
   - `Flatten(start_dim=1)` yields:
     $$\mathbf{z}_1 = f_\theta(X_{\text{gated}, 1}) \in \mathbb{R}^{B \times 128}$$
     $$\mathbf{z}_2 = f_\theta(X_{\text{gated}, 2}) \in \mathbb{R}^{B \times 128}$$

#### Symmetry Invariance
Because $f_\theta$ is shared across timepoints, the network guarantees strict mathematical symmetry:
$$X_1 = X_2 \implies \mathbf{z}_1 = \mathbf{z}_2$$

---

### 3.4 Stage D: Time-Aware Differential Progression Head (`models/progression_head.py`)

#### Operations
Given $\mathbf{z}_1, \mathbf{z}_2 \in \mathbb{R}^{B \times 128}$ and $\Delta t \in \mathbb{R}^{B \times 1}$:

1. **Latent Severity Scorer**:
   A shared linear projection followed by a sigmoid activation assigns a continuous disease severity score $s_t \in [0, 1]$ to each timepoint:
   $$s_1 = \sigma(\mathbf{w}_s^\top \mathbf{z}_1 + b_s) \in [0, 1]$$
   $$s_2 = \sigma(\mathbf{w}_s^\top \mathbf{z}_2 + b_s) \in [0, 1]$$

2. **Scalar Progression Delta**:
   $$\Delta = s_2 - s_1 \in [-1, 1]$$

3. **Time-Normalized Progression Velocity**:
   $$\Delta s = \frac{s_2 - s_1}{\max(\Delta t, 10^{-3})} \in \mathbb{R}^{B \times 1}$$
   - The numerical stabilizer $10^{-3}$ prevents division by zero if scans occur concurrently.
   - Directly encodes the *rate* of neurodegeneration per year.

4. **258-Dimensional Differential Fusion**:
   Combines temporal velocity, scalar delta, raw feature difference, and absolute feature difference:
   $$\mathbf{h} = \left[ \Delta s \;\|\; \Delta \;\|\; (\mathbf{z}_2 - \mathbf{z}_1) \;\|\; |\mathbf{z}_2 - \mathbf{z}_1| \right] \in \mathbb{R}^{B \times 258}$$
   - $\Delta s$ (1 dim): Rate of change.
   - $\Delta$ (1 dim): Net scalar progression magnitude.
   - $(\mathbf{z}_2 - \mathbf{z}_1)$ (128 dim): Directional trajectory in the latent space.
   - $|\mathbf{z}_2 - \mathbf{z}_1|$ (128 dim): Non-directional absolute magnitude of feature drift.

5. **Progression Classifier**:
   $$\mathbf{h}_1 = \text{ReLU}\left(\mathbf{W}_1 \mathbf{h} + \mathbf{b}_1\right), \quad \mathbf{W}_1 \in \mathbb{R}^{64 \times 258}$$
   $$\mathbf{h}_2 = \text{Dropout}_{p=0.4}(\mathbf{h}_1)$$
   $$\mathbf{y}_{\text{logits}} = \mathbf{W}_2 \mathbf{h}_2 + \mathbf{b}_2, \quad \mathbf{W}_2 \in \mathbb{R}^{3 \times 64}$$
   Predicted class probabilities: $\mathbf{p} = \text{Softmax}(\mathbf{y}_{\text{logits}}) \in \mathbb{R}^{B \times 3}$.

#### Progression Class Definitions
- `0: Stable`: No cognitive decline across the observed time interval.
- `1: Progressing`: Conversion from Mild Cognitive Impairment (MCI) to early Alzheimer's Disease (AD), or moderate progression.
- `2: Advanced`: Rapid conversion or advanced Alzheimer's disease with extensive neurodegeneration.

---

### 3.5 Stage E: Multi-Objective Composite Loss (`losses/composite_loss.py`)

The total objective function combines classification accuracy, temporal monotonic consistency, and mask sparsity:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_1 \mathcal{L}_{\text{ranking}} + \lambda_2 \mathcal{L}_{\text{sparsity}}$$

#### 1. Class-Weighted Cross-Entropy ($\mathcal{L}_{\text{CE}}$)
$$\mathcal{L}_{\text{CE}} = - \frac{1}{B} \sum_{i=1}^B w_{y_i} \log \left(\frac{\exp(y_{\text{logits}, i, y_i})}{\sum_{c=0}^2 \exp(y_{\text{logits}, i, c})}\right)$$
Default class weights: $\mathbf{w} = [1.0, 1.5, 2.0]$ to account for class imbalance in clinical cohorts.

#### 2. Monotonic Severity Ranking Loss ($\mathcal{L}_{\text{ranking}}$)
Enforces clinical realism on the latent severity scores:
- **Progressing Patients ($y_i \in \{1, 2\}$)**: Disease severity should strictly increase ($s_2 \ge s_1 + \epsilon$). Penalizes reversals:
  $$\mathcal{L}_{\text{prog}, i} = \max(0, \, s_{1, i} - s_{2, i} + \epsilon)$$
- **Stable Patients ($y_i = 0$)**: Disease severity should remain approximately constant ($|s_2 - s_1| \le \epsilon_{\text{stable}}$). Penalizes large divergence:
  $$\mathcal{L}_{\text{stable}, i} = \max(0, \, |s_{2, i} - s_{1, i}| - \epsilon_{\text{stable}})$$

$$\mathcal{L}_{\text{ranking}} = \frac{1}{B} \sum_{i=1}^B \left[ \mathbb{I}(y_i > 0) \cdot \mathcal{L}_{\text{prog}, i} + \mathbb{I}(y_i = 0) \cdot \mathcal{L}_{\text{stable}, i} \right]$$
Default hyperparameters: $\epsilon = 0.10$, $\epsilon_{\text{stable}} = 0.05$, $\lambda_1 = 0.5$.

#### 3. $L_1$ Mask Sparsity Loss ($\mathcal{L}_{\text{sparsity}}$)
Prevents trivial solutions where the gate sets $M_t = 1$ everywhere (which would bypass explainability):
$$\mathcal{L}_{\text{sparsity}} = \frac{1}{2 \cdot B \cdot 128^3} \sum_{i=1}^B \sum_{v=1}^{128^3} \left( |M_{1, i, v}| + |M_{2, i, v}| \right)$$
Default hyperparameter: $\lambda_2 = 0.05$.

---

## 4. Explainability and Counterfactual Validation

### 4.1 Counterfactual Confidence Drop ($\text{Drop}_{\text{cf}}$)
To quantitatively verify that the Ante-Hoc Saliency Gate discovers true causal regions rather than superficial background correlations, DLC-Net computes the Counterfactual Confidence Drop during evaluation:

1. Feed factual volumes $X_{\text{gated}, 1}, X_{\text{gated}, 2}$ through the network to obtain predicted probability on the true label $y$:
   $$P_{\text{factual}} = P(Y = y \mid X_{\text{gated}})$$
2. Feed inverted counterfactual volumes $X_{\text{cf}, 1}, X_{\text{cf}, 2}$ through the shared backbone and progression head:
   $$P_{\text{cf}} = P(Y = y \mid X_{\text{cf}})$$
3. Compute confidence drop:
   $$\text{Drop}_{\text{cf}} = P(Y = y \mid X_{\text{gated}}) - P(Y = y \mid X_{\text{cf}})$$

**Interpretation**:
- $\text{Drop}_{\text{cf}} \gg 0$: The masked regions contain the necessary causal evidence. Removing them causes model confidence to collapse.
- $\text{Drop}_{\text{cf}} \approx 0$ or $\le 0$: The saliency gate is highlighting irrelevant regions; the model can predict disease status equally well without them.

---

## 5. Dimensional Progression and Tensor Shape Trace

The following table traces tensor dimensions for a batch of $B = 2$ scan pairs:

| Pipeline Stage | Module Name | Tensor Identifier | Output Shape | Data Range / Dtype |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | `DataLoader` | $X_1, X_2$ | $(2, 1, 128, 128, 128)$ | $[0, 1]$, Float32 |
| | | $\Delta t$ | $(2, 1)$ | $> 0$, Float32 |
| **Stage B** | `BiTemporalSaliencyGate` | $D_{12} = \|X_2 - X_1\|$ | $(2, 1, 128, 128, 128)$ | $[0, 1]$, Float32 |
| | | $U_1, U_2$ | $(2, 2, 128, 128, 128)$ | Float32 |
| | | $M_1, M_2$ | $(2, 1, 128, 128, 128)$ | $(0, 1)$, Float32 |
| | | $X_{\text{gated}, 1}, X_{\text{gated}, 2}$ | $(2, 1, 128, 128, 128)$ | $[0, 1]$, Float32 |
| | | $X_{\text{cf}, 1}, X_{\text{cf}, 2}$ | $(2, 1, 128, 128, 128)$ | $[0, 1]$, Float32 |
| | | $\Delta M = \|M_2 - M_1\|$ | $(2, 1, 128, 128, 128)$ | $[0, 1]$, Float32 |
| **Stem** | `ResAxial3DEncoder.stem` | `stem_out` | $(2, 16, 64, 64, 64)$ | Float32 |
| **Stage 1** | `stage1_block` + `stage1_pool`| `stage1_out` | $(2, 32, 32, 32, 32)$ | Float32 |
| **Stage 2** | `stage2_block` | `s2_res` | $(2, 64, 32, 32, 32)$ | Float32 |
| | `AxialAttention3D` (D $\to$ H $\to$ W) | `s2_attn` | $(2, 64, 32, 32, 32)$ | Float32 |
| | `stage2_pool` | `stage2_out` | $(2, 64, 16, 16, 16)$ | Float32 |
| **Stage 3** | `stage3_block` | `s3_res` | $(2, 128, 16, 16, 16)$ | Float32 |
| | `AxialAttention3D` (D $\to$ H $\to$ W) | `s3_attn` | $(2, 128, 16, 16, 16)$ | Float32 |
| | `stage3_pool` | `stage3_out` | $(2, 128, 8, 8, 8)$ | Float32 |
| **Global Pool** | `AdaptiveAvgPool3D` + `Flatten` | $\mathbf{z}_1, \mathbf{z}_2$ | $(2, 128)$ | Float32 |
| **Stage D** | `severity_linear` + `Sigmoid` | $s_1, s_2$ | $(2, 1)$ | $[0, 1]$, Float32 |
| | Progression Delta | $\Delta$ | $(2, 1)$ | $[-1, 1]$, Float32 |
| | Progression Velocity | $\Delta s$ | $(2, 1)$ | Float32 |
| | Vector Concatenation | $\mathbf{h}$ | $(2, 258)$ | Float32 |
| | Classifier MLP | `logits` | $(2, 3)$ | Float32 |
| | Softmax | `probs` | $(2, 3)$ | $[0, 1]$, sum = 1 |
| **Stage E** | `CompositeDLCNetLoss` | `loss_total` | Scalar () | $\ge 0$, Float32 |

---

## 6. Structural Invariants and Guarantees

1. **Exact Symmetry**:
   $$\text{If } X_1 \equiv X_2 \implies M_1 = M_2 \implies \Delta M = \mathbf{0} \implies \mathbf{z}_1 = \mathbf{z}_2 \implies s_1 = s_2 \implies \Delta = 0 \implies \Delta s = 0$$
   Verified automatically in `test_pipeline.py`.
2. **Reconstruction Complementarity**:
   $$X_{\text{gated}, t} + X_{\text{cf}, t} \equiv X_t \quad \forall t \in \{1, 2\}$$
3. **Bounded Latent Severity**:
   $$s_t \in [0, 1] \quad \forall t \in \{1, 2\}$$
4. **Zero Subject-Level Data Leakage**:
   $$\text{Train}_{\text{ptids}} \cap \text{Val}_{\text{ptids}} = \emptyset, \quad \text{Train}_{\text{ptids}} \cap \text{Test}_{\text{ptids}} = \emptyset$$
