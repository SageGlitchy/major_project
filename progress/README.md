# Training Progress & Session Logs

This directory maintains the training session logs and publication-quality PDF progress reports for the **Bi-Temporal Ante-Hoc Siamese 3D Res-Axial Network (DLC-Net)** on the ADNI longitudinal MRI dataset.

---

## Session Registry & Master Log

| Session | PDF Report | Epochs | Class Weights | Best Checkpoint | Val Accuracy | Val Macro-F1 | Key Changes & Outcomes |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **01** | [Session_01_Baseline_GPU_15Epochs.pdf](./Session_01_Baseline_GPU_15Epochs.pdf) | 15 | `[1.0, 1.0, 1.0]` | Epoch 15 | 64.00% | 38.46% | GPU baseline established (~40s/epoch). Validated pipeline; identified minority Class 1 suppression (0% recall). |
| **02** | [Session_02_Weighted_GPU_50Epochs.pdf](./Session_02_Weighted_GPU_50Epochs.pdf) | 50 | `[1.0, 3.5, 2.0]` | Epoch 27 | 72.00% | 50.29% | 3.5x penalty on Class 1 + 50 epochs. Unlocked Class 1 F1 to 33.3%, Cohen's $\kappa$ reached 0.3269, Val Accuracy reached 72%. |
| **03** | [Session_03_Enhanced_100Epochs.pdf](./Session_03_Enhanced_100Epochs.pdf) | 100 | `[1.0, 2.8, 1.8]` | Epoch 28 | **76.00%** | **61.40%** | Coupled 3D spatial jitter, label smoothing (0.05), axial dropout (0.10). Record **76% Val Acc**, **61.4% Macro-F1**, Class 1 F1 reached **66.7%** (100% precision). |
| **04** | [Session_04_Extended_500Epochs.pdf](./Session_04_Extended_500Epochs.pdf) | 67 | `[1.0, 2.8, 1.8]` | Epoch 30 | 72.00% | 50.29% | Early stopping telemetry across 67 epochs. Confirmed 25–30 epoch convergence window (sweet spot codified: `epochs: 30`). |

---

## Documented Report Structure

Each generated PDF report includes:
1. **Executive Summary & Metadata**: Hardware specs (NVIDIA RTX 3050 Ti, CUDA 12.4), PyTorch configuration, cohort demographics, spatial resolution $(128 \times 128 \times 128)$.
2. **What Changed & Motivation**: Exact differential modifications relative to the previous session.
3. **Overall Working & Architecture**: Detailed breakdown of the Siamese 3D Res-Axial backbone, ante-hoc spatial gating heads, latent severity dynamics, and counterfactual perturbation.
4. **Key Techniques & Loss Formulations**: Multi-objective formulation ($L_{\text{total}} = L_{\text{CE}} + \lambda_{\text{rank}} L_{\text{rank}} + \lambda_{\text{sparse}} L_{\text{sparse}}$), ranking loss, and AMP.
5. **Quantitative Results Dashboard**: Embedded high-resolution Matplotlib figures (Metrics bar chart & Confusion Matrix heatmap), quantitative factor table, and per-class precision/recall/specificity breakdown.
6. **Ante-Hoc Explainability Visualizations (XAI)**: Embedded multi-planar longitudinal MRI slices ($z=64, y=64, x=64$) featuring baseline saliency ($M_1$), follow-up saliency ($M_2$), and differential atrophy maps ($\Delta M$).
7. **Clinical Conclusion & Actionable Next Steps**: In-depth analysis of model dynamics, failure modes, and guidance for subsequent training sessions.

---

## How to Generate New Reports

To re-generate or add new training sessions to the PDF suite:

```powershell
python progress/generate_report.py
```
