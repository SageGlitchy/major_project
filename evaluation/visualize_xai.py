"""Module 6: 3D NIfTI Export and 2D Multi-Planar Slice Overlays (`evaluation/visualize_xai.py`).

Generates:
1. 3D NIfTI volumes (.nii.gz) for baseline saliency mask M_1, follow-up saliency mask M_2,
   and longitudinal atrophy map |M_2 - M_1|.
2. Multi-planar (Axial, Coronal, Sagittal) 2D slice overlay figures superimposing M_1 on X_1,
   M_2 on X_2, and the longitudinal atrophy map |M_2 - M_1| on |X_2 - X_1|.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.preprocess_mri import save_mri_volume


def _to_3d_numpy(tensor_or_array: Union[torch.Tensor, np.ndarray], batch_idx: int = 0) -> np.ndarray:
    """Convert a 5D (B, 1, D, H, W), 4D (1, D, H, W), or 3D (D, H, W) tensor/array to 3D float32 numpy."""
    if isinstance(tensor_or_array, torch.Tensor):
        arr = tensor_or_array.detach().cpu().float().numpy()
    else:
        arr = np.asarray(tensor_or_array, dtype=np.float32)

    if arr.ndim == 5:
        arr = arr[batch_idx, 0]
    elif arr.ndim == 4:
        arr = arr[0]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 3D/4D/5D volume, got shape {arr.shape}")
    return arr.astype(np.float32)


def export_xai_nifti_volumes(
    X_1: Union[torch.Tensor, np.ndarray],
    X_2: Union[torch.Tensor, np.ndarray],
    M_1: Union[torch.Tensor, np.ndarray],
    M_2: Union[torch.Tensor, np.ndarray],
    output_dir: Union[str, Path],
    subject_id: str = "subject_001",
    batch_idx: int = 0,
    affine: Optional[np.ndarray] = None,
) -> Dict[str, Path]:
    """Save 3D NIfTI volumes for X_1, X_2, M_1, M_2, and Atrophy Map |M_2 - M_1|."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x1_np = _to_3d_numpy(X_1, batch_idx=batch_idx)
    x2_np = _to_3d_numpy(X_2, batch_idx=batch_idx)
    m1_np = _to_3d_numpy(M_1, batch_idx=batch_idx)
    m2_np = _to_3d_numpy(M_2, batch_idx=batch_idx)
    atrophy_np = np.abs(m2_np - m1_np)

    aff = affine if affine is not None else np.eye(4, dtype=np.float32)

    paths = {
        "X_1": save_mri_volume(x1_np, out_dir / f"{subject_id}_X1_baseline.nii.gz", affine=aff),
        "X_2": save_mri_volume(x2_np, out_dir / f"{subject_id}_X2_followup.nii.gz", affine=aff),
        "M_1": save_mri_volume(m1_np, out_dir / f"{subject_id}_M1_saliency.nii.gz", affine=aff),
        "M_2": save_mri_volume(m2_np, out_dir / f"{subject_id}_M2_saliency.nii.gz", affine=aff),
        "delta_M": save_mri_volume(
            atrophy_np, out_dir / f"{subject_id}_delta_M_atrophy_map.nii.gz", affine=aff
        ),
    }
    return paths


def plot_longitudinal_xai_overlays(
    X_1: Union[torch.Tensor, np.ndarray],
    X_2: Union[torch.Tensor, np.ndarray],
    M_1: Union[torch.Tensor, np.ndarray],
    M_2: Union[torch.Tensor, np.ndarray],
    output_path: Union[str, Path],
    slice_coords: Optional[Tuple[int, int, int]] = None,
    subject_id: str = "subject_001",
    s_1: Optional[float] = None,
    s_2: Optional[float] = None,
    pred_label: Optional[str] = None,
    batch_idx: int = 0,
    alpha: float = 0.45,
) -> Path:
    """Render a 3x3 publication-quality multi-planar (Axial, Coronal, Sagittal) figure showing:
    - Row 1: Baseline MRI X_1 overlaid with Ante-Hoc Saliency Mask M_1
    - Row 2: Follow-up MRI X_2 overlaid with Ante-Hoc Saliency Mask M_2
    - Row 3: Structural Change overlaid with Longitudinal Atrophy Map |M_2 - M_1|
    """
    x1_np = _to_3d_numpy(X_1, batch_idx=batch_idx)
    x2_np = _to_3d_numpy(X_2, batch_idx=batch_idx)
    m1_np = _to_3d_numpy(M_1, batch_idx=batch_idx)
    m2_np = _to_3d_numpy(M_2, batch_idx=batch_idx)
    delta_m_np = np.abs(m2_np - m1_np)

    D, H, W = x1_np.shape
    if slice_coords is None:
        d_idx, h_idx, w_idx = D // 2, H // 2, W // 2
    else:
        d_idx, h_idx, w_idx = slice_coords

    fig, axes = plt.subplots(3, 3, figsize=(12, 11), dpi=150)
    title_parts = [f"DLC-Net Longitudinal Ante-Hoc XAI — {subject_id}"]
    if s_1 is not None and s_2 is not None:
        title_parts.append(f"Severity: s1={s_1:.3f} -> s2={s_2:.3f} (Δ={s_2 - s_1:+.3f})")
    if pred_label is not None:
        title_parts.append(f"Prediction: {pred_label}")
    fig.suptitle(" | ".join(title_parts), fontsize=12, fontweight="bold", y=0.98)

    rows_data = [
        ("Baseline (X1 + M1)", x1_np, m1_np, "magma"),
        ("Follow-up (X2 + M2)", x2_np, m2_np, "magma"),
        ("Atrophy Map (|M2 - M1|)", 0.5 * (x1_np + x2_np), delta_m_np, "inferno"),
    ]
    plane_names = [
        f"Axial (z={d_idx})",
        f"Coronal (y={h_idx})",
        f"Sagittal (x={w_idx})",
    ]

    for row_idx, (row_label, base_vol, heat_vol, cmap_name) in enumerate(rows_data):
        base_slices = [
            base_vol[d_idx, :, :],
            base_vol[:, h_idx, :],
            base_vol[:, :, w_idx],
        ]
        heat_slices = [
            heat_vol[d_idx, :, :],
            heat_vol[:, h_idx, :],
            heat_vol[:, :, w_idx],
        ]
        vmax = max(float(np.max(heat_vol)), 1e-4)

        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            ax.imshow(np.rot90(base_slices[col_idx]), cmap="gray", vmin=0.0, vmax=1.0)
            im = ax.imshow(
                np.rot90(heat_slices[col_idx]),
                cmap=cmap_name,
                alpha=alpha,
                vmin=0.0,
                vmax=vmax,
            )
            if row_idx == 0:
                ax.set_title(plane_names[col_idx], fontsize=10, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=10, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 2:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_file), bbox_inches="tight")
    plt.close(fig)
    return out_file
