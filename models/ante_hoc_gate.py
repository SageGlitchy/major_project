"""Module 2: Stage B — Bi-Temporal Ante-Hoc Saliency Gate (`models/ante_hoc_gate.py`).

Enforces causal explainability at full 128 x 128 x 128 resolution prior to deep feature
extraction:
1. Structural voxel-wise difference:
       D_12 = |X_2 - X_1| in R^{B x 1 x 128 x 128 x 128}
2. Bi-temporal channel concatenation for each timepoint t in {1, 2}:
       U_t = [X_t || D_12] in R^{B x 2 x 128 x 128 x 128}
3. Shared discovery sub-network:
       Conv3d(2, 8, kernel_size=3, padding=1, bias=False)
       -> InstanceNorm3d(8, affine=True)
       -> ReLU(inplace=True)
       -> Conv3d(8, 1, kernel_size=1, bias=True)
       -> Sigmoid() -> M_t in (0, 1)^{B x 1 x 128 x 128 x 128}
4. Factual gating:
       X_{gated, t} = X_t * M_t
5. Counterfactual inversion:
       X_{cf, t} = X_t * (1 - M_t)
6. Longitudinal atrophy map:
       delta_M = |M_2 - M_1|
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class BiTemporalSaliencyGate(nn.Module):
    """Bi-Temporal Ante-Hoc Saliency Gate operating at full 128 x 128 x 128 resolution."""

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 8,
        out_channels: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if in_channels != 2:
            raise ValueError(
                f"BiTemporalSaliencyGate expects in_channels=2 ([X_t || D_12]), got {in_channels}."
            )
        padding = kernel_size // 2

        # Shared discovery layers applied identically to U_1 and U_2
        self.discovery_conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.discovery_norm = nn.InstanceNorm3d(
            num_features=hidden_channels,
            affine=True,
        )
        self.discovery_act = nn.ReLU(inplace=True)
        self.discovery_conv2 = nn.Conv3d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=True,
        )
        self.discovery_sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(
            self.discovery_conv1.weight, mode="fan_out", nonlinearity="relu"
        )
        if self.discovery_norm.weight is not None:
            nn.init.ones_(self.discovery_norm.weight)
        if self.discovery_norm.bias is not None:
            nn.init.zeros_(self.discovery_norm.bias)
        nn.init.xavier_uniform_(self.discovery_conv2.weight)
        if self.discovery_conv2.bias is not None:
            nn.init.zeros_(self.discovery_conv2.bias)

    def _compute_saliency_mask(self, u_t: torch.Tensor) -> torch.Tensor:
        """Compute saliency mask M_t in (0, 1)^{B x 1 x D x H x W} from U_t = [X_t || D_12]."""
        h = self.discovery_conv1(u_t)
        h = self.discovery_norm(h)
        h = self.discovery_act(h)
        logits_m = self.discovery_conv2(h)
        m_t = self.discovery_sigmoid(logits_m)
        return m_t

    def forward(
        self, X_1: torch.Tensor, X_2: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute factual gated volumes, counterfactual inverted volumes, and saliency masks.

        Args:
            X_1: Baseline 3D MRI tensor of shape (B, 1, 128, 128, 128).
            X_2: Follow-up 3D MRI tensor of shape (B, 1, 128, 128, 128).

        Returns:
            Dictionary with keys:
                "X_gated_1", "X_gated_2", "X_cf_1", "X_cf_2", "M_1", "M_2", "delta_M"
        """
        if X_1.ndim != 5 or X_2.ndim != 5:
            raise ValueError(
                f"Expected 5D inputs (B, 1, D, H, W), got {X_1.shape} and {X_2.shape}."
            )
        if X_1.shape != X_2.shape:
            raise ValueError(
                f"Baseline X_1 shape {X_1.shape} and follow-up X_2 shape {X_2.shape} must match."
            )

        # 1. Structural voxel-wise absolute difference: D_12 = |X_2 - X_1|
        D_12 = torch.abs(X_2 - X_1)

        # 2. Concatenate along channel dimension (dim=1): U_t = [X_t || D_12]
        U_1 = torch.cat([X_1, D_12], dim=1)
        U_2 = torch.cat([X_2, D_12], dim=1)

        # 3. Pass through shared discovery layers to obtain M_1 and M_2
        M_1 = self._compute_saliency_mask(U_1)
        M_2 = self._compute_saliency_mask(U_2)

        # 4. Factual gating: X_{gated, t} = X_t * M_t
        X_gated_1 = X_1 * M_1
        X_gated_2 = X_2 * M_2

        # 5. Counterfactual inversion: X_{cf, t} = X_t * (1 - M_t)
        X_cf_1 = X_1 * (1.0 - M_1)
        X_cf_2 = X_2 * (1.0 - M_2)

        # 6. Longitudinal atrophy map: delta_M = |M_2 - M_1|
        delta_M = torch.abs(M_2 - M_1)

        return {
            "X_gated_1": X_gated_1,
            "X_gated_2": X_gated_2,
            "X_cf_1": X_cf_1,
            "X_cf_2": X_cf_2,
            "M_1": M_1,
            "M_2": M_2,
            "delta_M": delta_M,
        }
