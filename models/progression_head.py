"""Module 5: Stage D — Time-Aware Differential Progression Head (`models/progression_head.py`).

Inputs:
    z_1 in R^{B x 128}
    z_2 in R^{B x 128}
    delta_t in R^{B x 1}

Operations:
1. Shared severity scorer:
       s_t = sigma(Linear_{128 -> 1}(z_t)) in [0, 1]^{B x 1} for t in {1, 2}
2. Scalar progression delta:
       Delta = s_2 - s_1 in [-1, 1]^{B x 1}
3. Time-normalized progression velocity:
       Delta_s = (s_2 - s_1) / max(delta_t, 10^{-3}) in R^{B x 1}
4. Differential vector fusion:
       h = [Delta_s || Delta || (z_2 - z_1) || |z_2 - z_1|] in R^{B x 258}
5. Classifier MLP:
       Linear(258, 64) -> ReLU() -> Dropout(0.4) -> Linear(64, 3) -> logits in R^{B x 3}
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class TimeAwareProgressionHead(nn.Module):
    """Shared severity scorer and Time-Aware Differential MLP Classifier Head."""

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.4,
        delta_t_min: float = 1e-3,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.fusion_dim = 2 + 2 * latent_dim  # 1 (Delta_s) + 1 (Delta) + 128 + 128 = 258
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.delta_t_min = float(delta_t_min)

        # Shared severity scorer: Linear(128 -> 1) followed by Sigmoid
        self.severity_linear = nn.Linear(latent_dim, 1)
        self.severity_sigmoid = nn.Sigmoid()

        # Classifier MLP: Linear(258, 64) -> ReLU() -> Dropout(0.4) -> Linear(64, 3)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.severity_linear.weight)
        if self.severity_linear.bias is not None:
            nn.init.zeros_(self.severity_linear.bias)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def score_severity(self, z_t: torch.Tensor) -> torch.Tensor:
        """Compute scalar disease severity score s_t = sigma(Linear(z_t)) in [0, 1]^{B x 1}."""
        return self.severity_sigmoid(self.severity_linear(z_t))

    def forward(
        self,
        z_1: torch.Tensor,
        z_2: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute latent severity scores, 258-D differential fusion vector, and class logits.

        Args:
            z_1: Baseline latent vector of shape (B, 128).
            z_2: Follow-up latent vector of shape (B, 128).
            delta_t: Time interval in years of shape (B, 1) or (B,).

        Returns:
            Dictionary containing:
                "logits": (B, 3)
                "s_1": (B, 1) in [0, 1]
                "s_2": (B, 1) in [0, 1]
                "delta": (B, 1) in [-1, 1]
                "delta_s": (B, 1)
                "h": (B, 258)
        """
        if z_1.ndim != 2 or z_2.ndim != 2:
            raise ValueError(f"Expected 2D latent vectors (B, 128), got {z_1.shape} and {z_2.shape}.")
        if z_1.shape != z_2.shape or z_1.shape[1] != self.latent_dim:
            raise ValueError(
                f"Latent vectors must both have shape (B, {self.latent_dim}), "
                f"got {z_1.shape} and {z_2.shape}."
            )

        if delta_t.ndim == 1:
            delta_t = delta_t.unsqueeze(1)
        elif delta_t.ndim != 2 or delta_t.shape[1] != 1:
            raise ValueError(f"Expected delta_t of shape (B, 1) or (B,), got {delta_t.shape}.")

        delta_t = delta_t.to(dtype=z_1.dtype, device=z_1.device)

        # 1. Shared severity scorer: s_t = sigma(Linear_{128 -> 1}(z_t)) in [0, 1]^{B x 1}
        s_1 = self.score_severity(z_1)
        s_2 = self.score_severity(z_2)

        # 2. Scalar progression delta: Delta = s_2 - s_1 in [-1, 1]^{B x 1}
        delta = s_2 - s_1

        # 3. Progression velocity: Delta_s = (s_2 - s_1) / max(delta_t, 10^{-3}) in R^{B x 1}
        safe_delta_t = torch.clamp(delta_t, min=self.delta_t_min)
        delta_s = delta / safe_delta_t

        # 4. Differential vector fusion:
        #    h = [Delta_s || Delta || (z_2 - z_1) || |z_2 - z_1|] in R^{B x 258}
        z_diff = z_2 - z_1
        z_abs_diff = torch.abs(z_diff)
        h = torch.cat([delta_s, delta, z_diff, z_abs_diff], dim=1)

        if h.shape[1] != self.fusion_dim:
            raise RuntimeError(f"Expected fusion vector dimension {self.fusion_dim}, got {h.shape[1]}.")

        # 5. Classifier MLP -> logits in R^{B x 3}
        logits = self.classifier(h)

        return {
            "logits": logits,
            "s_1": s_1,
            "s_2": s_2,
            "delta": delta,
            "delta_s": delta_s,
            "h": h,
        }
