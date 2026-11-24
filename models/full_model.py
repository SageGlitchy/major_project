"""End-to-End Wrapper for Bi-Temporal Ante-Hoc Siamese 3D Res-Axial Network (DLC-Net).

Integrates:
- Stage B: `BiTemporalSaliencyGate` (`models/ante_hoc_gate.py`)
- Stage C1 & C2: `Siamese3DBackbone` with `AxialAttention3D` (`models/siamese_backbone.py`)
- Stage D: `TimeAwareProgressionHead` (`models/progression_head.py`)

Provides both standard factual forward execution (`forward`) and counterfactual
intervention execution (`forward_counterfactual`) for computing `Drop_cf`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ante_hoc_gate import BiTemporalSaliencyGate
from .progression_head import TimeAwareProgressionHead
from .siamese_backbone import Siamese3DBackbone


class BiTemporalSiameseDLCNet(nn.Module):
    """Full Bi-Temporal Ante-Hoc Siamese 3D Res-Axial Network (DLC-Net)."""

    def __init__(
        self,
        in_channels: int = 1,
        gate_hidden_channels: int = 8,
        stem_channels: int = 16,
        stage1_channels: int = 32,
        stage2_channels: int = 64,
        stage3_channels: int = 128,
        stage2_spatial_dim: int = 32,
        stage3_spatial_dim: int = 16,
        axial_num_heads: int = 4,
        axial_dropout: float = 0.0,
        leaky_relu_slope: float = 0.01,
        head_hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.4,
        delta_t_min: float = 1e-3,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.ante_hoc_gate = BiTemporalSaliencyGate(
            in_channels=2 * in_channels,
            hidden_channels=gate_hidden_channels,
            out_channels=in_channels,
            kernel_size=3,
        )
        self.siamese_backbone = Siamese3DBackbone(
            in_channels=in_channels,
            stem_channels=stem_channels,
            stage1_channels=stage1_channels,
            stage2_channels=stage2_channels,
            stage3_channels=stage3_channels,
            stage2_spatial_dim=stage2_spatial_dim,
            stage3_spatial_dim=stage3_spatial_dim,
            num_heads=axial_num_heads,
            axial_dropout=axial_dropout,
            leaky_relu_slope=leaky_relu_slope,
            use_checkpoint=use_checkpoint,
        )
        self.progression_head = TimeAwareProgressionHead(
            latent_dim=stage3_channels,
            hidden_dim=head_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            delta_t_min=delta_t_min,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BiTemporalSiameseDLCNet":
        """Instantiate `BiTemporalSiameseDLCNet` from a parsed YAML configuration dictionary."""
        proj_cfg = config.get("project", {})
        model_cfg = config.get("model", {})
        gate_cfg = model_cfg.get("gate", {})
        bb_cfg = model_cfg.get("backbone", {})
        head_cfg = model_cfg.get("progression_head", {})

        return cls(
            in_channels=int(model_cfg.get("in_channels", 1)),
            gate_hidden_channels=int(gate_cfg.get("hidden_channels", 8)),
            stem_channels=int(bb_cfg.get("stem_channels", 16)),
            stage1_channels=int(bb_cfg.get("stage1_channels", 32)),
            stage2_channels=int(bb_cfg.get("stage2_channels", 64)),
            stage3_channels=int(bb_cfg.get("stage3_channels", 128)),
            stage2_spatial_dim=int(bb_cfg.get("stage2_spatial_dim", 32)),
            stage3_spatial_dim=int(bb_cfg.get("stage3_spatial_dim", 16)),
            axial_num_heads=int(bb_cfg.get("axial_num_heads", 4)),
            axial_dropout=float(bb_cfg.get("axial_dropout", 0.0)),
            leaky_relu_slope=float(bb_cfg.get("leaky_relu_slope", 0.01)),
            head_hidden_dim=int(head_cfg.get("hidden_dim", 64)),
            num_classes=int(head_cfg.get("num_classes", 3)),
            dropout=float(head_cfg.get("dropout", 0.4)),
            delta_t_min=float(head_cfg.get("delta_t_min", 1e-3)),
            use_checkpoint=bool(proj_cfg.get("use_checkpoint", True)),
        )

    def forward(
        self,
        X_1: torch.Tensor,
        X_2: torch.Tensor,
        delta_t: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, Any]:
        """End-to-end factual forward pass through Stage B, Stage C, and Stage D.

        Args:
            X_1: Baseline MRI scan tensor of shape (B, 1, 128, 128, 128).
            X_2: Follow-up MRI scan tensor of shape (B, 1, 128, 128, 128).
            delta_t: Time interval tensor of shape (B, 1) or (B,).
            return_intermediates: If True, include intermediate backbone stage outputs.

        Returns:
            Dictionary containing factual predictions, embeddings, severity scores,
            differential fusion vector, and ante-hoc saliency gate outputs.
        """
        gate_out = self.ante_hoc_gate(X_1, X_2)

        if return_intermediates:
            z_1, z_2, backbone_inter = self.siamese_backbone(
                gate_out["X_gated_1"],
                gate_out["X_gated_2"],
                return_intermediates=True,
            )
        else:
            z_1, z_2 = self.siamese_backbone(
                gate_out["X_gated_1"],
                gate_out["X_gated_2"],
                return_intermediates=False,
            )
            backbone_inter = None

        head_out = self.progression_head(z_1, z_2, delta_t)
        probs = F.softmax(head_out["logits"], dim=-1)

        output: Dict[str, Any] = {
            "logits": head_out["logits"],
            "probs": probs,
            "s_1": head_out["s_1"],
            "s_2": head_out["s_2"],
            "delta": head_out["delta"],
            "delta_s": head_out["delta_s"],
            "h": head_out["h"],
            "z_1": z_1,
            "z_2": z_2,
            "M_1": gate_out["M_1"],
            "M_2": gate_out["M_2"],
            "delta_M": gate_out["delta_M"],
            "X_gated_1": gate_out["X_gated_1"],
            "X_gated_2": gate_out["X_gated_2"],
            "X_cf_1": gate_out["X_cf_1"],
            "X_cf_2": gate_out["X_cf_2"],
        }
        if backbone_inter is not None:
            output["backbone_intermediates"] = backbone_inter
        return output

    def forward_counterfactual(
        self,
        X_1: torch.Tensor,
        X_2: torch.Tensor,
        delta_t: torch.Tensor,
        factual_outputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run both factual (X_gated) and counterfactual (X_cf = X * (1 - M)) branches
        through the shared Siamese backbone and Time-Aware Progression Head.

        Returns:
            Dictionary with keys:
                "logits_factual", "probs_factual", "s_1_factual", "s_2_factual",
                "logits_cf", "probs_cf", "s_1_cf", "s_2_cf", "z_1_cf", "z_2_cf",
                "M_1", "M_2", "delta_M", "X_gated_1", "X_gated_2", "X_cf_1", "X_cf_2"
        """
        if factual_outputs is None:
            factual_outputs = self.forward(X_1, X_2, delta_t, return_intermediates=False)

        X_cf_1 = factual_outputs["X_cf_1"]
        X_cf_2 = factual_outputs["X_cf_2"]

        z_1_cf, z_2_cf = self.siamese_backbone(X_cf_1, X_cf_2, return_intermediates=False)
        head_cf = self.progression_head(z_1_cf, z_2_cf, delta_t)
        probs_cf = F.softmax(head_cf["logits"], dim=-1)

        return {
            "logits_factual": factual_outputs["logits"],
            "probs_factual": factual_outputs["probs"],
            "s_1_factual": factual_outputs["s_1"],
            "s_2_factual": factual_outputs["s_2"],
            "logits_cf": head_cf["logits"],
            "probs_cf": probs_cf,
            "s_1_cf": head_cf["s_1"],
            "s_2_cf": head_cf["s_2"],
            "delta_cf": head_cf["delta"],
            "delta_s_cf": head_cf["delta_s"],
            "h_cf": head_cf["h"],
            "z_1_cf": z_1_cf,
            "z_2_cf": z_2_cf,
            "M_1": factual_outputs["M_1"],
            "M_2": factual_outputs["M_2"],
            "delta_M": factual_outputs["delta_M"],
            "X_gated_1": factual_outputs["X_gated_1"],
            "X_gated_2": factual_outputs["X_gated_2"],
            "X_cf_1": X_cf_1,
            "X_cf_2": X_cf_2,
        }


# Alias for convenience
DLCNet = BiTemporalSiameseDLCNet
