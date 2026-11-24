"""Module 6: Stage E — Composite Loss (`losses/composite_loss.py`).

Implements the joint objective function:
    L_total = L_CE(y, y_hat) + lambda_1 * L_ranking(s_1, s_2, y) + lambda_2 * L_sparsity(M_1, M_2)

Components:
1. L_CE: Class-weighted `nn.CrossEntropyLoss(weight=class_weights)`.
2. L_ranking: Monotonic temporal severity ranking loss:
   - Progressing samples (y in {1, 2}): penalize reversals where s_1 >= s_2 - epsilon:
         I(y_i > 0) * max(0, s_{1,i} - s_{2,i} + epsilon)
   - Stable samples (y == 0): penalize large divergence where |s_{2,i} - s_{1,i}| > epsilon_stable:
         I(y_i == 0) * max(0, |s_{2,i} - s_{1,i}| - epsilon_stable)
   Averaged over batch size B.
3. L_sparsity: Mean L1 norm across all voxels of M_1 and M_2:
         L_sparsity = (||M_1||_1 + ||M_2||_1) / (2 * B * 128^3)

Default hyperparameters:
    lambda_1 = 0.5, lambda_2 = 0.05, epsilon = 0.10, epsilon_stable = 0.05
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_monotonic_ranking_loss(
    s_1: torch.Tensor,
    s_2: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 0.10,
    epsilon_stable: float = 0.05,
) -> torch.Tensor:
    """Compute monotonic severity ranking loss for progressing (y > 0) and stable (y == 0) pairs.

    Args:
        s_1: Baseline severity scores of shape (B, 1) or (B,) in [0, 1].
        s_2: Follow-up severity scores of shape (B, 1) or (B,) in [0, 1].
        targets: Ground-truth progression class labels of shape (B,) in {0, 1, 2}.
        epsilon: Margin for progressing pairs (default 0.10).
        epsilon_stable: Tolerance for stable pairs (default 0.05).

    Returns:
        Scalar ranking loss tensor averaged over batch size B.
    """
    s1_flat = s_1.view(-1)
    s2_flat = s_2.view(-1)
    y_flat = targets.view(-1)

    if s1_flat.shape[0] != y_flat.shape[0] or s2_flat.shape[0] != y_flat.shape[0]:
        raise ValueError(
            f"Batch size mismatch in ranking loss: s_1={s_1.shape}, s_2={s_2.shape}, y={targets.shape}"
        )

    progressing_mask = (y_flat > 0).to(dtype=s1_flat.dtype)
    stable_mask = (y_flat == 0).to(dtype=s1_flat.dtype)

    # Progressing penalty: max(0, s_{1,i} - s_{2,i} + epsilon)
    prog_penalty = F.relu(s1_flat - s2_flat + float(epsilon))

    # Stable penalty: max(0, |s_{2,i} - s_{1,i}| - epsilon_stable)
    stable_penalty = F.relu(torch.abs(s2_flat - s1_flat) - float(epsilon_stable))

    sample_losses = progressing_mask * prog_penalty + stable_mask * stable_penalty
    return torch.mean(sample_losses)


def compute_mask_sparsity_loss(
    M_1: torch.Tensor,
    M_2: torch.Tensor,
) -> torch.Tensor:
    """Compute mean voxel-wise L1 sparsity regularization across M_1 and M_2.

    Formula:
        L_sparsity = (||M_1||_1 + ||M_2||_1) / (2 * B * D * H * W)
        (For 128 x 128 x 128 volumes, denominator is 2 * B * 128^3.)
    """
    if M_1.shape != M_2.shape:
        raise ValueError(
            f"Saliency masks M_1 ({M_1.shape}) and M_2 ({M_2.shape}) must have identical shapes."
        )
    total_voxels = float(M_1.numel() + M_2.numel())
    l1_sum = torch.sum(torch.abs(M_1)) + torch.sum(torch.abs(M_2))
    return l1_sum / total_voxels


class CompositeDLCNetLoss(nn.Module):
    """Stage E Composite Objective Function combining Cross-Entropy, Monotonic Ranking,
    and L1 Saliency Mask Sparsity."""

    def __init__(
        self,
        class_weights: Optional[Union[Sequence[float], torch.Tensor]] = None,
        lambda_1: float = 0.5,
        lambda_2: float = 0.05,
        epsilon: float = 0.10,
        epsilon_stable: float = 0.05,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.epsilon = float(epsilon)
        self.epsilon_stable = float(epsilon_stable)
        self.label_smoothing = float(label_smoothing)

        if class_weights is not None:
            weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", weight_tensor)
        else:
            self.class_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        s_1: torch.Tensor,
        s_2: torch.Tensor,
        M_1: torch.Tensor,
        M_2: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute total composite loss and individual loss terms.

        Args:
            logits: Predicted class logits of shape (B, 3).
            targets: Ground-truth progression labels of shape (B,) in {0, 1, 2}.
            s_1: Baseline severity scores of shape (B, 1).
            s_2: Follow-up severity scores of shape (B, 1).
            M_1: Baseline saliency mask of shape (B, 1, 128, 128, 128).
            M_2: Follow-up saliency mask of shape (B, 1, 128, 128, 128).

        Returns:
            Dictionary with keys:
                "loss_total": L_total scalar tensor (with grad)
                "loss_ce": L_CE scalar tensor
                "loss_ranking": L_ranking scalar tensor
                "loss_sparsity": L_sparsity scalar tensor
        """
        weight = (
            self.class_weights.to(device=logits.device, dtype=logits.dtype)
            if self.class_weights is not None
            else None
        )
        loss_ce = F.cross_entropy(
            logits,
            targets.long(),
            weight=weight,
            label_smoothing=self.label_smoothing,
        )
        loss_ranking = compute_monotonic_ranking_loss(
            s_1=s_1,
            s_2=s_2,
            targets=targets,
            epsilon=self.epsilon,
            epsilon_stable=self.epsilon_stable,
        )
        loss_sparsity = compute_mask_sparsity_loss(M_1=M_1, M_2=M_2)

        loss_total = (
            loss_ce
            + self.lambda_1 * loss_ranking
            + self.lambda_2 * loss_sparsity
        )

        return {
            "loss_total": loss_total,
            "loss_ce": loss_ce,
            "loss_ranking": loss_ranking,
            "loss_sparsity": loss_sparsity,
        }
