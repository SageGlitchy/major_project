"""Module 6: Evaluation Metrics and Counterfactual Confidence Drop (`evaluation/metrics.py`).

Implements:
- Classification metrics:
    * Accuracy
    * Macro-F1
    * Cohen's Kappa
    * Multi-class ROC-AUC (One-vs-Rest)
- Counterfactual Validation Metric (`Drop_cf`):
    Runs a secondary forward pass under `torch.no_grad()` by feeding X_{cf, 1} and X_{cf, 2}
    into the Siamese backbone and Progression Head, computing the confidence drop on true
    progression classes (c in {1, 2}):
        Drop_cf = P(Y = y | X_gated) - P(Y = y | X_cf)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

try:
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        f1_score,
        roc_auc_score,
    )

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute multi-class classification accuracy."""
    y_t = np.asarray(y_true, dtype=np.int64).ravel()
    y_p = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_t.size == 0:
        return 0.0
    if HAS_SKLEARN:
        return float(accuracy_score(y_t, y_p))
    return float(np.mean(y_t == y_p))


def compute_macro_f1(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3
) -> float:
    """Compute unweighted Macro-F1 score across all progression classes."""
    y_t = np.asarray(y_true, dtype=np.int64).ravel()
    y_p = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_t.size == 0:
        return 0.0
    if HAS_SKLEARN:
        return float(
            f1_score(
                y_t,
                y_p,
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )
        )

    f1_per_class = []
    for c in range(num_classes):
        tp = float(np.sum((y_t == c) & (y_p == c)))
        fp = float(np.sum((y_t != c) & (y_p == c)))
        fn = float(np.sum((y_t == c) & (y_p != c)))
        denom = 2.0 * tp + fp + fn
        f1_per_class.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1_per_class))


def compute_cohens_kappa(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3
) -> float:
    """Compute Cohen's Kappa inter-rater agreement statistic."""
    y_t = np.asarray(y_true, dtype=np.int64).ravel()
    y_p = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_t.size == 0:
        return 0.0
    if HAS_SKLEARN:
        if len(np.unique(y_t)) == 1 and len(np.unique(y_p)) == 1 and y_t[0] == y_p[0]:
            return 1.0
        return float(cohen_kappa_score(y_t, y_p, labels=list(range(num_classes))))

    n = float(y_t.size)
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(y_t, y_p):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1.0
    p_o = np.trace(cm) / n
    row_totals = np.sum(cm, axis=1)
    col_totals = np.sum(cm, axis=0)
    p_e = float(np.sum(row_totals * col_totals)) / (n * n)
    if np.isclose(1.0 - p_e, 0.0):
        return 1.0 if np.isclose(p_o, 1.0) else 0.0
    return float((p_o - p_e) / (1.0 - p_e))


def _binary_roc_auc(y_true_bin: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Compute exact Wilcoxon-Mann-Whitney ROC-AUC for a binary target vector."""
    pos_mask = y_true_bin == 1
    neg_mask = y_true_bin == 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    if n_pos == 0 or n_neg == 0:
        return None

    pos_scores = y_score[pos_mask][:, None]
    neg_scores = y_score[neg_mask][None, :]
    greater = np.sum(pos_scores > neg_scores)
    ties = np.sum(pos_scores == neg_scores)
    return float((greater + 0.5 * ties) / float(n_pos * n_neg))


def compute_multiclass_roc_auc(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    num_classes: int = 3,
) -> float:
    """Compute One-vs-Rest (OvR) macro-averaged multi-class ROC-AUC."""
    y_t = np.asarray(y_true, dtype=np.int64).ravel()
    probs = np.asarray(y_probs, dtype=np.float64)
    if y_t.size == 0 or probs.shape[0] != y_t.size:
        return 0.0

    if HAS_SKLEARN and len(np.unique(y_t)) == num_classes:
        try:
            return float(
                roc_auc_score(y_t, probs, multi_class="ovr", average="macro")
            )
        except ValueError:
            pass

    aucs = []
    for c in range(num_classes):
        y_bin = (y_t == c).astype(np.int64)
        auc_c = _binary_roc_auc(y_bin, probs[:, c])
        if auc_c is not None:
            aucs.append(auc_c)
    return float(np.mean(aucs)) if aucs else 0.5


def compute_counterfactual_drop(
    probs_factual: torch.Tensor,
    probs_cf: torch.Tensor,
    targets: torch.Tensor,
    progression_classes: Sequence[int] = (1, 2),
) -> Dict[str, Union[torch.Tensor, float]]:
    """Compute the Counterfactual Confidence Drop metric (`Drop_cf`):

        Drop_cf = P(Y = y | X_gated) - P(Y = y | X_cf)

    Computes both the per-sample drop on ground-truth class `y`, the mean drop on
    true progression classes (`y in {1, 2}`), and the overall mean drop across the batch.

    Args:
        probs_factual: Factual class probability tensor P(Y | X_gated) of shape (B, C).
        probs_cf: Counterfactual class probability tensor P(Y | X_cf) of shape (B, C).
        targets: Ground-truth class indices of shape (B,) in {0, 1, 2}.
        progression_classes: Classes representing active disease progression (default: (1, 2)).

    Returns:
        Dictionary containing:
            "drop_cf_per_sample": Tensor of shape (B,) with P(Y=y|X_gated) - P(Y=y|X_cf)
            "drop_cf_progression_mean": Float mean Drop_cf over samples with y in {1, 2}
            "drop_cf_all_mean": Float mean Drop_cf over all samples in the batch
            "p_factual_true": Tensor of shape (B,) with P(Y=y | X_gated)
            "p_cf_true": Tensor of shape (B,) with P(Y=y | X_cf)
    """
    if probs_factual.shape != probs_cf.shape:
        raise ValueError(
            f"Factual ({probs_factual.shape}) and counterfactual ({probs_cf.shape}) "
            "probability distributions must have matching shapes."
        )

    y_idx = targets.view(-1, 1).long()
    p_factual_true = torch.gather(probs_factual, dim=1, index=y_idx).squeeze(1)
    p_cf_true = torch.gather(probs_cf, dim=1, index=y_idx).squeeze(1)

    drop_per_sample = p_factual_true - p_cf_true
    drop_all_mean = float(torch.mean(drop_per_sample).item())

    y_flat = targets.view(-1)
    prog_mask = torch.zeros_like(y_flat, dtype=torch.bool)
    for cls_id in progression_classes:
        prog_mask = prog_mask | (y_flat == int(cls_id))

    if torch.any(prog_mask):
        drop_prog_mean = float(torch.mean(drop_per_sample[prog_mask]).item())
    else:
        drop_prog_mean = drop_all_mean

    return {
        "drop_cf_per_sample": drop_per_sample,
        "drop_cf_progression_mean": drop_prog_mean,
        "drop_cf_all_mean": drop_all_mean,
        "p_factual_true": p_factual_true,
        "p_cf_true": p_cf_true,
    }


@torch.no_grad()
def evaluate_counterfactual_batch(
    model: torch.nn.Module,
    X_1: torch.Tensor,
    X_2: torch.Tensor,
    delta_t: torch.Tensor,
    targets: torch.Tensor,
    progression_classes: Sequence[int] = (1, 2),
) -> Dict[str, Any]:
    """Run factual and counterfactual forward passes under `torch.no_grad()` and compute `Drop_cf`."""
    was_training = model.training
    model.eval()
    try:
        cf_out = model.forward_counterfactual(X_1, X_2, delta_t)
        drop_stats = compute_counterfactual_drop(
            probs_factual=cf_out["probs_factual"],
            probs_cf=cf_out["probs_cf"],
            targets=targets,
            progression_classes=progression_classes,
        )
        return {**cf_out, **drop_stats}
    finally:
        if was_training:
            model.train()


def compute_classification_metrics(
    y_true: Union[np.ndarray, Sequence[int]],
    y_probs: np.ndarray,
    drop_cf_progression: Optional[float] = None,
    drop_cf_all: Optional[float] = None,
    num_classes: int = 3,
) -> Dict[str, float]:
    """Compute complete evaluation dictionary: Accuracy, Macro-F1, Cohen's Kappa, ROC-AUC, and Drop_cf."""
    y_t = np.asarray(y_true, dtype=np.int64).ravel()
    probs = np.asarray(y_probs, dtype=np.float64)
    y_pred = np.argmax(probs, axis=1) if probs.ndim == 2 else np.asarray(probs, dtype=np.int64)

    metrics = {
        "accuracy": compute_accuracy(y_t, y_pred),
        "macro_f1": compute_macro_f1(y_t, y_pred, num_classes=num_classes),
        "cohens_kappa": compute_cohens_kappa(y_t, y_pred, num_classes=num_classes),
        "roc_auc": compute_multiclass_roc_auc(y_t, probs, num_classes=num_classes),
    }
    if drop_cf_progression is not None:
        metrics["drop_cf"] = float(drop_cf_progression)
    if drop_cf_all is not None:
        metrics["drop_cf_all"] = float(drop_cf_all)
    return metrics


# Alias for backward compatibility
evaluate_model_Counterfactual_and_metrics = evaluate_counterfactual_batch
