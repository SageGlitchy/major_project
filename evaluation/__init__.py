"""Evaluation metrics and XAI visualization tools for DLC-Net."""

from .metrics import (
    compute_accuracy,
    compute_macro_f1,
    compute_cohens_kappa,
    compute_multiclass_roc_auc,
    compute_counterfactual_drop,
    evaluate_model_Counterfactual_and_metrics,
     compute_classification_metrics,
)
from .visualize_xai import (
    export_xai_nifti_volumes,
    plot_longitudinal_xai_overlays,
)

__all__ = [
    "compute_accuracy",
    "compute_macro_f1",
    "compute_cohens_kappa",
    "compute_multiclass_roc_auc",
    "compute_counterfactual_drop",
    "evaluate_model_Counterfactual_and_metrics",
    "compute_classification_metrics",
    "export_xai_nifti_volumes",
    "plot_longitudinal_xai_overlays",
]
