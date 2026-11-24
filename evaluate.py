"""Standalone Model Evaluation and Diagnostics Script (evaluate.py).

Computes comprehensive performance factors and explainability metrics for DLC-Net:
1. Multi-Class Accuracy
2. Macro-Averaged F1-Score
3. Cohen's Kappa Inter-Rater Agreement (κ)
4. One-vs-Rest (OvR) Macro ROC-AUC
5. Counterfactual Confidence Drop (Drop_cf) on Progression Classes (y in {1, 2})
6. Confusion Matrix & Per-Class Precision / Recall / F1
7. Latent Disease Severity Progression Dynamics (s_1, s_2, Δ, Δs)
8. Optional 2D/3D Explainable AI (XAI) multi-planar overlays
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import yaml

from data.dataset import (
    LABEL_MAP,
    ScanPairRecord,
    create_longitudinal_dataloaders,
    load_manifest_csv,
)
from evaluation.metrics import (
    compute_classification_metrics,
    compute_counterfactual_drop,
)
from evaluation.visualize_xai import export_xai_nifti_volumes, plot_longitudinal_xai_overlays
from losses.composite_loss import CompositeDLCNetLoss
from models.full_model import BiTemporalSiameseDLCNet
from train import generate_synthetic_records, load_config, set_deterministic_seed


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """Compute (num_classes, num_classes) integer confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.ravel(), y_pred.ravel()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def compute_per_class_report(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> Dict[int, Dict[str, float]]:
    """Compute precision, recall, specificity, and F1-score per progression class."""
    report = {}
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    total_samples = float(np.sum(cm))

    for c in range(num_classes):
        tp = float(cm[c, c])
        fn = float(np.sum(cm[c, :])) - tp
        fp = float(np.sum(cm[:, c])) - tp
        tn = total_samples - (tp + fn + fp)

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        support = int(tp + fn)

        report[c] = {
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "support": support,
        }
    return report


@torch.no_grad()
def run_full_evaluation(
    model: BiTemporalSiameseDLCNet,
    dataloader: torch.utils.data.DataLoader,
    criterion: CompositeDLCNetLoss,
    device: torch.device,
    progression_classes: Tuple[int, ...] = (1, 2),
    output_xai_dir: Optional[Path] = None,
    max_visualizations: int = 3,
) -> Dict[str, Any]:
    """Execute complete factual & counterfactual evaluation across a test dataset split."""
    model.eval()

    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []
    all_s1: List[float] = []
    all_s2: List[float] = []
    all_delta: List[float] = []
    all_delta_s: List[float] = []
    all_drops_prog: List[float] = []
    all_drops_all: List[float] = []

    running_losses = {"total": 0.0, "ce": 0.0, "ranking": 0.0, "sparsity": 0.0}
    num_samples = 0
    vis_count = 0

    for batch_idx, (X_1, X_2, delta_t, labels) in enumerate(dataloader):
        B = int(X_1.shape[0])
        X_1 = X_1.to(device)
        X_2 = X_2.to(device)
        delta_t = delta_t.to(device)
        labels = labels.to(device)

        # 1. Factual Forward Pass
        factual_out = model(X_1, X_2, delta_t)

        loss_dict = criterion(
            logits=factual_out["logits"],
            targets=labels,
            s_1=factual_out["s_1"],
            s_2=factual_out["s_2"],
            M_1=factual_out["M_1"],
            M_2=factual_out["M_2"],
        )

        # 2. Counterfactual Forward Pass
        cf_out = model.forward_counterfactual(X_1, X_2, delta_t, factual_outputs=factual_out)

        # 3. Counterfactual Confidence Drop
        drop_stats = compute_counterfactual_drop(
            probs_factual=cf_out["probs_factual"],
            probs_cf=cf_out["probs_cf"],
            targets=labels,
            progression_classes=progression_classes,
        )

        probs_np = factual_out["probs"].detach().cpu().float().numpy()
        preds_np = np.argmax(probs_np, axis=1)
        targets_np = labels.detach().cpu().numpy()

        all_targets.extend(targets_np.tolist())
        all_preds.extend(preds_np.tolist())
        all_probs.append(probs_np)

        all_s1.extend(factual_out["s_1"].detach().cpu().view(-1).tolist())
        all_s2.extend(factual_out["s_2"].detach().cpu().view(-1).tolist())
        all_delta.extend(factual_out["delta"].detach().cpu().view(-1).tolist())
        all_delta_s.extend(factual_out["delta_s"].detach().cpu().view(-1).tolist())

        all_drops_prog.append(float(drop_stats["drop_cf_progression_mean"]))
        all_drops_all.append(float(drop_stats["drop_cf_all_mean"]))

        running_losses["total"] += float(loss_dict["loss_total"].item()) * B
        running_losses["ce"] += float(loss_dict["loss_ce"].item()) * B
        running_losses["ranking"] += float(loss_dict["loss_ranking"].item()) * B
        running_losses["sparsity"] += float(loss_dict["loss_sparsity"].item()) * B
        num_samples += B

        # 4. Optional XAI Visualizations
        if output_xai_dir is not None and vis_count < max_visualizations:
            for i in range(min(B, max_visualizations - vis_count)):
                pred_cls = LABEL_MAP.get(int(preds_np[i]), f"Class {preds_np[i]}")
                true_cls = LABEL_MAP.get(int(targets_np[i]), f"Class {targets_np[i]}")
                subj_id = f"test_sample_{vis_count:02d}_true_{true_cls}_pred_{pred_cls}"

                plot_longitudinal_xai_overlays(
                    X_1=X_1,
                    X_2=X_2,
                    M_1=factual_out["M_1"],
                    M_2=factual_out["M_2"],
                    output_path=output_xai_dir / f"{subj_id}_overlay.png",
                    subject_id=subj_id,
                    s_1=float(all_s1[-B + i]),
                    s_2=float(all_s2[-B + i]),
                    pred_label=f"Pred: {pred_cls} (True: {true_cls})",
                    batch_idx=i,
                )
                export_xai_nifti_volumes(
                    X_1=X_1,
                    X_2=X_2,
                    M_1=factual_out["M_1"],
                    M_2=factual_out["M_2"],
                    output_dir=output_xai_dir / "nifti_volumes",
                    subject_id=subj_id,
                    batch_idx=i,
                )
                vis_count += 1

    denom = max(num_samples, 1)
    y_true_arr = np.array(all_targets, dtype=np.int64)
    y_pred_arr = np.array(all_preds, dtype=np.int64)
    probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 3))

    # Calculate global classification factors
    cls_metrics = compute_classification_metrics(
        y_true=y_true_arr,
        y_probs=probs_arr,
        drop_cf_progression=float(np.mean(all_drops_prog)) if all_drops_prog else 0.0,
        drop_cf_all=float(np.mean(all_drops_all)) if all_drops_all else 0.0,
    )

    cm = compute_confusion_matrix(y_true_arr, y_pred_arr, num_classes=3)
    per_class_stats = compute_per_class_report(y_true_arr, y_pred_arr, num_classes=3)

    return {
        "num_samples": num_samples,
        "losses": {k: v / denom for k, v in running_losses.items()},
        "metrics": cls_metrics,
        "confusion_matrix": cm,
        "per_class": per_class_stats,
        "severity_dynamics": {
            "mean_s1": float(np.mean(all_s1)),
            "mean_s2": float(np.mean(all_s2)),
            "mean_delta": float(np.mean(all_delta)),
            "mean_delta_s": float(np.mean(all_delta_s)),
        },
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
    }


def print_evaluation_dashboard(results: Dict[str, Any]) -> None:
    """Print an ASCII dashboard summarizing all performance and explainability factors."""
    m = results["metrics"]
    losses = results["losses"]
    cm = results["confusion_matrix"]
    pc = results["per_class"]
    sev = results["severity_dynamics"]
    n = results["num_samples"]

    print("\n" + "=" * 78)
    print("      DLC-NET COMPREHENSIVE PERFORMANCE & EXPLAINABILITY REPORT      ")
    print("=" * 78)
    print(f"Total Evaluated Test Samples: {n}")
    print("-" * 78)

    print("1. PRIMARY CLASSIFICATION FACTORS:")
    print(f"   * Accuracy:              {m['accuracy'] * 100:6.2f}%")
    print(f"   * Macro-F1 Score:        {m['macro_f1'] * 100:6.2f}%")
    print(f"   * Cohen's Kappa (kappa): {m['cohens_kappa']:8.4f}  (Inter-rater agreement)")
    print(f"   * Multi-Class ROC-AUC:   {m['roc_auc']:8.4f}  (Macro One-vs-Rest)")
    print("-" * 78)

    print("2. CAUSAL EXPLAINABILITY FACTOR (Counterfactual Drop):")
    print(f"   * Drop_cf (Progression): {m['drop_cf']:+8.4f}")
    print(f"   * Drop_cf (All Classes): {m['drop_cf_all']:+8.4f}")
    if m["drop_cf"] > 0.05:
        verdict = "HIGH CAUSAL RELIANCE (Masked atrophy regions are clinically decisive)"
    elif m["drop_cf"] > 0.0:
        verdict = "MODERATE CAUSAL RELIANCE (Positive confidence shift)"
    else:
        verdict = "LOW CAUSAL RELIANCE (Model does not heavily rely on gated atrophy)"
    print(f"   * Explainability Status: {verdict}")
    print("-" * 78)

    print("3. MULTI-OBJECTIVE COMPOSITE LOSS FACTORS:")
    print(f"   * Total Loss (L_total):  {losses['total']:.4f}")
    print(f"   * Cross-Entropy (L_CE):  {losses['ce']:.4f}")
    print(f"   * Ranking Loss (L_rank): {losses['ranking']:.4f}  (Temporal consistency)")
    print(f"   * Mask Sparsity (L_sp):  {losses['sparsity']:.4f}  (Mean L1 mask volume)")
    print("-" * 78)

    print("4. LATENT SEVERITY DYNAMICS:")
    print(f"   * Baseline Severity (s_1):    {sev['mean_s1']:.4f} in [0, 1]")
    print(f"   * Follow-up Severity (s_2):   {sev['mean_s2']:.4f} in [0, 1]")
    print(f"   * Progression Delta (Delta):  {sev['mean_delta']:+.4f} (s_2 - s_1)")
    print(f"   * Progression Velocity (vel): {sev['mean_delta_s']:+.4f} (Delta / Delta_t per year)")
    print("-" * 78)

    print("5. PER-CLASS PERFORMANCE BREAKDOWN:")
    print(f"   {'Class Label':<16} {'Precision':<12} {'Recall':<12} {'Specificity':<12} {'F1-Score':<10} {'Support'}")
    print("   " + "-" * 70)
    for c in range(3):
        name = f"{c}: {LABEL_MAP[c]}"
        p = pc[c]["precision"] * 100
        r = pc[c]["recall"] * 100
        spec = pc[c]["specificity"] * 100
        f1 = pc[c]["f1"] * 100
        sup = pc[c]["support"]
        print(f"   {name:<16} {p:6.2f}%      {r:6.2f}%      {spec:6.2f}%       {f1:6.2f}%     {sup}")
    print("-" * 78)

    print("6. CONFUSION MATRIX (Rows: True, Columns: Predicted):")
    print(f"                  Pred: Stable(0)  Pred: Prog(1)  Pred: Adv(2)")
    for i in range(3):
        row_name = f"True: {LABEL_MAP[i]}({i})"
        print(f"   {row_name:<16}   {cm[i, 0]:<14}  {cm[i, 1]:<13}  {cm[i, 2]:<10}")
    print("=" * 78 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DLC-Net Performance Factors and Metrics")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model weights (.pt checkpoint).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Dataset split to evaluate on (default: 'test').",
    )
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Run evaluation on synthetic longitudinal MRI pairs without external data dependencies.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Export 3D NIfTI saliency volumes and 2D multi-planar overlay figures to visualizations directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./visualizations",
        help="Directory where visualization outputs will be saved.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("project", {}).get("seed", 42))
    set_deterministic_seed(seed)

    device_name = str(cfg.get("project", {}).get("device", "cuda"))
    device = torch.device("cuda" if (device_name == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"-> Target Device: {device}")

    # Prepare Data
    data_cfg = cfg.get("data", {})
    target_shape = tuple(int(x) for x in data_cfg.get("target_shape", [128, 128, 128]))

    if args.synthetic_demo:
        print("-> Using synthetic longitudinal MRI pairs for evaluation...")
        records = generate_synthetic_records(num_patients=20, spatial_shape=target_shape, seed=seed)
    else:
        manifest_csv = data_cfg.get("manifest_csv", "")
        if manifest_csv and Path(manifest_csv).exists():
            records = load_manifest_csv(manifest_csv)
        else:
            print("-> Manifest CSV not found; automatically using synthetic demo data...")
            records = generate_synthetic_records(num_patients=20, spatial_shape=target_shape, seed=seed)

    dataloaders = create_longitudinal_dataloaders(
        records=records,
        target_shape=target_shape,
        train_ratio=float(data_cfg.get("train_ratio", 0.70)),
        val_ratio=float(data_cfg.get("val_ratio", 0.15)),
        test_ratio=float(data_cfg.get("test_ratio", 0.15)),
        batch_size=int(data_cfg.get("batch_size", 2)),
        num_workers=0,
        pin_memory=False,
        augment_train=False,
        seed=seed,
    )
    eval_loader = dataloaders[args.split]

    # Instantiate Model
    model = BiTemporalSiameseDLCNet.from_config(cfg).to(device)

    # Load Weights if available
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        print(f"-> Loading model weights from: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
    else:
        print("-> No checkpoint specified; evaluating model in initialization state.")

    loss_cfg = cfg.get("loss", {})
    criterion = CompositeDLCNetLoss(
        class_weights=loss_cfg.get("class_weights", [1.0, 1.5, 2.0]),
        lambda_1=float(loss_cfg.get("lambda_1", 0.5)),
        lambda_2=float(loss_cfg.get("lambda_2", 0.05)),
        epsilon=float(loss_cfg.get("epsilon", 0.10)),
        epsilon_stable=float(loss_cfg.get("epsilon_stable", 0.05)),
    ).to(device)

    vis_dir = Path(args.output_dir) if args.visualize else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"-> Running evaluation on '{args.split}' split ({len(eval_loader.dataset)} scan pairs)...")
    results = run_full_evaluation(
        model=model,
        dataloader=eval_loader,
        criterion=criterion,
        device=device,
        output_xai_dir=vis_dir,
        max_visualizations=3,
    )

    print_evaluation_dashboard(results)

    if vis_dir:
        print(f"-> Visualizations successfully exported to: {vis_dir.resolve()}")


if __name__ == "__main__":
    main()
