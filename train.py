"""End-to-End Mixed-Precision (AMP) Training, Validation, and Counterfactual Evaluation Script
for Bi-Temporal Ante-Hoc Siamese 3D Res-Axial Network (DLC-Net).

Features:
- Automatic Mixed Precision (`torch.amp.autocast` and `GradScaler`)
- Gradient checkpointing inside 3D Axial Attention blocks
- Subject-level stratified Train/Val/Test split with zero-leakage assertion
- Composite loss (`L_CE + lambda_1 * L_ranking + lambda_2 * L_sparsity`)
- Validation loop computing Accuracy, Macro-F1, Cohen's Kappa, ROC-AUC, and Counterfactual
  Confidence Drop (`Drop_cf`)
- Best and latest model checkpointing
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
from evaluation.visualize_xai import plot_longitudinal_xai_overlays
from losses.composite_loss import CompositeDLCNetLoss
from models.full_model import BiTemporalSiameseDLCNet


def set_deterministic_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_synthetic_records(
    num_patients: int = 20,
    spatial_shape: Tuple[int, int, int] = (128, 128, 128),
    seed: int = 42,
) -> List[ScanPairRecord]:
    """Generate synthetic longitudinal 3D MRI records grouped by PTID for smoke testing."""
    rng = np.random.default_rng(seed)
    records: List[ScanPairRecord] = []
    for i in range(num_patients):
        ptid = f"ADNI_PT_{i:04d}"
        label = int(i % 3)  # Balanced across {0: Stable, 1: Progressing, 2: Advanced}
        delta_t = float(rng.choice([0.5, 1.0, 1.5, 2.0]))
        vol_1 = rng.uniform(0.05, 0.85, size=spatial_shape).astype(np.float32)
        if label == 0:
            vol_2 = np.clip(vol_1 + rng.normal(0.0, 0.02, size=spatial_shape), 0.0, 1.0).astype(
                np.float32
            )
        else:
            atrophy = rng.uniform(0.0, 0.15 * label, size=spatial_shape).astype(np.float32)
            vol_2 = np.clip(vol_1 - atrophy, 0.0, 1.0).astype(np.float32)

        records.append(
            ScanPairRecord(
                ptid=ptid,
                baseline_path=None,
                followup_path=None,
                delta_t=delta_t,
                label=label,
                baseline_volume=vol_1,
                followup_volume=vol_2,
            )
        )
    return records


def train_one_epoch(
    model: BiTemporalSiameseDLCNet,
    dataloader: DataLoader,
    criterion: CompositeDLCNetLoss,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[Any],
    device: torch.device,
    use_amp: bool = True,
    grad_clip_norm: float = 1.0,
    epoch: int = 1,
    epochs: int = 100,
) -> Dict[str, float]:
    """Execute one training epoch with Automatic Mixed Precision (AMP) and gradient clipping."""
    model.train()
    running_total = 0.0
    running_ce = 0.0
    running_rank = 0.0
    running_sparse = 0.0
    num_samples = 0
    total_batches = len(dataloader)

    amp_enabled = bool(use_amp and device.type == "cuda")
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    for batch_idx, (X_1, X_2, delta_t, labels) in enumerate(dataloader, 1):
        batch_size = int(X_1.shape[0])
        X_1 = X_1.to(device, non_blocking=True)
        X_2 = X_2.to(device, non_blocking=True)
        delta_t = delta_t.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled
        ):
            outputs = model(X_1, X_2, delta_t)
            loss_dict = criterion(
                logits=outputs["logits"],
                targets=labels,
                s_1=outputs["s_1"],
                s_2=outputs["s_2"],
                M_1=outputs["M_1"],
                M_2=outputs["M_2"],
            )
            loss_total = loss_dict["loss_total"]

        if scaler is not None and amp_enabled:
            scaler.scale(loss_total).backward()
            if grad_clip_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_total.backward()
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        running_total += float(loss_total.item()) * batch_size
        running_ce += float(loss_dict["loss_ce"].item()) * batch_size
        running_rank += float(loss_dict["loss_ranking"].item()) * batch_size
        running_sparse += float(loss_dict["loss_sparsity"].item()) * batch_size
        num_samples += batch_size

        if batch_idx % 10 == 0 or batch_idx == total_batches:
            print(
                f"  [Epoch {epoch:03d}/{epochs:03d} | Batch {batch_idx:02d}/{total_batches:02d}] "
                f"Loss: {loss_total.item():.4f} (CE: {loss_dict['loss_ce'].item():.4f}, "
                f"Rank: {loss_dict['loss_ranking'].item():.4f})",
                flush=True,
            )

    denom = max(num_samples, 1)
    return {
        "loss_total": running_total / denom,
        "loss_ce": running_ce / denom,
        "loss_ranking": running_rank / denom,
        "loss_sparsity": running_sparse / denom,
    }


@torch.no_grad()
def validate(
    model: BiTemporalSiameseDLCNet,
    dataloader: DataLoader,
    criterion: CompositeDLCNetLoss,
    device: torch.device,
    use_amp: bool = True,
    progression_classes: Tuple[int, ...] = (1, 2),
) -> Dict[str, float]:
    """Evaluate model on validation/test split including Counterfactual Confidence Drop (`Drop_cf`)."""
    model.eval()
    running_total = 0.0
    running_ce = 0.0
    running_rank = 0.0
    running_sparse = 0.0
    num_samples = 0

    all_targets: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_drops_prog: List[float] = []
    all_drops_total: List[float] = []

    amp_enabled = bool(use_amp and device.type == "cuda")
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    for X_1, X_2, delta_t, labels in dataloader:
        batch_size = int(X_1.shape[0])
        X_1 = X_1.to(device, non_blocking=True)
        X_2 = X_2.to(device, non_blocking=True)
        delta_t = delta_t.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled
        ):
            factual_out = model(X_1, X_2, delta_t)
            loss_dict = criterion(
                logits=factual_out["logits"],
                targets=labels,
                s_1=factual_out["s_1"],
                s_2=factual_out["s_2"],
                M_1=factual_out["M_1"],
                M_2=factual_out["M_2"],
            )
            cf_out = model.forward_counterfactual(
                X_1, X_2, delta_t, factual_outputs=factual_out
            )

        drop_stats = compute_counterfactual_drop(
            probs_factual=cf_out["probs_factual"],
            probs_cf=cf_out["probs_cf"],
            targets=labels,
            progression_classes=progression_classes,
        )

        running_total += float(loss_dict["loss_total"].item()) * batch_size
        running_ce += float(loss_dict["loss_ce"].item()) * batch_size
        running_rank += float(loss_dict["loss_ranking"].item()) * batch_size
        running_sparse += float(loss_dict["loss_sparsity"].item()) * batch_size
        num_samples += batch_size

        all_targets.append(labels.detach().cpu().numpy())
        all_probs.append(factual_out["probs"].detach().cpu().float().numpy())
        all_drops_prog.append(float(drop_stats["drop_cf_progression_mean"]))
        all_drops_total.append(float(drop_stats["drop_cf_all_mean"]))

    denom = max(num_samples, 1)
    y_true_np = np.concatenate(all_targets, axis=0) if all_targets else np.array([], dtype=np.int64)
    y_probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 3))

    cls_metrics = compute_classification_metrics(
        y_true=y_true_np,
        y_probs=y_probs_np,
        drop_cf_progression=float(np.mean(all_drops_prog)) if all_drops_prog else 0.0,
        drop_cf_all=float(np.mean(all_drops_total)) if all_drops_total else 0.0,
    )

    return {
        "loss_total": running_total / denom,
        "loss_ce": running_ce / denom,
        "loss_ranking": running_rank / denom,
        "loss_sparsity": running_sparse / denom,
        **cls_metrics,
    }


def save_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
) -> Path:
    """Serialize model weights, optimizer state, metrics, and configuration."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        str(path),
    )
    return path


def run_training(
    config: Dict[str, Any],
    records: Optional[List[ScanPairRecord]] = None,
) -> Dict[str, Any]:
    """Run full training, validation, and test evaluation pipeline from configuration dict."""
    proj_cfg = config.get("project", {})
    data_cfg = config.get("data", {})
    loss_cfg = config.get("loss", {})
    train_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})

    seed = int(proj_cfg.get("seed", 42))
    set_deterministic_seed(seed)

    requested_device = str(proj_cfg.get("device", "cuda"))
    device = torch.device(
        "cuda" if (requested_device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    use_amp = bool(proj_cfg.get("use_amp", True))

    target_shape = tuple(int(x) for x in data_cfg.get("target_shape", [128, 128, 128]))

    if records is None:
        manifest_csv = data_cfg.get("manifest_csv", "")
        if manifest_csv and Path(manifest_csv).exists():
            records = load_manifest_csv(manifest_csv)
        else:
            records = generate_synthetic_records(
                num_patients=20, spatial_shape=target_shape, seed=seed
            )

    dataloaders = create_longitudinal_dataloaders(
        records=records,
        target_shape=target_shape,
        train_ratio=float(data_cfg.get("train_ratio", 0.70)),
        val_ratio=float(data_cfg.get("val_ratio", 0.15)),
        test_ratio=float(data_cfg.get("test_ratio", 0.15)),
        batch_size=int(data_cfg.get("batch_size", 2)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False) and device.type == "cuda"),
        augment_train=bool(data_cfg.get("augment_train", True)),
        seed=seed,
    )

    model = BiTemporalSiameseDLCNet.from_config(config).to(device)
    criterion = CompositeDLCNetLoss(
        class_weights=loss_cfg.get("class_weights", [1.0, 1.5, 2.0]),
        lambda_1=float(loss_cfg.get("lambda_1", 0.5)),
        lambda_2=float(loss_cfg.get("lambda_2", 0.05)),
        epsilon=float(loss_cfg.get("epsilon", 0.10)),
        epsilon_stable=float(loss_cfg.get("epsilon_stable", 0.05)),
        label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-2)),
    )
    epochs = int(train_cfg.get("epochs", 100))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=float(train_cfg.get("min_learning_rate", 1e-6)),
    )

    scaler = (
        torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
        if hasattr(torch.amp, "GradScaler")
        else None
    )

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "./checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    history: List[Dict[str, Any]] = []
    prog_classes = tuple(int(c) for c in eval_cfg.get("progression_classes", [1, 2]))
    patience = int(train_cfg.get("early_stopping_patience", 0))
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
            epoch=epoch,
            epochs=epochs,
        )
        scheduler.step()

        val_metrics = validate(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            progression_classes=prog_classes,
        )

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_record)

        save_checkpoint(
            ckpt_dir / "last_model.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=val_metrics,
            config=config,
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            epochs_no_improve = 0
            save_checkpoint(
                ckpt_dir / "best_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
            )
        else:
            epochs_no_improve += 1

        patience_str = f" | Patience: {epochs_no_improve}/{patience}" if patience > 0 else ""
        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"Train Loss: {train_metrics['loss_total']:.4f} | "
            f"Val Loss: {val_metrics['loss_total']:.4f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
            f"Val Drop_cf: {val_metrics['drop_cf']:+.4f}"
            f"{patience_str}"
        )

        if patience > 0 and epochs_no_improve >= patience:
            print(
                f"\n-> Early stopping triggered at epoch {epoch:03d}! "
                f"No validation Macro-F1 improvement for {patience} consecutive epochs."
            )
            break

    best_ckpt_path = ckpt_dir / "best_model.pt"
    if best_ckpt_path.exists():
        best_ckpt = torch.load(str(best_ckpt_path), map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        print(f"-> Reloaded best model checkpoint from epoch {best_ckpt.get('epoch')} for test evaluation.")

    test_metrics = validate(
        model=model,
        dataloader=dataloaders["test"],
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        progression_classes=prog_classes,
    )
    print(
        f"[Test Evaluation] Acc: {test_metrics['accuracy']:.4f} | "
        f"Macro-F1: {test_metrics['macro_f1']:.4f} | "
        f"Kappa: {test_metrics['cohens_kappa']:.4f} | "
        f"ROC-AUC: {test_metrics['roc_auc']:.4f} | "
        f"Drop_cf: {test_metrics['drop_cf']:+.4f}"
    )

    return {
        "model": model,
        "history": history,
        "test_metrics": test_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Bi-Temporal Ante-Hoc Siamese 3D Res-Axial Network (DLC-Net)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for number of training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Optional override for early stopping patience.",
    )
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Run a quick synthetic longitudinal training & evaluation smoke test.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.patience is not None:
        cfg.setdefault("training", {})["early_stopping_patience"] = int(args.patience)
    if args.synthetic_demo:
        cfg.setdefault("training", {})["epochs"] = min(
            int(cfg.get("training", {}).get("epochs", 2)), 2
        )
        cfg.setdefault("data", {})["num_workers"] = 0
    run_training(cfg)
