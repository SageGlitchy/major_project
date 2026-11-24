"""Automated Verification and Acceptance Test Suite (`test_pipeline.py`).

Verifies all architectural, mathematical, and dimensional specifications in `ARGON_TASK_SPEC.md`:
1. Shape verification at full 128 x 128 x 128 resolution:
   Given dummy inputs `X_1, X_2 = torch.randn(2, 1, 128, 128, 128)` and
   `delta_t = torch.tensor([[1.0], [2.0]])`, verifies that all intermediate and output
   tensor shapes across Stage B, Stage C1, Stage C2, and Stage D match the specification.
2. Exact Siamese symmetry test:
   Verifies that when `X_1 == X_2`, `torch.allclose(z_1, z_2, atol=1e-6)` and
   `torch.allclose(s_1, s_2, atol=1e-6)` in `eval()` mode, and `delta == 0`, `delta_s == 0`,
   `delta_M == 0`.
3. Gradient flow and sparsity test:
   Performs one backward pass of `L_total.backward()` and asserts that parameters in
   `ante_hoc_gate`, `axial_attention_3d`, `siamese_backbone`, and `progression_head`
   all have non-zero, finite `.grad` tensors.
4. Counterfactual forward test:
   Verifies that `model.forward_counterfactual(X_1, X_2, delta_t)` executes cleanly and
   returns valid factual and counterfactual probability distributions and `Drop_cf`.
5. Subject-level zero-data-leakage split and dataset signature verification.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import (
    LongitudinalMRIDataset,
    ScanPairRecord,
    stratified_patient_split,
    verify_zero_data_leakage,
)
from evaluation.metrics import compute_counterfactual_drop
from losses.composite_loss import CompositeDLCNetLoss
from models.ante_hoc_gate import BiTemporalSaliencyGate
from models.axial_attention_3d import AxialAttention3D
from models.full_model import BiTemporalSiameseDLCNet
from models.progression_head import TimeAwareProgressionHead
from models.siamese_backbone import Siamese3DBackbone


def test_tensor_shapes_full_resolution() -> None:
    """Check 1: Verify all intermediate and output tensor shapes at full 128^3 resolution."""
    torch.manual_seed(42)
    B = 2
    X_1 = torch.randn(B, 1, 128, 128, 128)
    X_2 = torch.randn(B, 1, 128, 128, 128)
    delta_t = torch.tensor([[1.0], [2.0]], dtype=torch.float32)

    model = BiTemporalSiameseDLCNet(use_checkpoint=False)
    model.eval()

    with torch.no_grad():
        outputs = model(X_1, X_2, delta_t, return_intermediates=True)

    # Stage B: Bi-Temporal Ante-Hoc Saliency Gate shapes
    for key in ("X_gated_1", "X_gated_2", "X_cf_1", "X_cf_2", "M_1", "M_2", "delta_M"):
        assert key in outputs, f"Missing key {key} in model outputs"
        assert outputs[key].shape == (B, 1, 128, 128, 128), (
            f"Expected {key} shape {(B, 1, 128, 128, 128)}, got {outputs[key].shape}"
        )

    # Verify M_1, M_2 in (0, 1) and factual + counterfactual partition X_t
    assert torch.all((outputs["M_1"] > 0.0) & (outputs["M_1"] < 1.0))
    assert torch.all((outputs["M_2"] > 0.0) & (outputs["M_2"] < 1.0))
    assert torch.allclose(
        outputs["X_gated_1"] + outputs["X_cf_1"], X_1, atol=1e-5
    ), "X_gated_1 + X_cf_1 must reconstruct X_1"
    assert torch.allclose(
        outputs["X_gated_2"] + outputs["X_cf_2"], X_2, atol=1e-5
    ), "X_gated_2 + X_cf_2 must reconstruct X_2"

    # Stage C1 & C2: Siamese 3D Res-Axial Backbone intermediate and output shapes
    inter_t1 = outputs["backbone_intermediates"]["t1"]
    inter_t2 = outputs["backbone_intermediates"]["t2"]
    for inter in (inter_t1, inter_t2):
        assert inter["stem"].shape == (B, 16, 64, 64, 64), (
            f"Stem shape mismatch: {inter['stem'].shape}"
        )
        assert inter["stage1"].shape == (B, 32, 32, 32, 32), (
            f"Stage 1 shape mismatch: {inter['stage1'].shape}"
        )
        assert inter["stage2_pre_pool"].shape == (B, 64, 32, 32, 32), (
            f"Stage 2 AxialAttention3D shape mismatch: {inter['stage2_pre_pool'].shape}"
        )
        assert inter["stage2"].shape == (B, 64, 16, 16, 16), (
            f"Stage 2 output shape mismatch: {inter['stage2'].shape}"
        )
        assert inter["stage3_pre_pool"].shape == (B, 128, 16, 16, 16), (
            f"Stage 3 AxialAttention3D shape mismatch: {inter['stage3_pre_pool'].shape}"
        )
        assert inter["stage3"].shape == (B, 128, 8, 8, 8), (
            f"Stage 3 output shape mismatch: {inter['stage3'].shape}"
        )

    assert outputs["z_1"].shape == (B, 128), f"Expected z_1 (2, 128), got {outputs['z_1'].shape}"
    assert outputs["z_2"].shape == (B, 128), f"Expected z_2 (2, 128), got {outputs['z_2'].shape}"

    # Stage D: Time-Aware Differential Progression Head shapes
    assert outputs["s_1"].shape == (B, 1), f"Expected s_1 (2, 1), got {outputs['s_1'].shape}"
    assert outputs["s_2"].shape == (B, 1), f"Expected s_2 (2, 1), got {outputs['s_2'].shape}"
    assert outputs["delta"].shape == (B, 1), f"Expected delta (2, 1), got {outputs['delta'].shape}"
    assert outputs["delta_s"].shape == (B, 1), (
        f"Expected delta_s (2, 1), got {outputs['delta_s'].shape}"
    )
    assert outputs["h"].shape == (B, 258), (
        f"Expected 258-D differential fusion vector (2, 258), got {outputs['h'].shape}"
    )
    assert outputs["logits"].shape == (B, 3), (
        f"Expected logits (2, 3), got {outputs['logits'].shape}"
    )
    assert outputs["probs"].shape == (B, 3), (
        f"Expected probs (2, 3), got {outputs['probs'].shape}"
    )


def test_siamese_symmetry() -> None:
    """Check 2: Verify exact Siamese symmetry: X_1 == X_2 => z_1 == z_2 and s_1 == s_2 in eval() mode."""
    torch.manual_seed(123)
    B = 2
    X_1 = torch.randn(B, 1, 128, 128, 128)
    X_2 = X_1.clone()
    delta_t = torch.tensor([[1.0], [2.0]], dtype=torch.float32)

    model = BiTemporalSiameseDLCNet(use_checkpoint=False)
    model.eval()

    with torch.no_grad():
        outputs = model(X_1, X_2, delta_t)

    assert torch.allclose(outputs["M_1"], outputs["M_2"], atol=1e-6), (
        "When X_1 == X_2, saliency masks M_1 and M_2 must be identical."
    )
    assert torch.allclose(
        outputs["delta_M"], torch.zeros_like(outputs["delta_M"]), atol=1e-6
    ), "When X_1 == X_2, longitudinal atrophy map delta_M must be identically zero."
    assert torch.allclose(outputs["z_1"], outputs["z_2"], atol=1e-6), (
        "When X_1 == X_2, latent embeddings z_1 and z_2 must match within atol=1e-6."
    )
    assert torch.allclose(outputs["s_1"], outputs["s_2"], atol=1e-6), (
        "When X_1 == X_2, severity scores s_1 and s_2 must match within atol=1e-6."
    )
    assert torch.allclose(
        outputs["delta"], torch.zeros_like(outputs["delta"]), atol=1e-6
    ), "When X_1 == X_2, scalar progression delta must be zero."
    assert torch.allclose(
        outputs["delta_s"], torch.zeros_like(outputs["delta_s"]), atol=1e-6
    ), "When X_1 == X_2, progression velocity delta_s must be zero."


def test_gradient_flow_and_composite_loss() -> None:
    """Check 3: Perform one backward pass of L_total.backward() and assert that parameters in
    `ante_hoc_gate`, `axial_attention_3d`, `siamese_backbone`, and `progression_head` all
    receive non-zero, finite gradients."""
    torch.manual_seed(7)
    B = 2
    X_1 = torch.randn(B, 1, 128, 128, 128)
    X_2 = torch.randn(B, 1, 128, 128, 128)
    delta_t = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    labels = torch.tensor([0, 2], dtype=torch.long)

    model = BiTemporalSiameseDLCNet(use_checkpoint=True)
    model.train()

    criterion = CompositeDLCNetLoss(
        class_weights=[1.0, 1.5, 2.0],
        lambda_1=0.5,
        lambda_2=0.05,
        epsilon=0.10,
        epsilon_stable=0.05,
    )

    model.zero_grad()
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
    assert torch.isfinite(loss_total), f"L_total is not finite: {loss_total}"
    assert loss_total.item() > 0.0, f"Expected positive L_total, got {loss_total.item()}"
    assert loss_dict["loss_sparsity"].item() > 0.0, "L_sparsity must be strictly positive"

    loss_total.backward()

    module_groups = {
        "ante_hoc_gate": model.ante_hoc_gate,
        "axial_attention_3d_stage2": model.siamese_backbone.encoder.stage2_attn,
        "axial_attention_3d_stage3": model.siamese_backbone.encoder.stage3_attn,
        "siamese_backbone": model.siamese_backbone,
        "progression_head": model.progression_head,
    }

    for group_name, submodule in module_groups.items():
        params_with_grad = [
            (name, p) for name, p in submodule.named_parameters() if p.requires_grad
        ]
        assert len(params_with_grad) > 0, f"No trainable parameters found in {group_name}"

        grad_norms = []
        for param_name, param in params_with_grad:
            assert param.grad is not None, (
                f"Parameter '{param_name}' in '{group_name}' has None gradient!"
            )
            assert torch.all(torch.isfinite(param.grad)), (
                f"Parameter '{param_name}' in '{group_name}' has non-finite gradient values!"
            )
            grad_norms.append(float(torch.norm(param.grad).item()))

        total_group_norm = sum(grad_norms)
        assert total_group_norm > 0.0, (
            f"All gradients in '{group_name}' are identically zero (norm={total_group_norm})!"
        )


def test_counterfactual_forward_and_drop_metric() -> None:
    """Check 4: Verify `model.forward_counterfactual(X_1, X_2, delta_t)` and `Drop_cf` metric."""
    torch.manual_seed(99)
    B = 2
    X_1 = torch.randn(B, 1, 128, 128, 128)
    X_2 = torch.randn(B, 1, 128, 128, 128)
    delta_t = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    labels = torch.tensor([1, 2], dtype=torch.long)

    model = BiTemporalSiameseDLCNet(use_checkpoint=False)
    model.eval()

    with torch.no_grad():
        cf_out = model.forward_counterfactual(X_1, X_2, delta_t)

    assert "probs_factual" in cf_out and "probs_cf" in cf_out
    assert cf_out["probs_factual"].shape == (B, 3)
    assert cf_out["probs_cf"].shape == (B, 3)

    # Verify probability distributions sum to 1 along class dimension
    assert torch.allclose(
        torch.sum(cf_out["probs_factual"], dim=-1),
        torch.ones(B),
        atol=1e-5,
    )
    assert torch.allclose(
        torch.sum(cf_out["probs_cf"], dim=-1),
        torch.ones(B),
        atol=1e-5,
    )

    drop_stats = compute_counterfactual_drop(
        probs_factual=cf_out["probs_factual"],
        probs_cf=cf_out["probs_cf"],
        targets=labels,
        progression_classes=(1, 2),
    )
    assert drop_stats["drop_cf_per_sample"].shape == (B,)
    assert np.isfinite(drop_stats["drop_cf_progression_mean"])
    assert np.isfinite(drop_stats["drop_cf_all_mean"])


def test_patient_stratified_split_and_dataset_signature() -> None:
    """Check 5: Verify zero patient-level data leakage and LongitudinalMRIDataset return signature."""
    rng = np.random.default_rng(42)
    records = []
    # Create 20 patients, some with multiple longitudinal scan pairs
    for idx in range(20):
        ptid = f"PTID_{idx:03d}"
        label = idx % 3
        n_pairs = 2 if idx % 4 == 0 else 1
        for p in range(n_pairs):
            records.append(
                ScanPairRecord(
                    ptid=ptid,
                    baseline_path=None,
                    followup_path=None,
                    delta_t=1.0 + 0.5 * p,
                    label=label,
                    baseline_volume=rng.uniform(0.0, 1.0, size=(16, 16, 16)).astype(np.float32),
                    followup_volume=rng.uniform(0.0, 1.0, size=(16, 16, 16)).astype(np.float32),
                )
            )

    train_recs, val_recs, test_recs = stratified_patient_split(
        records, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )
    train_ptids = [r.ptid for r in train_recs]
    val_ptids = [r.ptid for r in val_recs]
    test_ptids = [r.ptid for r in test_recs]

    assert set(train_ptids).isdisjoint(set(val_ptids))
    assert set(train_ptids).isdisjoint(set(test_ptids))
    assert set(val_ptids).isdisjoint(set(test_ptids))
    assert verify_zero_data_leakage(train_ptids, val_ptids, test_ptids)

    ds = LongitudinalMRIDataset(train_recs[:2], target_shape=(128, 128, 128), augment=False)
    x1, x2, dt, y = ds[0]
    assert x1.shape == (1, 128, 128, 128)
    assert x2.shape == (1, 128, 128, 128)
    assert dt.shape == (1,)
    assert int(y.item()) in (0, 1, 2)


if __name__ == "__main__":
    print("=" * 78)
    print("Running Bi-Temporal Ante-Hoc Siamese 3D Res-Axial Network (DLC-Net) Pipeline Tests")
    print("=" * 78)

    tests = [
        ("1. Full 128^3 Resolution Tensor Shape Verification", test_tensor_shapes_full_resolution),
        ("2. Exact Siamese Symmetry Verification (X_1 == X_2)", test_siamese_symmetry),
        ("3. End-to-End Gradient Flow & Composite Loss Verification", test_gradient_flow_and_composite_loss),
        ("4. Counterfactual Forward Pass & Drop_cf Metric Verification", test_counterfactual_forward_and_drop_metric),
        ("5. Subject-Level Stratified Split & Dataset Return Signature", test_patient_stratified_split_and_dataset_signature),
    ]

    for name, fn in tests:
        print(f"-> Executing {name} ...", flush=True)
        fn()
        print(f"   [PASS] {name}")

    print("=" * 78)
    print("ALL DLC-NET PIPELINE ACCEPTANCE TESTS PASSED SUCCESSFULLY!")
    print("=" * 78)
