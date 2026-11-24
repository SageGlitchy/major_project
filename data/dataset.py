"""Module 1: Subject-Level Stratified Split and PyTorch LongitudinalMRIDataset.

Enforces:
- Strict grouping by PTID (Patient ID) before splitting into Train (70%), Val (15%), and Test (15%).
- Zero Data Leakage Rule:
    assert set(train_ptids).isdisjoint(set(val_ptids))
    assert set(train_ptids).isdisjoint(set(test_ptids))
    assert set(val_ptids).isdisjoint(set(test_ptids))
- Exact Dataset Return Signature:
    X_1: R^{1 x 128 x 128 x 128} (baseline scan)
    X_2: R^{1 x 128 x 128 x 128} (follow-up scan)
    delta_t: R^{1} (time interval t_2 - t_1 in years)
    label: Z in {0: Stable, 1: Progressing, 2: Advanced}
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .preprocess_mri import (
    load_mri_volume,
    normalize_intensity,
    preprocess_longitudinal_pair,
    resample_isotropic_volume,
)


LABEL_MAP: Dict[int, str] = {
    0: "Stable",
    1: "Progressing",
    2: "Advanced",
}


@dataclass
class ScanPairRecord:
    """Metadata record for a longitudinal pair of T1-weighted structural MRI scans."""

    ptid: str
    baseline_path: Optional[Union[str, Path]]
    followup_path: Optional[Union[str, Path]]
    delta_t: float
    label: int
    baseline_volume: Optional[np.ndarray] = None
    followup_volume: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.label not in (0, 1, 2):
            raise ValueError(
                f"Invalid progression label {self.label} for PTID={self.ptid}. "
                "Expected {0: Stable, 1: Progressing, 2: Advanced}."
            )
        if float(self.delta_t) <= 0.0:
            raise ValueError(
                f"Time interval delta_t must be positive (got {self.delta_t}) for PTID={self.ptid}."
            )


def verify_zero_data_leakage(
    train_ptids: Sequence[str],
    val_ptids: Sequence[str],
    test_ptids: Sequence[str],
) -> bool:
    """Assert that patient IDs across Train, Val, and Test splits are strictly disjoint."""
    train_set = set(train_ptids)
    val_set = set(val_ptids)
    test_set = set(test_ptids)

    assert train_set.isdisjoint(val_set), (
        f"Subject-level data leakage between train and val splits: {train_set & val_set}"
    )
    assert train_set.isdisjoint(test_set), (
        f"Subject-level data leakage between train and test splits: {train_set & test_set}"
    )
    assert val_set.isdisjoint(test_set), (
        f"Subject-level data leakage between val and test splits: {val_set & test_set}"
    )
    return True


def stratified_patient_split(
    records: Sequence[ Union[ScanPairRecord, Dict[str, Any]] ],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[ScanPairRecord], List[ScanPairRecord], List[ScanPairRecord]]:
    """Group all longitudinal scan pairs strictly by PTID and perform a stratified
    subject-level split into Train (70%), Val (15%), and Test (15%).

    Args:
        records: Sequence of ScanPairRecord objects or dicts with keys
                 ("ptid", "baseline_path", "followup_path", "delta_t", "label").
        train_ratio: Proportion of patients assigned to training set (default 0.70).
        val_ratio: Proportion of patients assigned to validation set (default 0.15).
        test_ratio: Proportion of patients assigned to test set (default 0.15).
        seed: Random seed for reproducible patient-level stratified splitting.

    Returns:
        Tuple of (train_records, val_records, test_records).
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-5):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio} + {val_ratio} + {test_ratio}"
        )

    parsed_records: List[ScanPairRecord] = []
    for item in records:
        if isinstance(item, ScanPairRecord):
            parsed_records.append(item)
        elif isinstance(item, dict):
            parsed_records.append(
                ScanPairRecord(
                    ptid=str(item["ptid"]),
                    baseline_path=item.get("baseline_path"),
                    followup_path=item.get("followup_path"),
                    delta_t=float(item["delta_t"]),
                    label=int(item["label"]),
                    baseline_volume=item.get("baseline_volume"),
                    followup_volume=item.get("followup_volume"),
                )
            )
        else:
            raise TypeError(f"Unsupported record type: {type(item)}")

    # Group all scan pairs strictly by PTID
    ptid_to_records: Dict[str, List[ScanPairRecord]] = {}
    for rec in parsed_records:
        ptid_to_records.setdefault(rec.ptid, []).append(rec)

    # Assign each patient a stratification label (highest severity label observed for that PTID)
    class_to_ptids: Dict[int, List[str]] = {0: [], 1: [], 2: []}
    for ptid, pt_records in sorted(ptid_to_records.items()):
        strat_label = max(r.label for r in pt_records)
        class_to_ptids[strat_label].append(ptid)

    rng = np.random.default_rng(seed)
    train_ptids: List[str] = []
    val_ptids: List[str] = []
    test_ptids: List[str] = []

    for label_cls, ptids in class_to_ptids.items():
        if not ptids:
            continue
        ptids_arr = np.array(ptids, dtype=object)
        rng.shuffle(ptids_arr)
        n_total = len(ptids_arr)

        if n_total >= 3:
            n_val = max(1, int(round(n_total * val_ratio)))
            n_test = max(1, int(round(n_total * test_ratio)))
            n_train = n_total - n_val - n_test
            if n_train < 1:
                n_train = 1
                n_val = max(1, (n_total - 1) // 2)
                n_test = n_total - n_train - n_val
        elif n_total == 2:
            n_train, n_val, n_test = 1, 1, 0
        else:
            n_train, n_val, n_test = 1, 0, 0

        train_ptids.extend(ptids_arr[:n_train].tolist())
        val_ptids.extend(ptids_arr[n_train : n_train + n_val].tolist())
        test_ptids.extend(ptids_arr[n_train + n_val :].tolist())

    # Enforce the Zero Data Leakage Rule
    verify_zero_data_leakage(train_ptids, val_ptids, test_ptids)

    train_records = [r for ptid in train_ptids for r in ptid_to_records[ptid]]
    val_records = [r for ptid in val_ptids for r in ptid_to_records[ptid]]
    test_records = [r for ptid in test_ptids for r in ptid_to_records[ptid]]

    # Verify again at the record level
    verify_zero_data_leakage(
        [r.ptid for r in train_records],
        [r.ptid for r in val_records],
        [r.ptid for r in test_records],
    )

    return train_records, val_records, test_records


def load_manifest_csv(manifest_path: Union[str, Path]) -> List[ScanPairRecord]:
    """Load longitudinal ADNI scan pair metadata from a CSV manifest file.

    Expected columns: ptid, baseline_path, followup_path, delta_t, label
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {path}")

    records: List[ScanPairRecord] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                ScanPairRecord(
                    ptid=str(row["ptid"]).strip(),
                    baseline_path=row["baseline_path"].strip(),
                    followup_path=row["followup_path"].strip(),
                    delta_t=float(row["delta_t"]),
                    label=int(row["label"]),
                )
            )
    return records


class LongitudinalMRIDataset(Dataset):
    """PyTorch Dataset for bi-temporal 3D structural T1-weighted MRI scan pairs.

    Returns:
        X_1: torch.FloatTensor of shape (1, 128, 128, 128) — baseline scan in [0, 1]
        X_2: torch.FloatTensor of shape (1, 128, 128, 128) — follow-up scan in [0, 1]
        delta_t: torch.FloatTensor of shape (1,) — inter-scan interval in years (t_2 - t_1)
        label: torch.LongTensor scalar — progression class in {0, 1, 2}
    """

    def __init__(
        self,
        records: Sequence[Union[ScanPairRecord, Dict[str, Any]]],
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        preprocess_on_fly: bool = False,
        mni_template_path: Optional[Union[str, Path]] = None,
        augment: bool = False,
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_dict: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.records: List[ScanPairRecord] = [
            r
            if isinstance(r, ScanPairRecord)
            else ScanPairRecord(
                ptid=str(r["ptid"]),
                baseline_path=r.get("baseline_path"),
                followup_path=r.get("followup_path"),
                delta_t=float(r["delta_t"]),
                label=int(r["label"]),
                baseline_volume=r.get("baseline_volume"),
                followup_volume=r.get("followup_volume"),
            )
            for r in records
        ]
        self.target_shape = tuple(target_shape)
        self.preprocess_on_fly = preprocess_on_fly
        self.mni_template_path = mni_template_path
        self.augment = augment
        self.transform = transform
        self.return_dict = return_dict
        self.rng = np.random.default_rng(seed)

    @property
    def ptids(self) -> List[str]:
        """Return list of unique patient IDs in this dataset split."""
        return sorted({r.ptid for r in self.records})

    def __len__(self) -> int:
        return len(self.records)

    def _load_pair_arrays(self, rec: ScanPairRecord) -> Tuple[np.ndarray, np.ndarray]:
        """Load or preprocess baseline and follow-up volumes to (1, D, H, W) float32 in [0, 1]."""
        if self.preprocess_on_fly:
            src_1 = rec.baseline_volume if rec.baseline_volume is not None else rec.baseline_path
            src_2 = rec.followup_volume if rec.followup_volume is not None else rec.followup_path
            if src_1 is None or src_2 is None:
                raise ValueError(f"Missing volume source for PTID={rec.ptid}")
            out = preprocess_longitudinal_pair(
                t1_source=src_1,
                t2_source=src_2,
                mni_template_source=self.mni_template_path,
                target_shape=self.target_shape,
            )
            return out["X_1"], out["X_2"]

        if rec.baseline_volume is not None:
            vol_1 = np.asarray(rec.baseline_volume, dtype=np.float32)
        elif rec.baseline_path is not None:
            vol_1, _, _ = load_mri_volume(rec.baseline_path)
        else:
            raise ValueError(f"Neither baseline_volume nor baseline_path provided for {rec.ptid}")

        if rec.followup_volume is not None:
            vol_2 = np.asarray(rec.followup_volume, dtype=np.float32)
        elif rec.followup_path is not None:
            vol_2, _, _ = load_mri_volume(rec.followup_path)
        else:
            raise ValueError(f"Neither followup_volume nor followup_path provided for {rec.ptid}")

        if vol_1.ndim == 4 and vol_1.shape[0] == 1:
            vol_1 = vol_1[0]
        if vol_2.ndim == 4 and vol_2.shape[0] == 1:
            vol_2 = vol_2[0]

        if tuple(vol_1.shape) != self.target_shape:
            vol_1 = resample_isotropic_volume(vol_1, target_shape=self.target_shape)
        if tuple(vol_2.shape) != self.target_shape:
            vol_2 = resample_isotropic_volume(vol_2, target_shape=self.target_shape)

        if vol_1.min() < 0.0 or vol_1.max() > 1.0:
            vol_1 = normalize_intensity(vol_1)
        if vol_2.min() < 0.0 or vol_2.max() > 1.0:
            vol_2 = normalize_intensity(vol_2)

        return vol_1[np.newaxis, ...].astype(np.float32), vol_2[np.newaxis, ...].astype(np.float32)

    def _apply_paired_augmentation(
        self, x1: np.ndarray, x2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply identical spatial transformations and intensity jitter to both timepoints."""
        # 1. Coupled sagittal (left-right) flip (axis=3 in (1, D, H, W))
        if self.rng.random() < 0.5:
            x1 = np.flip(x1, axis=3).copy()
            x2 = np.flip(x2, axis=3).copy()

        # 2. Coupled small 3D spatial translation jitter (-2 to +2 voxels)
        if self.rng.random() < 0.5:
            shift_d = int(self.rng.integers(-2, 3))
            shift_h = int(self.rng.integers(-2, 3))
            shift_w = int(self.rng.integers(-2, 3))
            if shift_d != 0 or shift_h != 0 or shift_w != 0:
                x1 = np.roll(x1, (shift_d, shift_h, shift_w), axis=(1, 2, 3))
                x2 = np.roll(x2, (shift_d, shift_h, shift_w), axis=(1, 2, 3))
                if shift_d > 0:
                    x1[:, :shift_d, :, :] = 0.0
                    x2[:, :shift_d, :, :] = 0.0
                elif shift_d < 0:
                    x1[:, shift_d:, :, :] = 0.0
                    x2[:, shift_d:, :, :] = 0.0
                if shift_h > 0:
                    x1[:, :, :shift_h, :] = 0.0
                    x2[:, :, :shift_h, :] = 0.0
                elif shift_h < 0:
                    x1[:, :, shift_h:, :] = 0.0
                    x2[:, :, shift_h:, :] = 0.0
                if shift_w > 0:
                    x1[:, :, :, :shift_w] = 0.0
                    x2[:, :, :, :shift_w] = 0.0
                elif shift_w < 0:
                    x1[:, :, :, shift_w:] = 0.0
                    x2[:, :, :, shift_w:] = 0.0

        # 3. Coupled brightness scaling
        if self.rng.random() < 0.4:
            scale = float(self.rng.uniform(0.94, 1.06))
            x1 = np.clip(x1 * scale, 0.0, 1.0).astype(np.float32)
            x2 = np.clip(x2 * scale, 0.0, 1.0).astype(np.float32)

        # 4. Coupled mild gamma/contrast scaling
        if self.rng.random() < 0.3:
            gamma = float(self.rng.uniform(0.92, 1.08))
            x1 = np.clip(np.power(np.maximum(x1, 0.0), gamma), 0.0, 1.0).astype(np.float32)
            x2 = np.clip(np.power(np.maximum(x2, 0.0), gamma), 0.0, 1.0).astype(np.float32)

        # 5. Coupled mild additive Gaussian noise
        if self.rng.random() < 0.3:
            sigma = float(self.rng.uniform(0.005, 0.015))
            noise = self.rng.normal(0.0, sigma, size=x1.shape).astype(np.float32)
            x1 = np.clip(x1 + noise, 0.0, 1.0)
            x2 = np.clip(x2 + noise, 0.0, 1.0)

        return x1, x2

    def __getitem__(
        self, index: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Dict[str, Union[torch.Tensor, str]],
    ]:
        rec = self.records[index]
        x1_np, x2_np = self._load_pair_arrays(rec)

        if self.augment:
            x1_np, x2_np = self._apply_paired_augmentation(x1_np, x2_np)

        x1_tensor = torch.as_tensor(x1_np, dtype=torch.float32)
        x2_tensor = torch.as_tensor(x2_np, dtype=torch.float32)

        if self.transform is not None:
            x1_tensor, x2_tensor = self.transform(x1_tensor, x2_tensor)

        delta_t_tensor = torch.tensor([float(rec.delta_t)], dtype=torch.float32)
        label_tensor = torch.tensor(int(rec.label), dtype=torch.long)

        if self.return_dict:
            return {
                "X_1": x1_tensor,
                "X_2": x2_tensor,
                "delta_t": delta_t_tensor,
                "label": label_tensor,
                "ptid": rec.ptid,
            }
        return x1_tensor, x2_tensor, delta_t_tensor, label_tensor


def create_longitudinal_dataloaders(
    records: Sequence[Union[ScanPairRecord, Dict[str, Any]]],
    target_shape: Tuple[int, int, int] = (128, 128, 128),
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 2,
    num_workers: int = 0,
    pin_memory: bool = False,
    augment_train: bool = True,
    preprocess_on_fly: bool = False,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """Create subject-stratified, zero-leakage Train, Val, and Test DataLoaders."""
    train_recs, val_recs, test_recs = stratified_patient_split(
        records=records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_ds = LongitudinalMRIDataset(
        train_recs,
        target_shape=target_shape,
        preprocess_on_fly=preprocess_on_fly,
        augment=augment_train,
        seed=seed,
    )
    val_ds = LongitudinalMRIDataset(
        val_recs,
        target_shape=target_shape,
        preprocess_on_fly=preprocess_on_fly,
        augment=False,
        seed=seed,
    )
    test_ds = LongitudinalMRIDataset(
        test_recs,
        target_shape=target_shape,
        preprocess_on_fly=preprocess_on_fly,
        augment=False,
        seed=seed,
    )

    verify_zero_data_leakage(train_ds.ptids, val_ds.ptids, test_ds.ptids)

    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
