"""Module: Automated ADNI Longitudinal DICOM Preprocessing and Manifest Generator.

Converts raw ADNI DICOM slice series into standardized 3D volumes:
1. Pairs baseline and follow-up scans per patient (PTID) from `Siamese_dataset_10_23_2026.csv`.
2. Stacks 2D DICOM (.dcm) slices by slice location into 3D volumes (D, H, W).
3. Parses XML metadata for CDR and MMSE scores to determine clinical progression labels:
     0: Stable (CN healthy controls & stable MCI)
     1: Progressing (MCI with cognitive decline: delta CDR >= 0.5 or delta MMSE <= -2)
     2: Advanced (Alzheimer's Disease)
4. Executes 3D preprocessing:
     - Skull-stripping via positive Otsu thresholding + 3D morphological closing
     - Log-bias field correction (homomorphic 3D Gaussian filtering)
     - Rigid co-registration: center-of-mass and alignment of follow-up T2 -> baseline T1
     - Isotropic trilinear resampling to (128, 128, 128)
     - Percentile clipping and intensity normalization to [0.0, 1.0]
5. Saves clean 3D volumes as fast-loading NumPy binaries (.npy) and generates `manifest.csv`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pydicom

from data.preprocess_mri import (
    normalize_intensity,
    preprocess_longitudinal_pair,
    resample_isotropic_volume,
    save_mri_volume,
    skull_strip_mri,
)


def load_dicom_series_to_volume(dicom_dir: Path) -> np.ndarray:
    """Read all .dcm slice files in a directory and stack into a 3D float32 numpy array (D, H, W)."""
    dcm_paths = list(dicom_dir.glob("*.dcm"))
    if not dcm_paths:
        raise FileNotFoundError(f"No .dcm files found in {dicom_dir}")

    slices: List[pydicom.FileDataset] = []
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=False)
            slices.append(ds)
        except Exception:
            continue

    if not slices:
        raise ValueError(f"Failed to read DICOM slices from {dicom_dir}")

    # Sort slices spatially by SliceLocation or ImagePositionPatient[2] or InstanceNumber
    def get_slice_sort_key(s: pydicom.FileDataset) -> float:
        if hasattr(s, "ImagePositionPatient") and len(s.ImagePositionPatient) >= 3:
            return float(s.ImagePositionPatient[2])
        if hasattr(s, "SliceLocation"):
            return float(s.SliceLocation)
        if hasattr(s, "InstanceNumber"):
            return float(s.InstanceNumber)
        return 0.0

    slices.sort(key=get_slice_sort_key)
    volume_3d = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=0)
    return volume_3d


def parse_xml_clinical_metadata(metadata_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Parse all XML assessment files in metadata_dir into a lookup table keyed by (PTID, ImageUID)."""
    meta_lookup: Dict[str, Dict[str, Any]] = {}
    for xml_file in metadata_dir.glob("*.xml"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            subj_node = root.find(".//subjectIdentifier")
            subj_id = subj_node.text.strip() if (subj_node is not None and subj_node.text) else None

            img_node = root.find(".//imageUID")
            img_id = f"I{img_node.text.strip()}" if (img_node is not None and img_node.text) else None

            mmse_node = root.find('.//assessmentScore[@attribute="MMSCORE"]')
            mmse = float(mmse_node.text) if (mmse_node is not None and mmse_node.text) else None

            cdr_node = root.find('.//assessmentScore[@attribute="CDGLOBAL"]')
            cdr = float(cdr_node.text) if (cdr_node is not None and cdr_node.text) else None

            group_node = root.find(".//researchGroup")
            group = group_node.text.strip() if (group_node is not None and group_node.text) else None

            if subj_id and img_id:
                meta_lookup[f"{subj_id}_{img_id}"] = {
                    "ptid": subj_id,
                    "image_id": img_id,
                    "mmse": mmse,
                    "cdr": cdr,
                    "group": group,
                }
        except Exception:
            continue
    return meta_lookup


def find_dicom_folder_for_scan(adni_dir: Path, subject: str, image_id: str) -> Optional[Path]:
    """Locate the DICOM directory corresponding to a subject and image ID."""
    subj_dir = adni_dir / subject
    if not subj_dir.exists():
        return None

    # Search for directory ending or containing image_id
    matches = list(subj_dir.rglob(f"*{image_id}*"))
    dir_matches = [m for m in matches if m.is_dir() and list(m.glob("*.dcm"))]
    if dir_matches:
        return dir_matches[0]
    return None


def derive_longitudinal_pairs(
    csv_path: Path,
    adni_dir: Path,
    metadata_dir: Path,
) -> List[Dict[str, Any]]:
    """Group CSV rows by Subject, order chronologically, calculate delta_t and progression label."""
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    meta_lookup = parse_xml_clinical_metadata(metadata_dir)

    # Group by subject
    subj_to_rows: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        subj_to_rows.setdefault(r["Subject"].strip(), []).append(r)

    longitudinal_pairs: List[Dict[str, Any]] = []

    for ptid, scan_rows in sorted(subj_to_rows.items()):
        if len(scan_rows) < 2:
            continue

        # Sort chronologically by Acq Date
        sorted_scans = sorted(
            scan_rows,
            key=lambda x: datetime.strptime(x["Acq Date"].strip(), "%m/%d/%Y"),
        )
        r1, r2 = sorted_scans[0], sorted_scans[1]

        d1 = datetime.strptime(r1["Acq Date"].strip(), "%m/%d/%Y")
        d2 = datetime.strptime(r2["Acq Date"].strip(), "%m/%d/%Y")
        delta_t = (d2 - d1).days / 365.25

        if delta_t <= 0:
            delta_t = 0.5  # Fallback minimum interval

        # Find DICOM paths on disk
        img_id_1 = r1["Image Data ID"].strip()
        img_id_2 = r2["Image Data ID"].strip()
        dir_1 = find_dicom_folder_for_scan(adni_dir, ptid, img_id_1)
        dir_2 = find_dicom_folder_for_scan(adni_dir, ptid, img_id_2)

        if dir_1 is None or dir_2 is None:
            continue

        # Look up clinical scores from XML
        meta1 = meta_lookup.get(f"{ptid}_{img_id_1}", {})
        meta2 = meta_lookup.get(f"{ptid}_{img_id_2}", {})

        cdr1, cdr2 = meta1.get("cdr"), meta2.get("cdr")
        mmse1, mmse2 = meta1.get("mmse"), meta2.get("mmse")

        g1 = r1["Group"].strip()
        g2 = r2["Group"].strip()

        # Clinical trajectory assignment:
        # 0: Stable (CN healthy controls & stable MCI)
        # 1: Progressing (MCI showing measurable cognitive decline)
        # 2: Advanced (Alzheimer's Disease)
        if g1 == "AD" or g2 == "AD":
            label = 2
        elif g1 == "CN" and g2 == "CN":
            label = 0
        elif g1 == "MCI":
            d_cdr = (cdr2 - cdr1) if (cdr1 is not None and cdr2 is not None) else 0.0
            d_mmse = (mmse2 - mmse1) if (mmse1 is not None and mmse2 is not None) else 0.0
            if d_cdr >= 0.5 or d_mmse <= -2.0:
                label = 1
            else:
                label = 0
        else:
            label = 1

        longitudinal_pairs.append({
            "ptid": ptid,
            "dir_1": dir_1,
            "dir_2": dir_2,
            "img_id_1": img_id_1,
            "img_id_2": img_id_2,
            "delta_t": round(delta_t, 3),
            "label": label,
            "g1": g1,
            "g2": g2,
        })

    return longitudinal_pairs


def preprocess_single_patient_pair(
    pair_info: Dict[str, Any],
    output_dir: Path,
    target_shape: Tuple[int, int, int] = (128, 128, 128),
    save_nifti: bool = False,
) -> Optional[Dict[str, Any]]:
    """Process a single longitudinal DICOM pair, save 128^3 volume arrays, and return manifest entry."""
    ptid = pair_info["ptid"]
    dir_1: Path = pair_info["dir_1"]
    dir_2: Path = pair_info["dir_2"]

    t1_out_npy = output_dir / f"{ptid}_T1_128.npy"
    t2_out_npy = output_dir / f"{ptid}_T2_128.npy"

    try:
        # Check if already preprocessed
        if t1_out_npy.exists() and t2_out_npy.exists():
            return {
                "ptid": ptid,
                "baseline_path": str(t1_out_npy.resolve()),
                "followup_path": str(t2_out_npy.resolve()),
                "delta_t": pair_info["delta_t"],
                "label": pair_info["label"],
            }

        # 1. Load 3D DICOM series
        vol_1 = load_dicom_series_to_volume(dir_1)
        vol_2 = load_dicom_series_to_volume(dir_2)

        # 2. Run complete longitudinal preprocessing
        processed = preprocess_longitudinal_pair(
            t1_source=vol_1,
            t2_source=vol_2,
            target_shape=target_shape,
            apply_skull_strip=True,
            apply_n4=True,
            apply_registration=True,
            pair_id=ptid,
        )

        # Extract (128, 128, 128) float32 in [0, 1]
        x1_arr = processed["X_1"][0].astype(np.float32)
        x2_arr = processed["X_2"][0].astype(np.float32)

        # 3. Save as NumPy binary files (.npy) for instantaneous PyTorch loading
        np.save(str(t1_out_npy), x1_arr)
        np.save(str(t2_out_npy), x2_arr)

        if save_nifti:
            t1_out_nii = output_dir / f"{ptid}_T1_128.nii.gz"
            t2_out_nii = output_dir / f"{ptid}_T2_128.nii.gz"
            save_mri_volume(x1_arr, t1_out_nii)
            save_mri_volume(x2_arr, t2_out_nii)

        return {
            "ptid": ptid,
            "baseline_path": str(t1_out_npy.resolve()),
            "followup_path": str(t2_out_npy.resolve()),
            "delta_t": pair_info["delta_t"],
            "label": pair_info["label"],
        }
    except Exception as exc:
        print(f"Error processing pair for {ptid}: {exc}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ADNI Longitudinal DICOM Pairs for DLC-Net")
    parser.add_argument(
        "--raw-csv",
        type=str,
        default="dataset/Siamese_dataset_10_23_2026.csv",
        help="Path to Siamese_dataset_10_23_2026.csv",
    )
    parser.add_argument(
        "--adni-dir",
        type=str,
        default="dataset/ADNI",
        help="Root path to ADNI DICOM directories",
    )
    parser.add_argument(
        "--meta-dir",
        type=str,
        default="dataset/Metadata",
        help="Path to XML metadata directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/preprocessed_128",
        help="Directory to save preprocessed (128, 128, 128) volume files and manifest.csv",
    )
    parser.add_argument(
        "--target-shape",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help="Target 3D isotropic voxel grid shape (default: 128 128 128)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional limit on number of patient pairs to process",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel worker processes for preprocessing (default: 4)",
    )
    parser.add_argument(
        "--save-nifti",
        action="store_true",
        help="Save .nii.gz volumes alongside .npy files",
    )
    args = parser.parse_args()

    csv_path = Path(args.raw_csv)
    adni_dir = Path(args.adni_dir)
    meta_dir = Path(args.meta_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_shape = tuple(args.target_shape)

    print("=" * 78)
    print("      ADNI LONGITUDINAL DICOM PREPROCESSING & MANIFEST PIPELINE      ")
    print("=" * 78)
    print(f"Master CSV:     {csv_path.resolve()}")
    print(f"ADNI Dir:       {adni_dir.resolve()}")
    print(f"Metadata Dir:   {meta_dir.resolve()}")
    print(f"Output Dir:     {out_dir.resolve()}")
    print(f"Target Shape:   {target_shape}")
    print("-" * 78)

    print("-> Discovering and pairing longitudinal scan pairs...")
    pairs = derive_longitudinal_pairs(csv_path, adni_dir, meta_dir)
    print(f"-> Successfully paired {len(pairs)} subjects.")

    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]
        print(f"-> Limited to {len(pairs)} pairs by --max-pairs.")

    manifest_entries: List[Dict[str, Any]] = []
    total = len(pairs)
    completed = 0

    print(f"-> Processing {total} pairs with {args.num_workers} workers...")

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_pair = {
                executor.submit(
                    preprocess_single_patient_pair,
                    pair,
                    out_dir,
                    target_shape,
                    args.save_nifti,
                ): pair
                for pair in pairs
            }
            for future in as_completed(future_to_pair):
                entry = future.result()
                if entry is not None:
                    manifest_entries.append(entry)
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f"   [{completed:03d}/{total:03d}] pairs preprocessed ({completed/total*100:5.1f}%)", flush=True)
    else:
        for pair in pairs:
            entry = preprocess_single_patient_pair(pair, out_dir, target_shape, args.save_nifti)
            if entry is not None:
                manifest_entries.append(entry)
            completed += 1
            if completed % 5 == 0 or completed == total:
                print(f"   [{completed:03d}/{total:03d}] pairs preprocessed ({completed/total*100:5.1f}%)", flush=True)

    # Sort manifest entries by PTID
    manifest_entries.sort(key=lambda x: x["ptid"])

    manifest_csv = out_dir / "manifest.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ptid", "baseline_path", "followup_path", "delta_t", "label"]
        )
        writer.writeheader()
        for e in manifest_entries:
            writer.writerow(e)

    print("-" * 78)
    print(f"-> Preprocessing Complete! {len(manifest_entries)} / {total} pairs saved.")
    print(f"-> Generated Manifest CSV: {manifest_csv.resolve()}")

    label_counts = {0: 0, 1: 0, 2: 0}
    for e in manifest_entries:
        label_counts[e["label"]] = label_counts.get(e["label"], 0) + 1

    print("\nManifest Clinical Trajectory Class Distribution:")
    print(f"   * Class 0 (Stable):      {label_counts[0]:3d} pairs")
    print(f"   * Class 1 (Progressing): {label_counts[1]:3d} pairs")
    print(f"   * Class 2 (Advanced):    {label_counts[2]:3d} pairs")
    print("=" * 78)


if __name__ == "__main__":
    main()
