# Implementation Plan: ADNI Data Preprocessing, Training & Evaluation Workflow

## Goal Description
Transition from the raw downloaded ADNI longitudinal MRI dataset (`dataset/` folder containing 344 DICOM slice series, master CSV, and clinical XML assessments) to a fully trained **Bi-Temporal Ante-Hoc Siamese 3D Res-Axial Network (DLC-Net)**, generating final clinical classification accuracies, macro metrics, confusion matrices, and explainability visualizations.

---

## User Review Required

> [!IMPORTANT]
> **1. DICOM Reading & Dependencies**:
> The 344 MRI scans in `dataset/ADNI` are stored as folders of 2D DICOM (`.dcm`) slices (160–180 slices per scan). We need `pydicom` (or `SimpleITK`) installed in Python (`pip install pydicom nibabel`) to stack the slices into 3D volumes. Alternatively, we can use a built-in pure-Python fallback DICOM parser.
>
> **2. Preprocessed Volume Format**:
> We recommend saving preprocessed volumes as **NumPy binary arrays (`.npy`)** inside `data/preprocessed_128/`. NumPy arrays load 5–10x faster than `.nii.gz` during PyTorch dataloading, avoiding I/O bottlenecks during training.
>
> **3. Compute Hardware (CPU vs GPU)**:
> If CUDA is available, training will utilize PyTorch Automatic Mixed Precision (`torch.amp.autocast`) and gradient checkpointing. If running on CPU, we will configure smaller batch sizes (`batch_size = 2`) and appropriate worker threads.

---

## Open Questions

1. **GPU Acceleration**: Do you have an NVIDIA GPU available with CUDA configured on this machine, or will training run on CPU?
2. **Label Granularity**: The current 3-class target trajectory is:
   - `0: Stable` (73 CN + 37 stable MCI = 110 subjects)
   - `1: Progressing` (27 MCI subjects showing significant CDR increase / MMSE drop)
   - `2: Advanced` (35 diagnosed Alzheimer's Disease subjects)
   Does this 3-class clinical mapping match your thesis/project requirements?

---

## Proposed Workflow and Changes

```
Raw ADNI Data (dataset/)
  ├── Siamese_dataset_10_23_2026.csv
  ├── Metadata/ (344 XML clinical assessments)
  └── ADNI/ (344 folders of 160-180 DICOM slices)
       │
       ▼ [Step 1: prepare_adni_dataset.py]
  1. Parse DICOM slice series -> 3D Volume (D, H, W)
  2. N4ITK / Homomorphic bias correction
  3. Otsu foreground skull-stripping
  4. Rigid alignment (MNI152 & T2 -> T1)
  5. Isotropic resampling to (128, 128, 128)
  6. Extract CDR / MMSE deltas from XML -> trajectory labels (0, 1, 2)
       │
       ▼ [Step 2: Preprocessed Artifacts]
  data/preprocessed_128/
    ├── PTID_visit_T1_128.npy
    ├── PTID_visit_T2_128.npy
    └── manifest.csv  [ptid, baseline_path, followup_path, delta_t, label]
       │
       ▼ [Step 3: Model Training]
  train.py (Stratified Patient Split -> Zero-Leakage -> AMP Training -> Checkpoints)
       │
       ▼ [Step 4: Comprehensive Evaluation]
  evaluate.py (Accuracy, Macro-F1, Cohen's Kappa, ROC-AUC, Drop_cf, Confusion Matrix, XAI Overlays)
```

---

### Component 1: Data Preparation & Preprocessing Pipeline

#### `[NEW] data/prepare_adni_dataset.py`
A high-throughput conversion and preprocessing script that:
1. Parses `dataset/Siamese_dataset_10_23_2026.csv` to map subject IDs, acquisition dates, and image directory paths.
2. Reads 2D DICOM series (`.dcm`) using `pydicom` / `SimpleITK`, sorting slices by `ImagePositionPatient[2]` or `InstanceNumber` into true 3D spatial volumes.
3. Parses `dataset/Metadata/*.xml` to extract:
   - Clinical Dementia Rating (`CDGLOBAL`)
   - Mini-Mental State Exam (`MMSCORE`)
   - Computes $\Delta \text{CDR}$ and $\Delta \text{MMSE}$ to assign ground-truth trajectory label $y \in \{0, 1, 2\}$.
4. Calls `preprocess_longitudinal_pair()` from [data/preprocess_mri.py](file:///c:/Users/madha/Desktop/Major%20Project/data/preprocess_mri.py):
   - Skull-stripping non-brain voxels.
   - Low-frequency intensity inhomogeneity correction.
   - Rigid spatial co-registration between follow-up and baseline scans.
   - Resampling isotropically to $(128, 128, 128)$ and normalizing intensities to $[0, 1]$.
5. Saves clean 3D arrays to `data/preprocessed_128/` and generates the final `manifest.csv`.

---

### Component 2: Configuration & Dataloader Integration

#### `[MODIFY] configs/default_config.yaml`
- Update `manifest_csv: "./data/preprocessed_128/manifest.csv"`.
- Verify loss weights ($\lambda_1 = 0.5, \lambda_2 = 0.05, \epsilon = 0.10, \epsilon_{\text{stable}} = 0.05$).
- Adjust epochs, learning rate (`1e-4`), and checkpoint paths.

#### `[MODIFY] data/dataset.py`
- Ensure `_load_pair_arrays()` natively supports fast-loading `.npy` arrays without re-running preprocessing at training time.

---

### Component 3: Model Training & Validation Execution

#### `train.py`
- Execute training on the preprocessed 172 longitudinal pairs.
- Automated subject-level stratified splitting into Train (70% = ~120 pairs), Val (15% = ~26 pairs), and Test (15% = ~26 pairs).
- Strictly enforce `verify_zero_data_leakage(train_ptids, val_ptids, test_ptids)`.
- Train with AdamW and Cosine Annealing, saving:
  - `checkpoints/best_model.pt` (highest validation Macro-F1).
  - `checkpoints/last_model.pt`.

---

### Component 4: Final Evaluation Dashboard & Explainability Exports

#### `evaluate.py`
Run the final comprehensive evaluation on the independent **Test Split**:
1. **Classification Factors**:
   - Accuracy (%)
   - Macro-F1 Score (%)
   - Cohen's Kappa ($\kappa$)
   - Multi-class ROC-AUC (Macro One-vs-Rest)
2. **Causal Explainability Factor**:
   - Counterfactual Confidence Drop ($\text{Drop}_{\text{cf}}$):
     $$\text{Drop}_{\text{cf}} = P(Y = y \mid X_{\text{gated}}) - P(Y = y \mid X_{\text{cf}})$$
3. **Clinical Diagnostics**:
   - Confusion Matrix (Rows: True, Columns: Predicted)
   - Per-Class Breakdown (Precision, Recall, Specificity, F1)
   - Latent Disease Progression Dynamics ($s_1, s_2, \Delta, \Delta s$)
4. **XAI Visualizations**:
   - Multi-planar (Axial, Coronal, Sagittal) overlay figures superimposing Saliency Masks $M_1, M_2$ and Atrophy Map $|M_2 - M_1|$ saved in `visualizations/`.

---

## Verification Plan

### Phase 1: Preprocessing Verification
```bash
# 1. Install required imaging libraries (if not already installed)
pip install pydicom nibabel

# 2. Run dry-run test on 2 patient pairs to verify slice stacking and registration
python data/prepare_adni_dataset.py --max-pairs 2

# 3. Verify output tensor shapes and manifest CSV
python -c "
import numpy as np, csv
data = np.load('data/preprocessed_128/041_S_1368_bl_128.npy')
print('Tensor shape:', data.shape)
assert data.shape == (128, 128, 128)
print('Value range:', data.min(), 'to', data.max())
assert 0.0 <= data.min() and data.max() <= 1.0
"
```

### Phase 2: Full Preprocessing Run
```bash
# Process all 172 longitudinal pairs
python data/prepare_adni_dataset.py
```

### Phase 3: Zero-Leakage & Pipeline Acceptance Verification
```bash
python test_pipeline.py
```

### Phase 4: Training Execution
```bash
# Run 1-epoch smoke test to verify gradient backpropagation on real ADNI data
python train.py --epochs 1

# Run complete training pipeline
python train.py --config configs/default_config.yaml
```

### Phase 5: Final Evaluation & Diagnostics Dashboard
```bash
# Run final evaluation on test split and generate publication-quality XAI overlays
python evaluate.py --checkpoint checkpoints/best_model.pt --split test --visualize
```
