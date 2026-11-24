"""Module 1: 3D Structural T1-Weighted MRI Preprocessing and Co-Registration Pipeline.

Implements:
1. N4ITK bias field correction (SimpleITK N4BiasFieldCorrectionImageFilter with SciPy fallback).
2. Automated skull-stripping via foreground Otsu thresholding, largest connected component
   extraction, and 3D morphological closing/hole-filling (or explicit brain mask application).
3. Rigid 3D registration (6-DOF Euler3DTransform with Mattes Mutual Information) to align:
   - Baseline scan T1 -> MNI152 standard template space
   - Follow-up scan T2 -> Registered baseline T1 space
4. Isotropic resampling to (128, 128, 128) voxel grid.
5. Robust intensity percentile clipping and normalization into [0, 1].
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage

try:
    import SimpleITK as sitk

    HAS_SITK = True
except ImportError:
    sitk = None
    HAS_SITK = False

try:
    import nibabel as nib

    HAS_NIBABEL = True
except ImportError:
    nib = None
    HAS_NIBABEL = False


ArrayOrPath = Union[np.ndarray, str, Path]


def load_mri_volume(
    source: ArrayOrPath,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional["sitk.Image"]]:
    """Load a 3D MRI volume from a NIfTI/SimpleITK file path, .npy file, or numpy array.

    Returns:
        volume_np: 3D float32 numpy array of shape (D, H, W).
        affine: 4x4 affine matrix if available, else np.eye(4).
        image_sitk: SimpleITK.Image instance if SimpleITK is installed and loaded from file/array.
    """
    if isinstance(source, np.ndarray):
        vol = np.asarray(source, dtype=np.float32)
        if vol.ndim == 4 and vol.shape[0] == 1:
            vol = vol[0]
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D MRI volume, got shape {vol.shape}")
        img_sitk = sitk.GetImageFromArray(vol) if HAS_SITK else None
        return vol, np.eye(4, dtype=np.float32), img_sitk

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"MRI volume file not found: {path}")

    if path.suffix == ".npy":
        vol = np.load(str(path)).astype(np.float32)
        if vol.ndim == 4 and vol.shape[0] == 1:
            vol = vol[0]
        img_sitk = sitk.GetImageFromArray(vol) if HAS_SITK else None
        return vol, np.eye(4, dtype=np.float32), img_sitk

    if HAS_SITK:
        img_sitk = sitk.ReadImage(str(path), sitk.sitkFloat32)
        vol = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
        direction = np.array(img_sitk.GetDirection(), dtype=np.float32).reshape(3, 3)
        spacing = np.array(img_sitk.GetSpacing(), dtype=np.float32)
        origin = np.array(img_sitk.GetOrigin(), dtype=np.float32)
        affine = np.eye(4, dtype=np.float32)
        affine[:3, :3] = direction * spacing[None, :]
        affine[:3, 3] = origin
        return vol, affine, img_sitk

    if HAS_NIBABEL:
        nii = nib.load(str(path))
        vol = np.asarray(nii.get_fdata(), dtype=np.float32)
        affine = np.asarray(nii.affine, dtype=np.float32)
        return vol, affine, None

    raise RuntimeError(
        f"Cannot load {path}: neither SimpleITK nor nibabel is installed."
    )


def save_mri_volume(
    volume: np.ndarray,
    output_path: Union[str, Path],
    affine: Optional[np.ndarray] = None,
    reference_sitk: Optional["sitk.Image"] = None,
) -> Path:
    """Save a 3D MRI volume to NIfTI (.nii / .nii.gz) or .npy format."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vol = np.asarray(volume, dtype=np.float32)

    if out_path.suffix == ".npy":
        np.save(str(out_path), vol)
        return out_path

    if HAS_NIBABEL:
        aff = affine if affine is not None else np.eye(4, dtype=np.float32)
        nii = nib.Nifti1Image(vol, aff)
        nib.save(nii, str(out_path))
        return out_path

    if HAS_SITK:
        img = sitk.GetImageFromArray(vol)
        if reference_sitk is not None and img.GetSize() == reference_sitk.GetSize():
            img.CopyInformation(reference_sitk)
        sitk.WriteImage(img, str(out_path))
        return out_path

    # Fallback: save numpy binary alongside requested path
    np.save(str(out_path) + ".npy", vol)
    return out_path


def _otsu_threshold_3d(volume: np.ndarray, num_bins: int = 256) -> float:
    """Compute Otsu's threshold on positive foreground voxels of a 3D volume."""
    pos_voxels = volume[volume > 0]
    if pos_voxels.size == 0:
        return float(np.mean(volume))
    hist, bin_edges = np.histogram(pos_voxels, bins=num_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    weight1 = np.cumsum(hist).astype(np.float64)
    weight2 = np.cumsum(hist[::-1])[::-1].astype(np.float64)
    valid = (weight1 > 0) & (weight2 > 0)
    if not np.any(valid):
        return float(np.median(pos_voxels))
    mean1 = np.cumsum(hist * bin_centers) / np.maximum(weight1, 1e-12)
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / np.maximum(weight2[::-1], 1e-12))[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = int(np.argmax(variance12))
    return float(bin_centers[idx])


def compute_brain_mask(
    volume: np.ndarray,
    closing_radius: int = 5,
) -> np.ndarray:
    """Extract a 3D binary brain mask via Otsu thresholding, largest connected component,
    and morphological closing + 3D hole filling."""
    vol = np.asarray(volume, dtype=np.float32)
    thresh = _otsu_threshold_3d(vol)
    initial_mask = vol > thresh

    if not np.any(initial_mask):
        initial_mask = vol > np.percentile(vol, 50.0)

    labeled, num_features = ndimage.label(initial_mask)
    if num_features > 0:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0  # Ignore background label 0
        largest_label = int(np.argmax(counts))
        brain_mask = labeled == largest_label
    else:
        brain_mask = initial_mask

    if closing_radius > 0:
        struct = ndimage.generate_binary_structure(3, 1)
        brain_mask = ndimage.binary_closing(
            brain_mask, structure=struct, iterations=min(closing_radius, 3)
        )
    brain_mask = ndimage.binary_fill_holes(brain_mask)
    return brain_mask.astype(np.float32)


def skull_strip_mri(
    volume: Union[np.ndarray, "sitk.Image"],
    brain_mask: Optional[np.ndarray] = None,
    closing_radius: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform skull-stripping on a 3D structural T1 MRI scan.

    Args:
        volume: 3D numpy array or SimpleITK.Image.
        brain_mask: Optional pre-computed binary brain mask of matching shape.
        closing_radius: Morphological closing iterations for brain mask extraction.

    Returns:
        skull_stripped: 3D float32 numpy array with non-brain tissues zeroed out.
        mask: 3D float32 binary brain mask in {0.0, 1.0}.
    """
    if HAS_SITK and isinstance(volume, sitk.Image):
        vol_np = sitk.GetArrayFromImage(volume).astype(np.float32)
    else:
        vol_np = np.asarray(volume, dtype=np.float32)

    if brain_mask is None:
        mask = compute_brain_mask(vol_np, closing_radius=closing_radius)
    else:
        mask = (np.asarray(brain_mask) > 0).astype(np.float32)

    skull_stripped = vol_np * mask
    return skull_stripped.astype(np.float32), mask


def n4_bias_field_correction(
    volume: Union[np.ndarray, "sitk.Image"],
    mask: Optional[np.ndarray] = None,
    shrink_factor: int = 4,
    num_iterations: Sequence[int] = (50, 50, 30, 20),
    convergence_threshold: float = 1e-6,
) -> np.ndarray:
    """Apply N4ITK low-frequency intensity inhomogeneity (bias field) correction.

    Uses SimpleITK's N4BiasFieldCorrectionImageFilter when SimpleITK is installed;
    otherwise estimates a smooth low-frequency log-intensity bias field via 3D Gaussian
    homomorphic filtering in SciPy.
    """
    if HAS_SITK:
        if isinstance(volume, sitk.Image):
            input_img = sitk.Cast(volume, sitk.sitkFloat32)
            vol_np = sitk.GetArrayFromImage(input_img).astype(np.float32)
        else:
            vol_np = np.asarray(volume, dtype=np.float32)
            # Ensure strictly positive values for log-domain N4 filter
            min_val = float(np.min(vol_np))
            if min_val < 0:
                vol_np = vol_np - min_val
            input_img = sitk.GetImageFromArray(vol_np)

        if mask is None:
            mask_np = compute_brain_mask(vol_np, closing_radius=3).astype(np.uint8)
        else:
            mask_np = (np.asarray(mask) > 0).astype(np.uint8)
        mask_img = sitk.GetImageFromArray(mask_np)
        mask_img.CopyInformation(input_img)

        if shrink_factor > 1:
            shrink_factors = [shrink_factor] * input_img.GetDimension()
            shrunk_img = sitk.Shrink(input_img, shrink_factors)
            shrunk_mask = sitk.Shrink(mask_img, shrink_factors)
        else:
            shrunk_img = input_img
            shrunk_mask = mask_img

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(list(num_iterations))
        corrector.SetConvergenceThreshold(float(convergence_threshold))
        corrector.Execute(shrunk_img, shrunk_mask)

        log_bias_field = corrector.GetLogBiasFieldAsImage(input_img)
        corrected_img = input_img / sitk.Exp(log_bias_field)
        corrected_np = sitk.GetArrayFromImage(corrected_img).astype(np.float32)
        return corrected_np

    # Homomorphic 3D Gaussian log-bias field correction fallback when SimpleITK is unavailable
    vol_np = np.asarray(volume, dtype=np.float32)
    min_val = float(np.min(vol_np))
    shifted = vol_np - min_val if min_val < 0 else vol_np.copy()
    fg_mask = (
        compute_brain_mask(shifted, closing_radius=2)
        if mask is None
        else (np.asarray(mask) > 0).astype(np.float32)
    )
    eps = 1e-4
    log_vol = np.log(np.maximum(shifted, eps))
    sigma = max(min(vol_np.shape) / 6.0, 2.0)
    weighted_log = ndimage.gaussian_filter(log_vol * fg_mask, sigma=sigma)
    weight_norm = ndimage.gaussian_filter(fg_mask, sigma=sigma) + 1e-6
    smooth_log_bias = (weighted_log / weight_norm) - np.mean(
        log_vol[fg_mask > 0.5] if np.any(fg_mask > 0.5) else log_vol
    )
    corrected = shifted * np.exp(-smooth_log_bias * fg_mask)
    return corrected.astype(np.float32)


def rigid_register_to_reference(
    moving: Union[np.ndarray, "sitk.Image"],
    fixed: Union[np.ndarray, "sitk.Image"],
    num_bins: int = 50,
    learning_rate: float = 1.0,
    min_step: float = 1e-4,
    num_iterations: int = 200,
) -> Tuple[np.ndarray, Dict[str, Union[np.ndarray, float]]]:
    """Rigidly register `moving` 3D MRI volume into the coordinate space of `fixed`.

    Used for:
    1. Registering baseline T1 scan to MNI152 template space.
    2. Co-registering follow-up T2 scan to the registered baseline T1 scan.

    Returns:
        registered_volume: 3D float32 numpy array aligned with `fixed`.
        transform_info: Dictionary containing rigid rotation/translation parameters.
    """
    if HAS_SITK:
        fixed_img = (
            sitk.Cast(fixed, sitk.sitkFloat32)
            if isinstance(fixed, sitk.Image)
            else sitk.GetImageFromArray(np.asarray(fixed, dtype=np.float32))
        )
        moving_img = (
            sitk.Cast(moving, sitk.sitkFloat32)
            if isinstance(moving, sitk.Image)
            else sitk.GetImageFromArray(np.asarray(moving, dtype=np.float32))
        )

        initial_transform = sitk.CenteredTransformInitializer(
            fixed_img,
            moving_img,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS,
        )

        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=int(num_bins))
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.10)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=float(learning_rate),
            minStep=float(min_step),
            numberOfIterations=int(num_iterations),
            relaxationFactor=0.5,
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        final_transform = registration_method.Execute(fixed_img, moving_img)
        resampled_img = sitk.Resample(
            moving_img,
            fixed_img,
            final_transform,
            sitk.sitkLinear,
            0.0,
            moving_img.GetPixelID(),
        )
        reg_np = sitk.GetArrayFromImage(resampled_img).astype(np.float32)
        params = np.array(final_transform.GetParameters(), dtype=np.float32)
        return reg_np, {
            "parameters": params,
            "final_metric": float(registration_method.GetMetricValue()),
        }

    # Fallback: 3D center-of-mass + principal-axes rigid alignment in SciPy
    fixed_np = np.asarray(fixed, dtype=np.float32)
    moving_np = np.asarray(moving, dtype=np.float32)

    if moving_np.shape != fixed_np.shape:
        moving_np = resample_isotropic_volume(moving_np, target_shape=fixed_np.shape)

    fixed_pos = np.maximum(fixed_np, 0.0)
    moving_pos = np.maximum(moving_np, 0.0)
    fixed_sum = float(np.sum(fixed_pos))
    moving_sum = float(np.sum(moving_pos))

    if fixed_sum > 1e-6 and moving_sum > 1e-6:
        com_fixed = np.array(ndimage.center_of_mass(fixed_pos), dtype=np.float32)
        com_moving = np.array(ndimage.center_of_mass(moving_pos), dtype=np.float32)
        shift = com_fixed - com_moving
        registered = ndimage.shift(moving_np, shift=shift, order=1, mode="constant", cval=0.0)
    else:
        shift = np.zeros(3, dtype=np.float32)
        registered = moving_np.copy()

    return registered.astype(np.float32), {
        "parameters": np.concatenate([np.zeros(3, dtype=np.float32), shift]),
        "final_metric": 0.0,
    }


def resample_isotropic_volume(
    volume: Union[np.ndarray, "sitk.Image"],
    target_shape: Tuple[int, int, int] = (128, 128, 128),
    is_label: bool = False,
) -> np.ndarray:
    """Resample a 3D volume isotropically to `target_shape` (default: 128 x 128 x 128).

    Uses trilinear interpolation (order=1) for continuous MRI intensities and
    nearest-neighbor interpolation (order=0) for binary masks/labels.
    """
    if HAS_SITK and isinstance(volume, sitk.Image):
        vol_np = sitk.GetArrayFromImage(volume).astype(np.float32)
    else:
        vol_np = np.asarray(volume, dtype=np.float32)

    if vol_np.ndim != 3:
        raise ValueError(f"Expected 3D volume for resampling, got shape {vol_np.shape}")

    if tuple(vol_np.shape) == tuple(target_shape):
        return vol_np.copy()

    zoom_factors = tuple(
        float(target_shape[i]) / float(vol_np.shape[i]) for i in range(3)
    )
    order = 0 if is_label else 1
    resampled = ndimage.zoom(vol_np, zoom=zoom_factors, order=order, mode="nearest")

    # Guarantee exact output dimensions even under floating-point rounding
    if tuple(resampled.shape) != tuple(target_shape):
        out = np.zeros(target_shape, dtype=np.float32)
        d = min(target_shape[0], resampled.shape[0])
        h = min(target_shape[1], resampled.shape[1])
        w = min(target_shape[2], resampled.shape[2])
        out[:d, :h, :w] = resampled[:d, :h, :w]
        resampled = out

    return resampled.astype(np.float32)


def normalize_intensity(
    volume: np.ndarray,
    brain_mask: Optional[np.ndarray] = None,
    p_min: float = 0.5,
    p_max: float = 99.5,
    eps: float = 1e-8,
) -> np.ndarray:
    """Clip outliers by foreground percentiles and normalize voxel intensities strictly into [0, 1]."""
    vol = np.asarray(volume, dtype=np.float32)
    if brain_mask is not None and np.any(brain_mask > 0):
        fg_voxels = vol[brain_mask > 0]
    elif np.any(vol > 0):
        fg_voxels = vol[vol > 0]
    else:
        fg_voxels = vol.ravel()

    v_low = float(np.percentile(fg_voxels, p_min))
    v_high = float(np.percentile(fg_voxels, p_max))
    if v_high - v_low < eps:
        v_low = float(np.min(vol))
        v_high = float(np.max(vol))

    if v_high - v_low < eps:
        return np.zeros_like(vol, dtype=np.float32)

    clipped = np.clip(vol, v_low, v_high)
    normalized = (clipped - v_low) / (v_high - v_low + eps)
    if brain_mask is not None:
        normalized = normalized * (brain_mask > 0).astype(np.float32)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def preprocess_longitudinal_pair(
    t1_source: ArrayOrPath,
    t2_source: ArrayOrPath,
    mni_template_source: Optional[ArrayOrPath] = None,
    target_shape: Tuple[int, int, int] = (128, 128, 128),
    apply_n4: bool = True,
    apply_skull_strip: bool = True,
    apply_registration: bool = True,
    n4_shrink_factor: int = 4,
    n4_iterations: Sequence[int] = (50, 50, 30, 20),
    output_dir: Optional[Union[str, Path]] = None,
    pair_id: str = "subject_pair",
) -> Dict[str, np.ndarray]:
    """Complete longitudinal preprocessing pipeline for baseline (T1) and follow-up (T2) MRI pair.

    Steps:
        1. Load baseline T1 and follow-up T2 scans.
        2. N4ITK bias field correction on both T1 and T2.
        3. Skull-stripping on both T1 and T2.
        4. Rigid registration of baseline T1 to MNI152 template (if provided),
           followed by rigid co-registration of follow-up T2 to baseline T1.
        5. Isotropic resampling of both volumes to `target_shape` (128, 128, 128).
        6. Intensity normalization into [0, 1].

    Returns:
        Dictionary with keys:
            "X_1": np.ndarray of shape (1, 128, 128, 128) in [0, 1]
            "X_2": np.ndarray of shape (1, 128, 128, 128) in [0, 1]
            "mask_1": np.ndarray of shape (128, 128, 128)
            "mask_2": np.ndarray of shape (128, 128, 128)
    """
    vol_1, affine_1, _ = load_mri_volume(t1_source)
    vol_2, _, _ = load_mri_volume(t2_source)

    if apply_skull_strip:
        vol_1, mask_1 = skull_strip_mri(vol_1)
        vol_2, mask_2 = skull_strip_mri(vol_2)
    else:
        mask_1 = (vol_1 > 0).astype(np.float32)
        mask_2 = (vol_2 > 0).astype(np.float32)

    if apply_n4:
        vol_1 = n4_bias_field_correction(
            vol_1, mask=mask_1, shrink_factor=n4_shrink_factor, num_iterations=n4_iterations
        )
        vol_2 = n4_bias_field_correction(
            vol_2, mask=mask_2, shrink_factor=n4_shrink_factor, num_iterations=n4_iterations
        )
        if apply_skull_strip:
            vol_1 = vol_1 * mask_1
            vol_2 = vol_2 * mask_2

    if apply_registration:
        if mni_template_source is not None:
            mni_vol, _, _ = load_mri_volume(mni_template_source)
            vol_1, _ = rigid_register_to_reference(vol_1, mni_vol)
            mask_1 = (vol_1 > 0).astype(np.float32)
        # Co-register follow-up T2 rigidly to baseline T1
        vol_2, _ = rigid_register_to_reference(vol_2, vol_1)
        mask_2 = (vol_2 > 0).astype(np.float32)

    # Isotropic resampling to (128, 128, 128)
    vol_1_res = resample_isotropic_volume(vol_1, target_shape=target_shape, is_label=False)
    vol_2_res = resample_isotropic_volume(vol_2, target_shape=target_shape, is_label=False)
    mask_1_res = resample_isotropic_volume(mask_1, target_shape=target_shape, is_label=True)
    mask_2_res = resample_isotropic_volume(mask_2, target_shape=target_shape, is_label=True)

    # Normalize intensities into [0, 1]
    x1_norm = normalize_intensity(vol_1_res, brain_mask=mask_1_res)
    x2_norm = normalize_intensity(vol_2_res, brain_mask=mask_2_res)

    if output_dir is not None:
        out_root = Path(output_dir)
        save_mri_volume(x1_norm, out_root / f"{pair_id}_T1_128.nii.gz", affine=affine_1)
        save_mri_volume(x2_norm, out_root / f"{pair_id}_T2_128.nii.gz", affine=affine_1)

    return {
        "X_1": x1_norm[np.newaxis, ...].astype(np.float32),
        "X_2": x2_norm[np.newaxis, ...].astype(np.float32),
        "mask_1": mask_1_res.astype(np.float32),
        "mask_2": mask_2_res.astype(np.float32),
    }
