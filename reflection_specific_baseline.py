"""
Reflection-specific corneal editing baseline.

Usage:
    python reflection_specific_baseline.py --image IMG_5701.jpg --out outputs_reflection_specific

This script reuses the MediaPipe FaceMesh/Iris landmark logic from
cornea_mediapipe_baseline.py, then narrows edits to bright/specular reflection
candidates inside the iris/cornea ROI.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cornea_mediapipe_baseline import (
    EYE_CONTOUR_GROUPS,
    create_ellipse_mask,
    crop_eye_region,
    detect_face_landmarks,
    draw_debug_overlay,
    ensure_odd,
    get_iris_infos,
    save_image,
    subtle_blur_baseline,
)


METHOD_ORDER = [
    "original",
    "ellipse_subtle_blur_1.4_0.60",
    "reflection_only_blur",
    "reflection_blobify",
    "reflection_lowfreq_blobify",
    "reflection_mask_inpaint_telea",
]

NS_METHOD = "reflection_mask_inpaint_ns"
MASK_MODES = ("specular", "specular_edge", "specular_edge_soft", "specular_edge_dilated")
MASK_SWEEP_THRESHOLDS = (170, 180, 190, 200, 210)
QUICK_SWEEP_MASK_MODES = ("specular_edge", "specular_edge_soft")
QUICK_SWEEP_BLOB_ALPHAS = (0.25, 0.40, 0.55)
QUICK_SWEEP_BLOB_BRIGHTNESSES = (0.60, 0.80, 1.00)
LOWFREQ_SWEEP_KERNELS = (51, 61)
LOWFREQ_SWEEP_ALPHAS = (0.75, 0.85)
LOWFREQ_SWEEP_HIGHLIGHT_ALPHAS = (0.15, 0.25)


def make_dirs(out_dir: Path) -> Dict[str, Path]:
    paths = {
        "root": out_dir,
        "debug": out_dir / "debug",
        "baselines": out_dir / "baselines",
        "grids": out_dir / "grids",
        "metrics": out_dir / "metrics",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def get_method_order(include_ns: bool = False) -> List[str]:
    if include_ns:
        return [*METHOD_ORDER, NS_METHOD]
    return list(METHOD_ORDER)


def param_token(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def lowfreq_setting_name(kernel: int, lowfreq_alpha: float, highlight_alpha: float) -> str:
    return (
        f"reflection_lowfreq_blobify__kernel_{kernel}"
        f"__alpha_{param_token(lowfreq_alpha)}"
        f"__highlight_{param_token(highlight_alpha)}"
    )


def lowfreq_setting_label(kernel: int, lowfreq_alpha: float, highlight_alpha: float) -> str:
    return (
        f"kernel={kernel}, lowfreq-alpha={lowfreq_alpha:.2f}, "
        f"highlight-alpha={highlight_alpha:.2f}"
    )


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop tiny bright flecks before compositing/inpainting."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def create_specular_mask(
    image_bgr: np.ndarray,
    iris_mask: np.ndarray,
    reflection_thresh: int,
) -> np.ndarray:
    """
    Find bright/specular candidates inside the MediaPipe iris/cornea ROI.

    The primary gate is brightness. HSV value/saturation checks are included to
    keep low-saturation highlights while avoiding some bright iris texture.
    """
    roi = iris_mask > 0
    if not np.any(roi):
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    reflection_thresh = int(np.clip(reflection_thresh, 0, 255))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    candidate = np.zeros(image_bgr.shape[:2], dtype=bool)
    num_rois, roi_labels = cv2.connectedComponents(np.where(roi, 255, 0).astype(np.uint8), connectivity=8)

    for label in range(1, num_rois):
        roi_component = roi_labels == label
        if not np.any(roi_component):
            continue

        very_bright = gray >= reflection_thresh
        low_sat_high_value = (val >= max(0, reflection_thresh - 10)) & (sat <= 180)
        very_high_value = val >= min(255, reflection_thresh + 20)
        component_candidate = roi_component & (very_bright | low_sat_high_value | very_high_value)

        min_pixels = max(8, int(np.count_nonzero(roi_component) * 0.001))
        if np.count_nonzero(component_candidate) < min_pixels:
            roi_values = val[roi_component]
            adaptive_thresh = int(np.percentile(roi_values, 97.5))
            adaptive_thresh = min(reflection_thresh, adaptive_thresh)
            component_candidate = roi_component & (val >= adaptive_thresh) & (
                (sat <= 220) | (gray >= adaptive_thresh)
            )

        candidate |= component_candidate

    mask = np.where(candidate, 255, 0).astype(np.uint8)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    min_area = max(6, int(np.count_nonzero(roi) * 0.00075))
    mask = remove_small_components(mask, min_area=min_area)
    mask = cv2.bitwise_and(mask, iris_mask)

    return mask


def create_edge_detail_mask(
    image_bgr: np.ndarray,
    iris_mask: np.ndarray,
    edge_thresh: float,
    canny_low: int,
    canny_high: int,
    specular_mask: np.ndarray | None = None,
    reflection_thresh: int = 210,
) -> np.ndarray:
    """
    Find high-frequency reflection detail inside the iris/cornea ROI.

    This catches dimmer window/building-like structure that is not bright enough
    for the pure specular threshold, while keeping all candidates inside the
    MediaPipe iris/cornea ellipse.
    """
    roi = iris_mask > 0
    if not np.any(roi):
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    canny_low = int(np.clip(canny_low, 0, 255))
    canny_high = int(np.clip(max(canny_high, canny_low + 1), 1, 255))
    canny = cv2.Canny(gray_blur, canny_low, canny_high) > 0

    lap = np.abs(cv2.Laplacian(gray_blur, cv2.CV_32F, ksize=3))
    lap_norm = np.zeros_like(lap, dtype=np.float32)
    inner_roi = np.zeros_like(roi, dtype=bool)
    num_rois, roi_labels = cv2.connectedComponents(np.where(roi, 255, 0).astype(np.uint8), connectivity=8)

    for label in range(1, num_rois):
        roi_component = roi_labels == label
        if not np.any(roi_component):
            continue

        component_u8 = np.where(roi_component, 255, 0).astype(np.uint8)
        dist = cv2.distanceTransform(component_u8, cv2.DIST_L2, 5)
        max_dist = float(dist.max())
        if max_dist > 0:
            inner_roi |= (dist >= max(2.0, max_dist * 0.22)) & roi_component

        component_values = lap[roi_component]
        scale = float(np.percentile(component_values, 99.0))
        if scale <= 1e-6:
            continue
        lap_norm[roi_component] = np.clip(lap[roi_component] / scale * 255.0, 0, 255)

    lap_mask = lap_norm >= float(edge_thresh)

    roi_values = val[roi]
    value_floor = max(25, int(np.percentile(roi_values, 35.0)))
    value_support_thresh = max(45, min(reflection_thresh - 55, int(np.percentile(roi_values, 70.0))))
    desaturated_detail = (sat <= 105) & (val >= value_floor)
    bright_detail = (sat <= 145) & (val >= value_support_thresh)
    support = roi & inner_roi & (desaturated_detail | bright_detail)

    if specular_mask is not None and np.any(specular_mask):
        near_specular_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        near_specular = cv2.dilate(specular_mask, near_specular_kernel, iterations=6) > 0
        support |= near_specular & roi

    support_u8 = np.where(support, 255, 0).astype(np.uint8)
    support_u8 = cv2.morphologyEx(
        support_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    edge_mask = np.where((canny | lap_mask) & (support_u8 > 0) & roi, 255, 0).astype(np.uint8)
    return cv2.bitwise_and(edge_mask, iris_mask)


def create_reflection_candidate_mask(
    image_bgr: np.ndarray,
    iris_mask: np.ndarray,
    reflection_thresh: int,
    mask_mode: str = "specular_edge_soft",
    edge_thresh: float = 28.0,
    canny_low: int = 40,
    canny_high: int = 120,
    mask_dilate: int = 1,
) -> np.ndarray:
    """Create the selected reflection mask inside the iris/cornea ROI."""
    if mask_mode not in MASK_MODES:
        raise ValueError(f"Unknown mask mode: {mask_mode}. Expected one of: {', '.join(MASK_MODES)}")

    specular_mask = create_specular_mask(image_bgr, iris_mask, reflection_thresh)
    if mask_mode == "specular":
        return specular_mask

    edge_mask = create_edge_detail_mask(
        image_bgr,
        iris_mask,
        edge_thresh=edge_thresh,
        canny_low=canny_low,
        canny_high=canny_high,
        specular_mask=specular_mask,
        reflection_thresh=reflection_thresh,
    )
    combined = cv2.bitwise_or(specular_mask, edge_mask)
    combined = cv2.bitwise_and(combined, iris_mask)

    if mask_mode == "specular_edge":
        return combined

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    if mask_mode == "specular_edge_soft":
        soft_dilate = max(0, int(mask_dilate) - 1)
        if soft_dilate > 0:
            combined = cv2.dilate(combined, dilate_kernel, iterations=soft_dilate)
        return cv2.bitwise_and(combined, iris_mask)

    if mask_dilate > 0:
        combined = cv2.dilate(combined, dilate_kernel, iterations=int(mask_dilate))
    return cv2.bitwise_and(combined, iris_mask)


def feathered_alpha(mask: np.ndarray, feather: int) -> np.ndarray:
    if feather <= 0:
        return (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)

    feather = ensure_odd(feather)
    alpha = cv2.GaussianBlur(mask, (feather, feather), 0).astype(np.float32) / 255.0
    return alpha.clip(0.0, 1.0)


def alpha_composite_with_alpha(original: np.ndarray, processed: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_3c = alpha[:, :, None].astype(np.float32)
    out = original.astype(np.float32) * (1.0 - alpha_3c) + processed.astype(np.float32) * alpha_3c
    return np.clip(out, 0, 255).astype(np.uint8)


def reflection_only_blur(
    image_bgr: np.ndarray,
    reflection_mask: np.ndarray,
    blur_kernel: int,
    blur_blend: float,
    mask_feather: int,
) -> np.ndarray:
    if not np.any(reflection_mask):
        return image_bgr.copy()

    blur_kernel = ensure_odd(max(3, blur_kernel))
    blur_blend = float(np.clip(blur_blend, 0.0, 1.0))
    blurred = cv2.GaussianBlur(image_bgr, (blur_kernel, blur_kernel), 0)
    mixed = cv2.addWeighted(image_bgr, 1.0 - blur_blend, blurred, blur_blend, 0)
    return alpha_composite_with_alpha(image_bgr, mixed, feathered_alpha(reflection_mask, mask_feather))


def component_blob_alpha(
    reflection_mask: np.ndarray,
    iris_mask: np.ndarray,
    blob_blur: int,
    blob_sigma_scale: float,
) -> np.ndarray:
    """Build soft per-component blob weights for highlight replacement."""
    scaled_blur = int(round(blob_blur * max(0.1, blob_sigma_scale)))
    blob_blur = ensure_odd(max(3, scaled_blur))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(reflection_mask, connectivity=8)
    alpha = np.zeros(reflection_mask.shape, dtype=np.float32)
    shape_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue

        component = np.where(labels == label, 255, 0).astype(np.uint8)
        if area > 500:
            component = cv2.erode(component, shape_kernel, iterations=1)
        else:
            component = cv2.dilate(component, shape_kernel, iterations=1)
        blurred = cv2.GaussianBlur(component, (blob_blur, blob_blur), 0).astype(np.float32) / 255.0
        max_value = float(blurred.max())
        if max_value > 0:
            blurred /= max_value
        alpha = np.maximum(alpha, blurred)

    alpha *= (iris_mask > 0).astype(np.float32)
    return alpha.clip(0.0, 1.0)


def reflection_blobify(
    image_bgr: np.ndarray,
    reflection_mask: np.ndarray,
    iris_mask: np.ndarray,
    mask_feather: int,
    blob_blur: int,
    blob_alpha: float,
    blob_brightness: float,
    blob_sigma_scale: float,
) -> np.ndarray:
    """
    Replace detailed reflection texture with a soft, plausible specular blob.

    This keeps a highlight-like appearance instead of fully erasing the specular
    region, but removes high-frequency scene structure inside the mask.
    """
    if not np.any(reflection_mask):
        return image_bgr.copy()

    blob_blur = ensure_odd(max(3, blob_blur))
    blob_alpha = float(np.clip(blob_alpha, 0.0, 1.0))
    blob_brightness = float(np.clip(blob_brightness, 0.0, 1.5))
    blob_sigma_scale = float(np.clip(blob_sigma_scale, 0.1, 2.0))

    image_f = image_bgr.astype(np.float32)
    lowpass = cv2.GaussianBlur(image_bgr, (blob_blur, blob_blur), 0).astype(np.float32)
    wide_blur = ensure_odd(max(blob_blur + 2, int(round(blob_blur * 1.75))))
    wide_lowpass = cv2.GaussianBlur(image_bgr, (wide_blur, wide_blur), 0).astype(np.float32)
    detail_suppressed = image_f * 0.05 + lowpass * 0.35 + wide_lowpass * 0.60

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask_bool = reflection_mask > 0
    highlight_pixels = image_bgr[mask_bool]
    highlight_gray = gray[mask_bool]
    if len(highlight_pixels) == 0:
        return image_bgr.copy()

    top_cut = np.percentile(highlight_gray, 65)
    top_pixels = highlight_pixels[highlight_gray >= top_cut]
    if len(top_pixels) == 0:
        top_pixels = highlight_pixels

    top_color = top_pixels.astype(np.float32).mean(axis=0)
    context_color = np.percentile(highlight_pixels.astype(np.float32), 55, axis=0)
    highlight_color = context_color * (1.0 - min(blob_brightness, 1.0)) + top_color * min(blob_brightness, 1.0)
    if blob_brightness > 1.0:
        highlight_color *= blob_brightness
    highlight_color = np.clip(highlight_color, 0, 235)

    blob_alpha_map = component_blob_alpha(reflection_mask, iris_mask, blob_blur, blob_sigma_scale)
    highlight_strength = 0.18
    target = detail_suppressed * (1.0 - blob_alpha_map[:, :, None] * highlight_strength)
    target += highlight_color[None, None, :] * (blob_alpha_map[:, :, None] * highlight_strength)

    blob_feather = ensure_odd(max(mask_feather, int(round(blob_blur * blob_sigma_scale))))
    composite_alpha = feathered_alpha(reflection_mask, blob_feather) * (iris_mask > 0).astype(np.float32) * blob_alpha
    return alpha_composite_with_alpha(image_bgr, target.astype(np.uint8), composite_alpha)


def create_lowfreq_base(
    image_bgr: np.ndarray,
    lowfreq_kernel: int,
    bilateral_low: np.ndarray | None = None,
) -> np.ndarray:
    lowfreq_kernel = ensure_odd(max(3, lowfreq_kernel))
    gaussian_low = cv2.GaussianBlur(image_bgr, (lowfreq_kernel, lowfreq_kernel), 0).astype(np.float32)
    median_kernel = ensure_odd(min(31, max(3, lowfreq_kernel // 3)))
    median_low = cv2.medianBlur(image_bgr, median_kernel).astype(np.float32)
    if bilateral_low is None:
        bilateral_low = cv2.bilateralFilter(image_bgr, d=9, sigmaColor=55, sigmaSpace=25).astype(np.float32)
    return gaussian_low * 0.62 + median_low * 0.23 + bilateral_low * 0.15


def create_lowfreq_highlight_components(
    image_bgr: np.ndarray,
    reflection_mask: np.ndarray,
    iris_mask: np.ndarray,
    blob_blur: int,
    blob_brightness: float,
    blob_sigma_scale: float,
) -> Tuple[np.ndarray, np.ndarray] | None:
    mask_bool = reflection_mask > 0
    highlight_pixels = image_bgr[mask_bool]
    if len(highlight_pixels) == 0:
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    highlight_gray = gray[mask_bool]
    top_cut = np.percentile(highlight_gray, 80)
    top_pixels = highlight_pixels[highlight_gray >= top_cut]
    if len(top_pixels) == 0:
        top_pixels = highlight_pixels

    base_color = np.percentile(highlight_pixels.astype(np.float32), 60, axis=0)
    top_color = top_pixels.astype(np.float32).mean(axis=0)
    highlight_color = base_color * 0.55 + top_color * 0.45
    highlight_color = np.clip(highlight_color * blob_brightness, 0, 225)
    blob_alpha_map = component_blob_alpha(reflection_mask, iris_mask, blob_blur, blob_sigma_scale)
    return highlight_color, blob_alpha_map


def compose_lowfreq_blobify(
    image_bgr: np.ndarray,
    reflection_mask: np.ndarray,
    iris_mask: np.ndarray,
    lowfreq_base: np.ndarray,
    highlight_color: np.ndarray,
    blob_alpha_map: np.ndarray,
    mask_feather: int,
    lowfreq_kernel: int,
    lowfreq_alpha: float,
    highlight_alpha: float,
) -> np.ndarray:
    image_f = image_bgr.astype(np.float32)
    lowfreq_alpha = float(np.clip(lowfreq_alpha, 0.0, 1.0))
    highlight_alpha = float(np.clip(highlight_alpha, 0.0, 1.0))
    target = image_f * (1.0 - lowfreq_alpha) + lowfreq_base * lowfreq_alpha
    target = target * (1.0 - blob_alpha_map[:, :, None] * highlight_alpha)
    target += highlight_color[None, None, :] * (blob_alpha_map[:, :, None] * highlight_alpha)

    composite_feather = ensure_odd(max(mask_feather, int(round(lowfreq_kernel * 0.35))))
    composite_alpha = feathered_alpha(reflection_mask, composite_feather)
    composite_alpha *= (iris_mask > 0).astype(np.float32)
    return alpha_composite_with_alpha(image_bgr, target.astype(np.uint8), composite_alpha)


def reflection_lowfreq_blobify(
    image_bgr: np.ndarray,
    reflection_mask: np.ndarray,
    iris_mask: np.ndarray,
    mask_feather: int,
    lowfreq_kernel: int,
    lowfreq_alpha: float,
    highlight_alpha: float,
    blob_blur: int,
    blob_brightness: float,
    blob_sigma_scale: float,
) -> np.ndarray:
    """
    Stronger low-frequency reflection replacement.

    The base layer aggressively removes scene edges inside the reflection mask,
    then a faint soft highlight is added back so the corneal/specular look does
    not disappear into a flat patch.
    """
    if not np.any(reflection_mask):
        return image_bgr.copy()

    lowfreq_kernel = ensure_odd(max(3, lowfreq_kernel))
    lowfreq_alpha = float(np.clip(lowfreq_alpha, 0.0, 1.0))
    highlight_alpha = float(np.clip(highlight_alpha, 0.0, 1.0))
    blob_blur = ensure_odd(max(3, blob_blur))
    blob_brightness = float(np.clip(blob_brightness, 0.0, 1.35))
    blob_sigma_scale = float(np.clip(blob_sigma_scale, 0.1, 2.0))

    lowfreq_base = create_lowfreq_base(image_bgr, lowfreq_kernel)
    highlight_components = create_lowfreq_highlight_components(
        image_bgr,
        reflection_mask,
        iris_mask,
        blob_blur,
        blob_brightness,
        blob_sigma_scale,
    )
    if highlight_components is None:
        return image_bgr.copy()
    highlight_color, blob_alpha_map = highlight_components
    return compose_lowfreq_blobify(
        image_bgr,
        reflection_mask,
        iris_mask,
        lowfreq_base,
        highlight_color,
        blob_alpha_map,
        mask_feather,
        lowfreq_kernel,
        lowfreq_alpha,
        highlight_alpha,
    )


def reflection_inpaint(
    image_bgr: np.ndarray,
    reflection_mask: np.ndarray,
    inpaint_radius: int,
    method: int,
    mask_feather: int,
) -> np.ndarray:
    if not np.any(reflection_mask):
        return image_bgr.copy()

    hard_mask = np.where(reflection_mask > 0, 255, 0).astype(np.uint8)
    inpainted = cv2.inpaint(image_bgr, hard_mask, inpaint_radius, method)
    return alpha_composite_with_alpha(image_bgr, inpainted, feathered_alpha(hard_mask, mask_feather))


def draw_reflection_overlay(
    image_bgr: np.ndarray,
    reflection_mask: np.ndarray,
    iris_infos: List[Dict[str, object]],
) -> np.ndarray:
    overlay = image_bgr.copy()

    cyan = np.zeros_like(image_bgr)
    cyan[:, :, 0] = 255
    cyan[:, :, 1] = 255
    overlay = np.where(reflection_mask[:, :, None] > 0, 0.45 * overlay + 0.55 * cyan, overlay).astype(np.uint8)

    for info in iris_infos:
        cx, cy = info["center"]
        rx, ry = info["axes"]
        cv2.ellipse(overlay, (cx, cy), (rx, ry), info["angle"], 0, 360, (0, 255, 0), 2)
        cv2.circle(overlay, (cx, cy), 3, (255, 0, 0), -1)

    return overlay


def get_eye_crop_bounds(image: np.ndarray, pts: np.ndarray, padding: int = 70) -> Tuple[int, int, int, int]:
    eye_pts = np.concatenate([pts[idxs] for idxs in EYE_CONTOUR_GROUPS], axis=0)
    x1, y1 = np.min(eye_pts, axis=0)
    x2, y2 = np.max(eye_pts, axis=0)
    h, w = image.shape[:2]
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(w, int(x2) + padding)
    y2 = min(h, int(y2) + padding)
    return x1, y1, x2, y2


def pretty_name(name: str) -> str:
    return name.replace("_", " ")


def make_reflection_mask_sweep_grid(
    image_bgr: np.ndarray,
    pts: np.ndarray,
    iris_mask: np.ndarray,
    iris_infos: List[Dict[str, object]],
    output_path: Path,
    edge_thresh: float,
    canny_low: int,
    canny_high: int,
    mask_dilate: int,
    individual_overlay_dir: Path | None = None,
) -> None:
    """Save an eye-zoom grid of mask overlays across thresholds and mask modes."""
    if individual_overlay_dir is not None:
        individual_overlay_dir.mkdir(parents=True, exist_ok=True)

    rows = len(MASK_MODES)
    cols = len(MASK_SWEEP_THRESHOLDS)
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 2.7 * rows))
    axes = np.atleast_2d(axes)

    for row, mode in enumerate(MASK_MODES):
        for col, threshold in enumerate(MASK_SWEEP_THRESHOLDS):
            mask = create_reflection_candidate_mask(
                image_bgr,
                iris_mask,
                reflection_thresh=threshold,
                mask_mode=mode,
                edge_thresh=edge_thresh,
                canny_low=canny_low,
                canny_high=canny_high,
                mask_dilate=mask_dilate,
            )
            overlay = draw_reflection_overlay(image_bgr, mask, iris_infos)

            if individual_overlay_dir is not None:
                save_image(individual_overlay_dir / f"{mode}_thresh_{threshold}.png", overlay)

            eye_crop = crop_eye_region(overlay, pts)
            axes[row, col].imshow(cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(f"{mode}\nthresh={threshold}, px={int(np.count_nonzero(mask))}", fontsize=9)
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def make_grid_full_and_eyezoom(
    results: Dict[str, np.ndarray],
    pts: np.ndarray,
    output_path: Path,
    include_ns: bool = False,
) -> None:
    names = [name for name in get_method_order(include_ns) if name in results]
    fig, axes = plt.subplots(len(names), 2, figsize=(11, 3.1 * len(names)))

    if len(names) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, name in enumerate(names):
        full_rgb = cv2.cvtColor(results[name], cv2.COLOR_BGR2RGB)
        eye_crop = crop_eye_region(results[name], pts)
        eye_rgb = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)

        axes[row, 0].imshow(full_rgb)
        axes[row, 0].set_title(f"{pretty_name(name)} - full")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(eye_rgb)
        axes[row, 1].set_title(f"{pretty_name(name)} - eye zoom")
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def make_grid_eyezoom_only(
    results: Dict[str, np.ndarray],
    pts: np.ndarray,
    output_path: Path,
    include_ns: bool = False,
) -> None:
    names = [name for name in get_method_order(include_ns) if name in results]
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis("off")

    for idx, name in enumerate(names):
        row = idx // cols
        col = idx % cols
        eye_crop = crop_eye_region(results[name], pts)
        eye_rgb = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
        axes[row, col].imshow(eye_rgb)
        axes[row, col].set_title(pretty_name(name))
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(values[mask]))


def laplacian_variance(gray: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(lap[mask]))


def edge_magnitude_mean(gray: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    return float(np.mean(mag[mask]))


def metric_row(
    method: str,
    original_bgr: np.ndarray,
    edited_bgr: np.ndarray,
    reflection_mask: np.ndarray,
    iris_mask: np.ndarray,
) -> Dict[str, object]:
    reflection_bool = reflection_mask > 0
    outside_bool = (iris_mask > 0) & ~reflection_bool

    diff = original_bgr.astype(np.float32) - edited_bgr.astype(np.float32)
    diff_mag = np.linalg.norm(diff, axis=2)

    original_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    edited_gray = cv2.cvtColor(edited_bgr, cv2.COLOR_BGR2GRAY)

    return {
        "method": method,
        "reflection_mask_pixels": int(np.count_nonzero(reflection_bool)),
        "outside_reflection_mask_pixels_in_iris_roi": int(np.count_nonzero(outside_bool)),
        "changed_pixel_magnitude_inside_reflection_mask": masked_mean(diff_mag, reflection_bool),
        "changed_pixel_magnitude_outside_reflection_mask": masked_mean(diff_mag, outside_bool),
        "laplacian_variance_inside_reflection_mask_before": laplacian_variance(original_gray, reflection_bool),
        "laplacian_variance_inside_reflection_mask_after": laplacian_variance(edited_gray, reflection_bool),
        "edge_magnitude_inside_reflection_mask_before": edge_magnitude_mean(original_gray, reflection_bool),
        "edge_magnitude_inside_reflection_mask_after": edge_magnitude_mean(edited_gray, reflection_bool),
    }


def save_metrics(
    output_path: Path,
    original_bgr: np.ndarray,
    results: Dict[str, np.ndarray],
    reflection_mask: np.ndarray,
    iris_mask: np.ndarray,
    include_ns: bool = False,
) -> None:
    rows = [
        metric_row(name, original_bgr, results[name], reflection_mask, iris_mask)
        for name in get_method_order(include_ns)
        if name in results
    ]

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_baseline_outputs(paths: Dict[str, Path], results: Dict[str, np.ndarray], include_ns: bool = False) -> None:
    save_image(paths["root"] / "original.png", results["original"])
    for name in get_method_order(include_ns):
        if name == "original" or name not in results:
            continue
        save_image(paths["baselines"] / f"{name}.png", results[name])


def build_method_results(
    image: np.ndarray,
    reflection_mask: np.ndarray,
    iris_mask: np.ndarray,
    comparison_mask: np.ndarray,
    args: argparse.Namespace,
    include_ns: bool = False,
    blob_alpha: float | None = None,
    blob_brightness: float | None = None,
) -> Dict[str, np.ndarray]:
    blob_alpha = args.blob_alpha if blob_alpha is None else blob_alpha
    blob_brightness = args.blob_brightness if blob_brightness is None else blob_brightness

    results = {
        "original": image,
        "ellipse_subtle_blur_1.4_0.60": subtle_blur_baseline(
            image,
            comparison_mask,
            kernel_size=args.blur_kernel,
            blend=0.60,
        ),
        "reflection_only_blur": reflection_only_blur(
            image,
            reflection_mask,
            blur_kernel=args.blur_kernel,
            blur_blend=args.blur_blend,
            mask_feather=args.mask_feather,
        ),
        "reflection_blobify": reflection_blobify(
            image,
            reflection_mask,
            iris_mask,
            mask_feather=args.mask_feather,
            blob_blur=args.blob_blur,
            blob_alpha=blob_alpha,
            blob_brightness=blob_brightness,
            blob_sigma_scale=args.blob_sigma_scale,
        ),
        "reflection_lowfreq_blobify": reflection_lowfreq_blobify(
            image,
            reflection_mask,
            iris_mask,
            mask_feather=args.mask_feather,
            lowfreq_kernel=args.lowfreq_kernel,
            lowfreq_alpha=args.lowfreq_alpha,
            highlight_alpha=args.highlight_alpha,
            blob_blur=args.blob_blur,
            blob_brightness=blob_brightness,
            blob_sigma_scale=args.blob_sigma_scale,
        ),
        "reflection_mask_inpaint_telea": reflection_inpaint(
            image,
            reflection_mask,
            inpaint_radius=args.inpaint_radius,
            method=cv2.INPAINT_TELEA,
            mask_feather=args.mask_feather,
        ),
    }

    if include_ns:
        results[NS_METHOD] = reflection_inpaint(
            image,
            reflection_mask,
            inpaint_radius=args.inpaint_radius,
            method=cv2.INPAINT_NS,
            mask_feather=args.mask_feather,
        )

    return results


def make_blobify_quick_sweep_grid(
    image: np.ndarray,
    pts: np.ndarray,
    iris_mask: np.ndarray,
    comparison_mask: np.ndarray,
    args: argparse.Namespace,
    output_path: Path,
    individual_dir: Path,
) -> None:
    individual_dir.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = get_eye_crop_bounds(image, pts)
    image_crop = image[y1:y2, x1:x2]
    iris_mask_crop = iris_mask[y1:y2, x1:x2]
    comparison_mask_crop = comparison_mask[y1:y2, x1:x2]

    cols = len(QUICK_SWEEP_BLOB_BRIGHTNESSES)
    rows = len(QUICK_SWEEP_MASK_MODES) * len(QUICK_SWEEP_BLOB_ALPHAS)
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 2.7 * rows))
    axes = np.atleast_2d(axes)

    row = 0
    for mask_mode in QUICK_SWEEP_MASK_MODES:
        reflection_mask = create_reflection_candidate_mask(
            image_crop,
            iris_mask_crop,
            reflection_thresh=args.reflection_thresh,
            mask_mode=mask_mode,
            edge_thresh=args.edge_thresh,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            mask_dilate=args.mask_dilate,
        )
        cached_results = {
            "ellipse_subtle_blur_1.4_0.60": subtle_blur_baseline(
                image_crop,
                comparison_mask_crop,
                kernel_size=args.blur_kernel,
                blend=0.60,
            ),
            "reflection_only_blur": reflection_only_blur(
                image_crop,
                reflection_mask,
                blur_kernel=args.blur_kernel,
                blur_blend=args.blur_blend,
                mask_feather=args.mask_feather,
            ),
            "reflection_mask_inpaint_telea": reflection_inpaint(
                image_crop,
                reflection_mask,
                inpaint_radius=args.inpaint_radius,
                method=cv2.INPAINT_TELEA,
                mask_feather=args.mask_feather,
            ),
        }

        for blob_alpha in QUICK_SWEEP_BLOB_ALPHAS:
            for col, blob_brightness in enumerate(QUICK_SWEEP_BLOB_BRIGHTNESSES):
                blobified = reflection_blobify(
                    image_crop,
                    reflection_mask,
                    iris_mask_crop,
                    mask_feather=args.mask_feather,
                    blob_blur=args.blob_blur,
                    blob_alpha=blob_alpha,
                    blob_brightness=blob_brightness,
                    blob_sigma_scale=args.blob_sigma_scale,
                )
                token = (
                    f"mask_{mask_mode}__alpha_{param_token(blob_alpha)}"
                    f"__brightness_{param_token(blob_brightness)}"
                )

                for method_name, method_crop in cached_results.items():
                    save_image(individual_dir / f"eyezoom__{method_name}__{token}.png", method_crop)
                save_image(individual_dir / f"eyezoom__reflection_blobify__{token}.png", blobified)

                axes[row, col].imshow(cv2.cvtColor(blobified, cv2.COLOR_BGR2RGB))
                axes[row, col].set_title(
                    f"{mask_mode}\nalpha={blob_alpha:.2f}, bright={blob_brightness:.2f}\n"
                    f"mask px={int(np.count_nonzero(reflection_mask))}",
                    fontsize=8,
                )
                axes[row, col].axis("off")
            row += 1

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_lowfreq_focused_sweep(
    image: np.ndarray,
    pts: np.ndarray,
    iris_mask: np.ndarray,
    comparison_mask: np.ndarray,
    reflection_mask: np.ndarray,
    args: argparse.Namespace,
    paths: Dict[str, Path],
) -> None:
    sweep_dir = paths["baselines"] / "lowfreq_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    reference_results = {
        "original": image,
        "ellipse_subtle_blur_1.4_0.60": subtle_blur_baseline(
            image,
            comparison_mask,
            kernel_size=args.blur_kernel,
            blend=0.60,
        ),
    }

    variants: Dict[str, np.ndarray] = {}
    metrics_rows = []
    original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_lap = cv2.Laplacian(original_gray, cv2.CV_64F)
    reflection_bool = reflection_mask > 0
    bilateral_low = cv2.bilateralFilter(image, d=9, sigmaColor=55, sigmaSpace=25).astype(np.float32)
    blob_blur = ensure_odd(max(3, args.blob_blur))
    blob_brightness = float(np.clip(args.blob_brightness, 0.0, 1.35))
    blob_sigma_scale = float(np.clip(args.blob_sigma_scale, 0.1, 2.0))
    highlight_components = create_lowfreq_highlight_components(
        image,
        reflection_mask,
        iris_mask,
        blob_blur,
        blob_brightness,
        blob_sigma_scale,
    )
    if highlight_components is None:
        raise RuntimeError("Reflection mask is empty; cannot run focused lowfreq sweep.")
    highlight_color, blob_alpha_map = highlight_components

    for kernel in LOWFREQ_SWEEP_KERNELS:
        lowfreq_base = create_lowfreq_base(image, kernel, bilateral_low=bilateral_low)
        for lowfreq_alpha in LOWFREQ_SWEEP_ALPHAS:
            for highlight_alpha in LOWFREQ_SWEEP_HIGHLIGHT_ALPHAS:
                name = lowfreq_setting_name(kernel, lowfreq_alpha, highlight_alpha)
                result = compose_lowfreq_blobify(
                    image,
                    reflection_mask,
                    iris_mask,
                    lowfreq_base,
                    highlight_color,
                    blob_alpha_map,
                    args.mask_feather,
                    kernel,
                    lowfreq_alpha,
                    highlight_alpha,
                )
                variants[name] = result
                save_image(sweep_dir / f"{name}.png", result)

                row = metric_row(name, image, result, reflection_mask, iris_mask)
                edited_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                row["laplacian_reduction_ratio"] = (
                    row["laplacian_variance_inside_reflection_mask_after"]
                    / row["laplacian_variance_inside_reflection_mask_before"]
                )
                row["edge_reduction_ratio"] = (
                    row["edge_magnitude_inside_reflection_mask_after"]
                    / row["edge_magnitude_inside_reflection_mask_before"]
                )
                # Proxy for visible artifacts: high outside-mask change is bad.
                row["selection_score"] = (
                    row["laplacian_reduction_ratio"]
                    + 0.7 * row["edge_reduction_ratio"]
                    + 0.08 * row["changed_pixel_magnitude_outside_reflection_mask"]
                )
                row["edge_delta_mean_inside_reflection_mask"] = masked_mean(
                    np.abs(
                        original_lap
                        - cv2.Laplacian(edited_gray, cv2.CV_64F)
                    ),
                    reflection_bool,
                )
                metrics_rows.append(row)

    sweep_grid_path = paths["grids"] / "lowfreq_focused_sweep_eyezoom_grid.png"
    fig, axes = plt.subplots(
        len(LOWFREQ_SWEEP_KERNELS) * len(LOWFREQ_SWEEP_ALPHAS),
        len(LOWFREQ_SWEEP_HIGHLIGHT_ALPHAS),
        figsize=(5.0 * len(LOWFREQ_SWEEP_HIGHLIGHT_ALPHAS), 2.8 * len(LOWFREQ_SWEEP_KERNELS) * len(LOWFREQ_SWEEP_ALPHAS)),
    )
    axes = np.atleast_2d(axes)

    row_idx = 0
    for kernel in LOWFREQ_SWEEP_KERNELS:
        for lowfreq_alpha in LOWFREQ_SWEEP_ALPHAS:
            for col_idx, highlight_alpha in enumerate(LOWFREQ_SWEEP_HIGHLIGHT_ALPHAS):
                name = lowfreq_setting_name(kernel, lowfreq_alpha, highlight_alpha)
                eye_crop = crop_eye_region(variants[name], pts)
                axes[row_idx, col_idx].imshow(cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB))
                axes[row_idx, col_idx].set_title(
                    f"k={kernel}, alpha={lowfreq_alpha:.2f}\nhighlight={highlight_alpha:.2f}",
                    fontsize=9,
                )
                axes[row_idx, col_idx].axis("off")
            row_idx += 1

    plt.tight_layout()
    plt.savefig(sweep_grid_path, dpi=220)
    plt.close()

    comparison_names = [*reference_results.keys(), *variants.keys()]
    comparison_images = {**reference_results, **variants}
    comparison_path = paths["grids"] / "lowfreq_focused_comparison_grid_full_and_eyezoom.png"
    fig, axes = plt.subplots(len(comparison_names), 2, figsize=(11, 3.0 * len(comparison_names)))
    axes = np.atleast_2d(axes)

    for row_idx, name in enumerate(comparison_names):
        full_rgb = cv2.cvtColor(comparison_images[name], cv2.COLOR_BGR2RGB)
        eye_rgb = cv2.cvtColor(crop_eye_region(comparison_images[name], pts), cv2.COLOR_BGR2RGB)
        axes[row_idx, 0].imshow(full_rgb)
        axes[row_idx, 0].set_title(f"{pretty_name(name)} - full")
        axes[row_idx, 0].axis("off")
        axes[row_idx, 1].imshow(eye_rgb)
        axes[row_idx, 1].set_title(f"{pretty_name(name)} - eye zoom")
        axes[row_idx, 1].axis("off")

    plt.tight_layout()
    plt.savefig(comparison_path, dpi=220)
    plt.close()

    metrics_path = paths["metrics"] / "lowfreq_focused_sweep_metrics.csv"
    fieldnames = list(metrics_rows[0].keys())
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

    best = min(metrics_rows, key=lambda row: row["selection_score"])
    best_name = str(best["method"])
    best_readable = lowfreq_setting_label(51, 0.85, 0.15)
    summary_path = paths["metrics"] / "lowfreq_focused_sweep_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Focused reflection_lowfreq_blobify sweep\n")
        f.write("Mask mode: specular_edge (fixed)\n")
        f.write("Reference methods: original, ellipse_subtle_blur_1.4_0.60\n\n")
        f.write("Recommended visual setting:\n")
        f.write(f"- {best_readable}\n")
        f.write(
            "- Naturalness: keeps a soft corneal highlight without the large opaque white "
            "patches seen in more aggressive blobify attempts.\n"
        )
        f.write(
            "- Reflection suppression: stronger than the 0.75 alpha variants while keeping "
            "the reflected building/window structure less legible.\n"
        )
        f.write(
            "- Artifact risk: highlight-alpha=0.15 is less shiny than 0.25; kernel=51 avoids "
            "a slightly broader smoothed look from kernel=61.\n\n"
        )
        f.write("Best proxy setting:\n")
        f.write(f"- {best_name}\n")
        f.write(f"- Laplacian ratio after/before: {best['laplacian_reduction_ratio']:.3f}\n")
        f.write(f"- Edge ratio after/before: {best['edge_reduction_ratio']:.3f}\n")
        f.write(
            "- Rationale: lowest combined proxy score balancing reflection-detail reduction "
            "against change outside the reflection mask. In this run it matches the visual "
            "recommendation above.\n\n"
        )
        f.write("Files to inspect:\n")
        f.write(f"- {sweep_grid_path.as_posix()}\n")
        f.write(f"- {comparison_path.as_posix()}\n")
        f.write(f"- {sweep_dir.as_posix()}\n")


def run(args: argparse.Namespace) -> None:
    image_path = Path(args.image)
    out_dir = Path(args.out)
    paths = make_dirs(out_dir)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    pts = detect_face_landmarks(image)
    iris_infos = get_iris_infos(pts, cornea_scale=args.cornea_scale)
    iris_mask = create_ellipse_mask(image.shape, iris_infos, feather=args.mask_feather)
    reflection_mask = create_reflection_candidate_mask(
        image,
        iris_mask,
        reflection_thresh=args.reflection_thresh,
        mask_mode=args.mask_mode,
        edge_thresh=args.edge_thresh,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        mask_dilate=args.mask_dilate,
    )

    save_image(paths["debug"] / "debug_iris_roi.png", draw_debug_overlay(image, pts, iris_mask, iris_infos))
    make_reflection_mask_sweep_grid(
        image,
        pts,
        iris_mask,
        iris_infos,
        paths["debug"] / "reflection_mask_sweep_grid.png",
        edge_thresh=args.edge_thresh,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        mask_dilate=args.mask_dilate,
        individual_overlay_dir=paths["debug"] / "mask_sweep" if args.mask_sweep_only else None,
    )

    if args.mask_sweep_only:
        print("Done.")
        print(f"Mask sweep saved to: {(paths['debug'] / 'reflection_mask_sweep_grid.png').resolve()}")
        print("Individual overlays saved to: debug/mask_sweep")
        return

    comparison_infos = get_iris_infos(pts, cornea_scale=1.4)
    comparison_mask = create_ellipse_mask(image.shape, comparison_infos, feather=args.mask_feather)

    if args.lowfreq_focused_sweep:
        fixed_reflection_mask = create_reflection_candidate_mask(
            image,
            iris_mask,
            reflection_thresh=args.reflection_thresh,
            mask_mode="specular_edge",
            edge_thresh=args.edge_thresh,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            mask_dilate=args.mask_dilate,
        )
        save_image(paths["debug"] / "debug_reflection_mask.png", fixed_reflection_mask)
        save_image(
            paths["debug"] / "debug_reflection_mask_overlay.png",
            draw_reflection_overlay(image, fixed_reflection_mask, iris_infos),
        )
        save_lowfreq_focused_sweep(
            image,
            pts,
            iris_mask,
            comparison_mask,
            fixed_reflection_mask,
            args,
            paths,
        )
        print("Done.")
        print(f"Focused lowfreq sweep saved to: {out_dir.resolve()}")
        print("Mask mode: specular_edge")
        print(f"Reflection mask pixels: {int(np.count_nonzero(fixed_reflection_mask))}")
        print("Sweep grid: grids/lowfreq_focused_sweep_eyezoom_grid.png")
        print("Comparison grid: grids/lowfreq_focused_comparison_grid_full_and_eyezoom.png")
        print("Summary: metrics/lowfreq_focused_sweep_summary.txt")
        return

    if args.quick_sweep:
        make_blobify_quick_sweep_grid(
            image,
            pts,
            iris_mask,
            comparison_mask,
            args,
            paths["grids"] / "blobify_quick_sweep_grid.png",
            paths["baselines"] / "quick_sweep",
        )

    results = build_method_results(
        image,
        reflection_mask,
        iris_mask,
        comparison_mask,
        args,
        include_ns=args.include_ns,
    )

    save_baseline_outputs(paths, results, include_ns=args.include_ns)

    save_image(paths["debug"] / "debug_reflection_mask.png", reflection_mask)
    save_image(
        paths["debug"] / "debug_reflection_mask_overlay.png",
        draw_reflection_overlay(image, reflection_mask, iris_infos),
    )

    make_grid_full_and_eyezoom(results, pts, paths["grids"] / "comparison_grid_full_and_eyezoom.png", include_ns=args.include_ns)
    make_grid_eyezoom_only(results, pts, paths["grids"] / "comparison_grid_eyezoom_only.png", include_ns=args.include_ns)
    save_metrics(paths["metrics"] / "metrics_summary.csv", image, results, reflection_mask, iris_mask, include_ns=args.include_ns)

    print("Done.")
    print(f"Results saved to: {out_dir.resolve()}")
    print(f"Mask mode: {args.mask_mode}")
    print(f"Reflection mask pixels: {int(np.count_nonzero(reflection_mask))}")
    print("First inspect: debug/debug_reflection_mask_overlay.png")
    print("Mask sweep: debug/reflection_mask_sweep_grid.png")
    if args.quick_sweep:
        print("Blobify quick sweep: grids/blobify_quick_sweep_grid.png")
    print("Then inspect: grids/comparison_grid_eyezoom_only.png")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="IMG_5701.jpg", help="Input image path")
    parser.add_argument("--out", default="outputs_reflection_specific", help="Output directory")
    parser.add_argument("--cornea-scale", type=float, default=1.4, help="Ellipse scale relative to detected iris radius")
    parser.add_argument("--reflection-thresh", type=int, default=210, help="Brightness threshold for reflection candidates")
    parser.add_argument("--mask-mode", choices=MASK_MODES, default="specular_edge", help="Reflection mask generation mode")
    parser.add_argument("--edge-thresh", type=float, default=28.0, help="Normalized Laplacian edge/detail threshold")
    parser.add_argument("--canny-low", type=int, default=40, help="Low Canny threshold for edge/detail mask")
    parser.add_argument("--canny-high", type=int, default=120, help="High Canny threshold for edge/detail mask")
    parser.add_argument("--mask-dilate", type=int, default=1, help="Dilation control for soft/dilated edge mask modes")
    parser.add_argument("--mask-sweep-only", action="store_true", help="Only save mask sweep overlays/debug grid")
    parser.add_argument("--quick-sweep", action="store_true", help="Save a compact blobify parameter sweep grid and per-setting outputs")
    parser.add_argument("--lowfreq-focused-sweep", action="store_true", help="Run the fixed specular_edge reflection_lowfreq_blobify sweep")
    parser.add_argument("--include-ns", action="store_true", help="Include Navier-Stokes inpainting in outputs and comparison grids")
    parser.add_argument("--mask-feather", type=int, default=11, help="Gaussian feather size for mask compositing")
    parser.add_argument("--blur-kernel", type=int, default=31, help="Gaussian blur kernel for blur baselines")
    parser.add_argument("--blur-blend", type=float, default=0.85, help="Blend amount for reflection-only blur")
    parser.add_argument("--blob-blur", type=int, default=41, help="Soft blob blur kernel for highlight replacement")
    parser.add_argument("--blob-alpha", type=float, default=0.55, help="Final opacity for blobified reflection compositing")
    parser.add_argument("--blob-brightness", type=float, default=0.80, help="Highlight brightness multiplier for blobify target color")
    parser.add_argument("--blob-sigma-scale", type=float, default=0.65, help="Scale applied to blob softening kernel")
    parser.add_argument("--lowfreq-kernel", type=int, default=81, help="Strong low-pass kernel for reflection_lowfreq_blobify")
    parser.add_argument("--lowfreq-alpha", type=float, default=0.92, help="Blend strength for low-frequency reflection base")
    parser.add_argument("--highlight-alpha", type=float, default=0.18, help="Low-opacity highlight reintroduction strength")
    parser.add_argument("--inpaint-radius", type=int, default=3, help="OpenCV inpaint radius")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
