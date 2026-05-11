"""
MediaPipe Iris-based localized cornea/iris perturbation baseline.

Usage:
    pip install opencv-python mediapipe matplotlib numpy
    python cornea_mediapipe_baseline.py --image IMG_5701.jpg --out outputs_mediapipe

What it does:
    1) Detects iris landmarks with MediaPipe FaceMesh(refine_landmarks=True).
    2) Builds localized masks around the iris/cornea area.
    3) Applies baseline perturbations only inside the localized masks.
    4) Saves full-image outputs, eye zoom comparisons, and mask debug overlays.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np


# MediaPipe Iris landmarks are appended after the 468 face-mesh landmarks.
# We avoid relying on semantic "left/right" names because camera/selfie orientation
# can be confusing. Instead, we compute two iris groups and sort by x-coordinate.
IRIS_GROUPS = [
    {"center": 468, "contour": [469, 470, 471, 472]},
    {"center": 473, "contour": [474, 475, 476, 477]},
]

# Standard FaceMesh eye-contour landmark sets. Used only for zoom/debug crop.
EYE_CONTOUR_GROUPS = [
    [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
]


def ensure_odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def landmarks_to_pixels(face_landmarks, width: int, height: int) -> np.ndarray:
    """Convert normalized MediaPipe landmarks to pixel coordinates."""
    pts = []
    for lm in face_landmarks.landmark:
        x = int(round(lm.x * width))
        y = int(round(lm.y * height))
        pts.append((x, y))
    return np.array(pts, dtype=np.int32)


def detect_face_landmarks(image_bgr: np.ndarray) -> np.ndarray:
    """Return pixel-space landmarks for the first detected face."""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        raise RuntimeError("No face landmarks detected. Try a clearer/frontal image or lower min_detection_confidence.")

    face_landmarks = results.multi_face_landmarks[0]
    pts = landmarks_to_pixels(face_landmarks, w, h)

    if len(pts) < 478:
        raise RuntimeError(
            f"Only {len(pts)} landmarks detected. Iris landmarks require refine_landmarks=True and 478 landmarks."
        )

    return pts


def get_iris_infos(pts: np.ndarray, cornea_scale: float = 1.8) -> List[Dict[str, object]]:
    """
    Build ellipse parameters from MediaPipe iris landmarks.

    cornea_scale > 1 expands the mask beyond the iris to cover the visible corneal
    reflection area. Start with 1.6-2.2 and inspect debug_mask_overlay.png.
    """
    infos = []

    for group in IRIS_GROUPS:
        center_idx = group["center"]
        contour_idx = group["contour"]

        center = pts[center_idx].astype(np.float32)
        contour = pts[contour_idx].astype(np.float32)

        # Estimate iris radius from contour points. Use separate rx/ry to allow slight ellipticity.
        dx = np.abs(contour[:, 0] - center[0])
        dy = np.abs(contour[:, 1] - center[1])
        rx = max(2, int(round(np.max(dx) * cornea_scale)))
        ry = max(2, int(round(np.max(dy) * cornea_scale)))

        infos.append(
            {
                "center": (int(round(center[0])), int(round(center[1]))),
                "axes": (rx, ry),
                "angle": 0,
                "contour": contour.astype(np.int32),
            }
        )

    # Stable naming by image x-coordinate, not anatomical left/right.
    infos = sorted(infos, key=lambda item: item["center"][0])
    infos[0]["name"] = "image_left_eye"
    infos[1]["name"] = "image_right_eye"
    return infos


def create_ellipse_mask(image_shape: Tuple[int, int, int], iris_infos: List[Dict[str, object]], feather: int = 9) -> np.ndarray:
    """Create a soft-ish binary mask around both iris/cornea regions."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for info in iris_infos:
        cv2.ellipse(
            mask,
            center=info["center"],
            axes=info["axes"],
            angle=info["angle"],
            startAngle=0,
            endAngle=360,
            color=255,
            thickness=-1,
        )

    if feather > 0:
        feather = ensure_odd(feather)
        # Keep it binary for inpainting, but smooth edge for masked compositing later.
        mask = cv2.GaussianBlur(mask, (feather, feather), 0)
        mask = np.where(mask > 20, 255, 0).astype(np.uint8)

    return mask


def create_center_crop_mask(image_shape: Tuple[int, int, int], iris_infos: List[Dict[str, object]], crop_scale: float = 2.0) -> np.ndarray:
    """Alternative baseline: rectangular center crop around iris/cornea area."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    h, w = image_shape[:2]

    for info in iris_infos:
        cx, cy = info["center"]
        rx, ry = info["axes"]
        half_w = int(round(rx * crop_scale / 2.0))
        half_h = int(round(ry * crop_scale / 2.0))
        x1 = max(0, cx - half_w)
        y1 = max(0, cy - half_h)
        x2 = min(w, cx + half_w)
        y2 = min(h, cy + half_h)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    return mask


def alpha_composite_masked(original: np.ndarray, processed: np.ndarray, mask: np.ndarray, feather: int = 21) -> np.ndarray:
    """Composite processed image into original using a feathered mask edge."""
    feather = ensure_odd(feather)
    alpha = cv2.GaussianBlur(mask, (feather, feather), 0).astype(np.float32) / 255.0
    alpha = alpha[:, :, None]
    out = original.astype(np.float32) * (1.0 - alpha) + processed.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def gaussian_blur_baseline(image: np.ndarray, mask: np.ndarray, kernel_size: int = 31) -> np.ndarray:
    kernel_size = ensure_odd(kernel_size)
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return alpha_composite_masked(image, blurred, mask)


def subtle_blur_baseline(image: np.ndarray, mask: np.ndarray, kernel_size: int = 15, blend: float = 0.65) -> np.ndarray:
    """Less destructive blur; useful when full blur looks fake."""
    kernel_size = ensure_odd(kernel_size)
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    mixed = cv2.addWeighted(image, 1.0 - blend, blurred, blend, 0)
    return alpha_composite_masked(image, mixed, mask)


def pixelation_baseline(image: np.ndarray, mask: np.ndarray, pixel_size: int = 8) -> np.ndarray:
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return alpha_composite_masked(image, pixelated, mask)


def down_up_noise_baseline(image: np.ndarray, mask: np.ndarray, pixel_size: int = 8, noise_std: float = 8.0) -> np.ndarray:
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
    noisy = np.clip(restored.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return alpha_composite_masked(image, noisy, mask)


def telea_inpainting_baseline(image: np.ndarray, mask: np.ndarray, radius: int = 3) -> np.ndarray:
    # cv2.inpaint expects a hard 8-bit mask.
    inpainted = cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)
    return alpha_composite_masked(image, inpainted, mask)


def navier_stokes_inpainting_baseline(image: np.ndarray, mask: np.ndarray, radius: int = 3) -> np.ndarray:
    inpainted = cv2.inpaint(image, mask, radius, cv2.INPAINT_NS)
    return alpha_composite_masked(image, inpainted, mask)


def crop_eye_region(image: np.ndarray, pts: np.ndarray, padding: int = 70) -> np.ndarray:
    """Crop both eyes using FaceMesh eye contour landmarks for zoomed comparison."""
    eye_pts = []
    for idxs in EYE_CONTOUR_GROUPS:
        eye_pts.append(pts[idxs])
    eye_pts = np.concatenate(eye_pts, axis=0)

    x1, y1 = np.min(eye_pts, axis=0)
    x2, y2 = np.max(eye_pts, axis=0)
    h, w = image.shape[:2]
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(w, int(x2) + padding)
    y2 = min(h, int(y2) + padding)
    return image[y1:y2, x1:x2]


def draw_debug_overlay(image: np.ndarray, pts: np.ndarray, mask: np.ndarray, iris_infos: List[Dict[str, object]]) -> np.ndarray:
    overlay = image.copy()

    red = np.zeros_like(image)
    red[:, :, 2] = 255
    overlay = np.where(mask[:, :, None] > 0, 0.55 * overlay + 0.45 * red, overlay).astype(np.uint8)

    for info in iris_infos:
        cx, cy = info["center"]
        rx, ry = info["axes"]
        cv2.ellipse(overlay, (cx, cy), (rx, ry), info["angle"], 0, 360, (0, 255, 0), 2)
        cv2.circle(overlay, (cx, cy), 3, (255, 0, 0), -1)
        cv2.putText(overlay, info["name"], (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    # Draw iris contour points.
    for group in IRIS_GROUPS:
        for idx in [group["center"], *group["contour"]]:
            x, y = pts[idx]
            cv2.circle(overlay, (int(x), int(y)), 2, (255, 255, 0), -1)

    return overlay


def save_image(path: Path, image_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image_bgr)


def make_result_grid(results: Dict[str, np.ndarray], pts: np.ndarray, output_path: Path) -> None:
    names = list(results.keys())
    fig, axes = plt.subplots(len(names), 2, figsize=(11, 3.2 * len(names)))
    if len(names) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, name in enumerate(names):
        full_rgb = cv2.cvtColor(results[name], cv2.COLOR_BGR2RGB)
        eye_crop = crop_eye_region(results[name], pts)
        eye_rgb = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)

        axes[row, 0].imshow(full_rgb)
        axes[row, 0].set_title(f"{name} - full")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(eye_rgb)
        axes[row, 1].set_title(f"{name} - eye zoom")
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def run(args: argparse.Namespace) -> None:
    image_path = Path(args.image)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    pts = detect_face_landmarks(image)
    iris_infos = get_iris_infos(pts, cornea_scale=args.cornea_scale)

    ellipse_mask = create_ellipse_mask(image.shape, iris_infos, feather=args.mask_feather)
    center_crop_mask = create_center_crop_mask(image.shape, iris_infos, crop_scale=args.center_crop_scale)

    # Primary comparison: ellipse-based localized cornea/iris mask.
    results_ellipse = {
        "original": image,
        "ellipse_subtle_blur": subtle_blur_baseline(image, ellipse_mask, kernel_size=args.subtle_blur_kernel, blend=args.subtle_blur_blend),
        "ellipse_gaussian_blur": gaussian_blur_baseline(image, ellipse_mask, kernel_size=args.blur_kernel),
        "ellipse_pixelation": pixelation_baseline(image, ellipse_mask, pixel_size=args.pixel_size),
        "ellipse_down_up_noise": down_up_noise_baseline(image, ellipse_mask, pixel_size=args.pixel_size, noise_std=args.noise_std),
        "ellipse_telea_inpaint": telea_inpainting_baseline(image, ellipse_mask, radius=args.inpaint_radius),
        "ellipse_ns_inpaint": navier_stokes_inpainting_baseline(image, ellipse_mask, radius=args.inpaint_radius),
    }

    # Secondary comparison: simple center crop mask, for Jae's suggested baseline.
    results_center = {
        "original": image,
        "center_crop_subtle_blur": subtle_blur_baseline(image, center_crop_mask, kernel_size=args.subtle_blur_kernel, blend=args.subtle_blur_blend),
        "center_crop_telea_inpaint": telea_inpainting_baseline(image, center_crop_mask, radius=args.inpaint_radius),
    }

    save_image(out_dir / "mask_ellipse.png", ellipse_mask)
    save_image(out_dir / "mask_center_crop.png", center_crop_mask)
    save_image(out_dir / "debug_mask_overlay_ellipse.png", draw_debug_overlay(image, pts, ellipse_mask, iris_infos))
    save_image(out_dir / "debug_mask_overlay_center_crop.png", draw_debug_overlay(image, pts, center_crop_mask, iris_infos))

    for name, result in results_ellipse.items():
        save_image(out_dir / f"{name}.png", result)
    for name, result in results_center.items():
        save_image(out_dir / f"{name}.png", result)

    make_result_grid(results_ellipse, pts, out_dir / "baseline_grid_ellipse.png")
    make_result_grid(results_center, pts, out_dir / "baseline_grid_center_crop.png")

    print("Done.")
    print(f"Results saved to: {out_dir.resolve()}")
    print("First inspect: debug_mask_overlay_ellipse.png and baseline_grid_ellipse.png")
    print("If the mask is too small/large, tune --cornea-scale, --mask-feather, and --center-crop-scale.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="IMG_5701.jpg", help="Input image path")
    parser.add_argument("--out", default="outputs_mediapipe", help="Output directory")
    parser.add_argument("--cornea-scale", type=float, default=1.8, help="Ellipse scale relative to detected iris radius")
    parser.add_argument("--center-crop-scale", type=float, default=2.4, help="Center crop size relative to ellipse axes")
    parser.add_argument("--mask-feather", type=int, default=9, help="Small blur before binarizing mask")
    parser.add_argument("--blur-kernel", type=int, default=31)
    parser.add_argument("--subtle-blur-kernel", type=int, default=15)
    parser.add_argument("--subtle-blur-blend", type=float, default=0.65)
    parser.add_argument("--pixel-size", type=int, default=12)
    parser.add_argument("--noise-std", type=float, default=8.0)
    parser.add_argument("--inpaint-radius", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
