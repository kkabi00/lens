from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import math

# ============================================================
# ER-001: Eye Reflection Similar-Object Substitution
# Goal:
#   Replace only the tiny reflection patterns inside the pupils
#   with a similar but different window/grid-like reflection.
# ============================================================

INPUT_PATH = Path("/mnt/data/b93caaa3-cb40-4e2a-a811-9ee1a76eae8b.png")
OUTPUT_PATH = Path("/mnt/data/er001_reflection_substitution.png")
COMPARISON_PATH = Path("/mnt/data/er001_eye_crop_comparison.png")

# Manual pupil/reflection masks.
# Coordinates are in the full-resolution image coordinate system.
# Tune cx/cy/rx/ry if the mask is not aligned enough.
REFLECTION_SPECS = [
    {
        "name": "left_eye_reflection",
        "cx": 616,
        "cy": 807,
        "rx": 34,
        "ry": 25,
        "angle_deg": -4,
        "alpha": 0.48,
        "variant": 0,
    },
    {
        "name": "right_eye_reflection",
        "cx": 950,
        "cy": 837,
        "rx": 36,
        "ry": 27,
        "angle_deg": 5,
        "alpha": 0.48,
        "variant": 1,
    },
]


def make_soft_ellipse_mask(width, height, rx, ry, feather=5):
    """Create a soft elliptical mask centered in a patch."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    cx, cy = width // 2, height // 2
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return np.asarray(mask).astype(np.float32) / 255.0


def rotate_image_keep_size(img, angle_deg):
    """Rotate a PIL image without expanding canvas."""
    return img.rotate(angle_deg, resample=Image.Resampling.BICUBIC, expand=False)


def make_window_grid_patch(width, height, variant=0):
    """
    Build a tiny synthetic reflection patch that resembles a different
    rectangular window/grid reflection. It is intentionally subtle.
    """
    # Base patch: dark pupil-like background with cool reflection tint.
    patch = Image.new("RGB", (width, height), (32, 35, 40))
    draw = ImageDraw.Draw(patch, "RGBA")

    # Slight bluish rectangular reflection.
    if variant == 0:
        rect = (int(width * 0.18), int(height * 0.20), int(width * 0.82), int(height * 0.78))
        fill = (135, 170, 190, 110)
        line = (220, 235, 240, 125)
        shadow = (15, 18, 22, 100)
    else:
        rect = (int(width * 0.12), int(height * 0.18), int(width * 0.88), int(height * 0.80))
        fill = (120, 165, 185, 105)
        line = (230, 238, 240, 120)
        shadow = (20, 20, 24, 100)

    # Main reflected window area.
    draw.rounded_rectangle(rect, radius=4, fill=fill)

    x0, y0, x1, y1 = rect

    # Different grid/frame pattern depending on variant.
    if variant == 0:
        # Two vertical frame bars and one horizontal bar.
        for frac in [0.38, 0.66]:
            x = int(x0 + (x1 - x0) * frac)
            draw.line((x, y0, x, y1), fill=line, width=2)
        y = int(y0 + (y1 - y0) * 0.48)
        draw.line((x0, y, x1, y), fill=line, width=1)
    else:
        # One thicker vertical edge + three thin horizontal bands.
        x = int(x0 + (x1 - x0) * 0.52)
        draw.line((x, y0, x, y1), fill=line, width=3)
        for frac in [0.30, 0.55, 0.74]:
            y = int(y0 + (y1 - y0) * frac)
            draw.line((x0, y, x1, y), fill=line, width=1)

    # Add a small dark camera-like silhouette but in a different location
    # so the reflection remains plausible while being counterfactual.
    if variant == 0:
        draw.rectangle(
            (int(width * 0.55), int(height * 0.46), int(width * 0.72), int(height * 0.66)),
            fill=shadow,
        )
    else:
        draw.rectangle(
            (int(width * 0.24), int(height * 0.42), int(width * 0.40), int(height * 0.64)),
            fill=shadow,
        )

    # Mild blur makes it look like a corneal reflection rather than a pasted object.
    patch = patch.filter(ImageFilter.GaussianBlur(radius=1.2))

    # Add subtle noise for natural texture.
    arr = np.asarray(patch).astype(np.float32)
    rng = np.random.default_rng(123 + variant)
    noise = rng.normal(0, 3.0, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_reflection_patch(img, spec):
    """
    Apply synthetic reflection patch to a local elliptical region.
    """
    cx, cy = spec["cx"], spec["cy"]
    rx, ry = spec["rx"], spec["ry"]
    alpha = spec["alpha"]

    pad = 10
    x0 = max(0, cx - rx - pad)
    y0 = max(0, cy - ry - pad)
    x1 = min(img.width, cx + rx + pad)
    y1 = min(img.height, cy + ry + pad)

    region = img.crop((x0, y0, x1, y1)).convert("RGB")
    w, h = region.size

    patch = make_window_grid_patch(w, h, variant=spec.get("variant", 0))
    patch = rotate_image_keep_size(patch, spec.get("angle_deg", 0))

    mask = make_soft_ellipse_mask(w, h, rx=rx, ry=ry, feather=5)
    mask = mask[..., None] * alpha

    region_arr = np.asarray(region).astype(np.float32)
    patch_arr = np.asarray(patch).astype(np.float32)

    # Blend patch into local region.
    blended = region_arr * (1 - mask) + patch_arr * mask

    # Preserve the darkest pupil structure slightly by keeping local contrast.
    # This prevents the edit from looking like a flat sticker.
    gray = region_arr.mean(axis=2, keepdims=True)
    contrast = (gray - gray.mean()) * 0.10
    blended = np.clip(blended + contrast, 0, 255).astype(np.uint8)

    out_region = Image.fromarray(blended)
    img.paste(out_region, (x0, y0))
    return img


def make_eye_crop_comparison(original, edited, path):
    """Save side-by-side eye crop comparison."""
    crop_box = (550, 720, 1120, 950)
    o = original.crop(crop_box)
    e = edited.crop(crop_box)

    canvas = Image.new("RGB", (o.width * 2 + 30, o.height + 50), "white")
    draw = ImageDraw.Draw(canvas)
    canvas.paste(o, (0, 40))
    canvas.paste(e, (o.width + 30, 40))
    draw.text((10, 10), "Original eye crop", fill=(0, 0, 0))
    draw.text((o.width + 40, 10), "Edited eye crop", fill=(0, 0, 0))
    canvas.save(path)


def main():
    original = Image.open(INPUT_PATH).convert("RGB")
    edited = original.copy()

    for spec in REFLECTION_SPECS:
        edited = apply_reflection_patch(edited, spec)

    edited.save(OUTPUT_PATH)
    make_eye_crop_comparison(original, edited, COMPARISON_PATH)

    print(f"Saved edited image: {OUTPUT_PATH}")
    print(f"Saved eye crop comparison: {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
