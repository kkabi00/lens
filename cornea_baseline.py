
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# 1. Config
# =========================

IMAGE_PATH = "IMG_5701.jpg"   # 여기에 base image 경로 넣기
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# 눈 주변 bbox: (x1, y1, x2, y2)
# 일단 대충 잡고 결과 보면서 조정하면 됨
EYE_BBOXES = {
    "left_eye":  (1527, 2120, 1942, 2384),
    "right_eye": (2438, 2209, 2831, 2474),
}

# 각막/홍채 반사 mask를 ellipse로 대충 지정
# bbox 내부 좌표 기준: (cx, cy, rx, ry, angle)
# cx, cy는 bbox crop 안에서의 중심 좌표
CORNEA_MASKS = {
    "left_eye":  [
        (60, 45, 18, 14, 0),   # 필요하면 여러 개 추가 가능
    ],
    "right_eye": [
        (55, 45, 18, 14, 0),
    ],
}

# baseline parameter
BLUR_KERNEL = 81
PIXEL_SIZE = 20
NOISE_STD = 30
INPAINT_RADIUS = 7

# =========================
# 2. Utility functions
# =========================

def ensure_odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def create_mask_for_bbox(bbox, ellipses, image_shape):
    """
    Full image 크기의 binary mask 생성.
    mask 영역은 255, 나머지는 0.
    """
    x1, y1, x2, y2 = bbox
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for cx, cy, rx, ry, angle in ellipses:
        center = (x1 + cx, y1 + cy)
        axes = (rx, ry)
        cv2.ellipse(
            mask,
            center=center,
            axes=axes,
            angle=angle,
            startAngle=0,
            endAngle=360,
            color=255,
            thickness=-1,
        )

    return mask


def combine_masks(image, eye_bboxes, cornea_masks):
    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # DEBUG: 일단 눈 bbox 전체에 효과 적용
    for eye_name, bbox in eye_bboxes.items():
        x1, y1, x2, y2 = bbox
        cv2.rectangle(full_mask, (x1, y1), (x2, y2), 255, thickness=-1)

    return full_mask


def apply_masked_region(original, processed, mask):
    """
    mask 영역만 processed 이미지로 치환.
    """
    result = original.copy()
    mask_bool = mask > 0
    result[mask_bool] = processed[mask_bool]
    return result


# =========================
# 3. Baseline methods
# =========================

def gaussian_blur_baseline(image, mask, kernel_size=31):
    kernel_size = ensure_odd(kernel_size)
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return apply_masked_region(image, blurred, mask)


def pixelation_baseline(image, mask, pixel_size=8):
    h, w = image.shape[:2]

    small = cv2.resize(
        image,
        (max(1, w // pixel_size), max(1, h // pixel_size)),
        interpolation=cv2.INTER_LINEAR,
    )
    pixelated = cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )

    return apply_masked_region(image, pixelated, mask)


def down_up_noise_baseline(image, mask, pixel_size=8, noise_std=10):
    h, w = image.shape[:2]

    small = cv2.resize(
        image,
        (max(1, w // pixel_size), max(1, h // pixel_size)),
        interpolation=cv2.INTER_AREA,
    )
    restored = cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
    noisy = restored.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return apply_masked_region(image, noisy, mask)


def telea_inpainting_baseline(image, mask, radius=5):
    return cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)


def navier_stokes_inpainting_baseline(image, mask, radius=5):
    return cv2.inpaint(image, mask, radius, cv2.INPAINT_NS)


# =========================
# 4. Visualization
# =========================

def crop_eye_region(image, eye_bboxes, padding=30):
    """
    양쪽 눈이 포함되는 하나의 crop 반환.
    결과 비교용 zoom image.
    """
    xs1, ys1, xs2, ys2 = [], [], [], []

    for bbox in eye_bboxes.values():
        x1, y1, x2, y2 = bbox
        xs1.append(x1)
        ys1.append(y1)
        xs2.append(x2)
        ys2.append(y2)

    h, w = image.shape[:2]
    x1 = max(0, min(xs1) - padding)
    y1 = max(0, min(ys1) - padding)
    x2 = min(w, max(xs2) + padding)
    y2 = min(h, max(ys2) + padding)

    return image[y1:y2, x1:x2]


def save_image_bgr(path, image_bgr):
    cv2.imwrite(str(path), image_bgr)


def make_result_grid(results, eye_bboxes):
    """
    full image와 eye zoom을 함께 보여주는 grid 저장.
    """
    names = list(results.keys())

    fig, axes = plt.subplots(len(names), 2, figsize=(10, 4 * len(names)))

    if len(names) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, name in enumerate(names):
        image_bgr = results[name]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        eye_crop_bgr = crop_eye_region(image_bgr, eye_bboxes)
        eye_crop_rgb = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGR2RGB)

        axes[row, 0].imshow(image_rgb)
        axes[row, 0].set_title(f"{name} - full image")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(eye_crop_rgb)
        axes[row, 1].set_title(f"{name} - eye zoom")
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "baseline_grid.png", dpi=200)
    plt.close()


def draw_debug_overlay(image, mask, eye_bboxes):
    overlay = image.copy()

    # mask 영역 빨간색으로 표시
    red = np.zeros_like(image)
    red[:, :, 2] = 255
    overlay = np.where(mask[:, :, None] > 0, 0.6 * overlay + 0.4 * red, overlay)
    overlay = overlay.astype(np.uint8)

    # bbox 표시
    for name, bbox in eye_bboxes.items():
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            overlay,
            name,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return overlay


# =========================
# 5. Main
# =========================

def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    mask = combine_masks(image, EYE_BBOXES, CORNEA_MASKS)
    
    print("image shape:", image.shape)
    print("mask pixels:", np.sum(mask > 0))

    debug_overlay = draw_debug_overlay(image, mask, EYE_BBOXES)
    save_image_bgr(OUTPUT_DIR / "debug_mask_overlay.png", debug_overlay)
    save_image_bgr(OUTPUT_DIR / "mask.png", mask)

    results = {
        "original": image,
        "gaussian_blur": gaussian_blur_baseline(
            image,
            mask,
            kernel_size=BLUR_KERNEL,
        ),
        "pixelation": pixelation_baseline(
            image,
            mask,
            pixel_size=PIXEL_SIZE,
        ),
        "down_up_noise": down_up_noise_baseline(
            image,
            mask,
            pixel_size=PIXEL_SIZE,
            noise_std=NOISE_STD,
        ),
        "telea_inpainting": telea_inpainting_baseline(
            image,
            mask,
            radius=INPAINT_RADIUS,
        ),
        "navier_stokes_inpainting": navier_stokes_inpainting_baseline(
            image,
            mask,
            radius=INPAINT_RADIUS,
        ),
    }

    for name, result in results.items():
        save_image_bgr(OUTPUT_DIR / f"{name}.png", result)

    make_result_grid(results, EYE_BBOXES)

    print(f"Done. Results saved to: {OUTPUT_DIR.resolve()}")
    print("Check debug_mask_overlay.png first. If mask/bbox is off, adjust EYE_BBOXES and CORNEA_MASKS.")


if __name__ == "__main__":
    main()