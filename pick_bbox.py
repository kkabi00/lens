import cv2

IMAGE_PATH = "IMG_5701.jpg"  # 네 이미지 이름으로 맞추기
MAX_WIDTH = 1200

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

h, w = image.shape[:2]
scale = MAX_WIDTH / w
resized = cv2.resize(image, (MAX_WIDTH, int(h * scale)))

print("Drag a box around LEFT eye, then press ENTER or SPACE.")
left_roi = cv2.selectROI("Select LEFT eye", resized, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select LEFT eye")

print("Drag a box around RIGHT eye, then press ENTER or SPACE.")
right_roi = cv2.selectROI("Select RIGHT eye", resized, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select RIGHT eye")

def scale_roi_back(roi):
    x, y, rw, rh = roi
    x1 = int(x / scale)
    y1 = int(y / scale)
    x2 = int((x + rw) / scale)
    y2 = int((y + rh) / scale)
    return x1, y1, x2, y2

left_bbox = scale_roi_back(left_roi)
right_bbox = scale_roi_back(right_roi)

print("\nCopy this into cornea_baseline.py:\n")
print("EYE_BBOXES = {")
print(f'    "left_eye":  {left_bbox},')
print(f'    "right_eye": {right_bbox},')
print("}")