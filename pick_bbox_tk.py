from PIL import Image, ImageTk
import tkinter as tk

IMAGE_PATH = "IMG_5701.jpg"  # 실제 이미지 파일명으로 바꾸기
MAX_WIDTH = 1200

img = Image.open(IMAGE_PATH).convert("RGB")
orig_w, orig_h = img.size

scale = MAX_WIDTH / orig_w
display_w = MAX_WIDTH
display_h = int(orig_h * scale)

display_img = img.resize((display_w, display_h))

root = tk.Tk()
root.title("Select eye bounding boxes")

tk_img = ImageTk.PhotoImage(display_img)

canvas = tk.Canvas(root, width=display_w, height=display_h)
canvas.pack()

canvas.create_image(0, 0, anchor="nw", image=tk_img)

selections = []
current_rect = None
start_x = None
start_y = None

labels = ["left_eye", "right_eye"]
instruction = tk.Label(root, text="Drag a box around LEFT eye")
instruction.pack()


def on_mouse_down(event):
    global start_x, start_y, current_rect
    start_x, start_y = event.x, event.y

    if current_rect is not None:
        canvas.delete(current_rect)

    current_rect = canvas.create_rectangle(
        start_x,
        start_y,
        start_x,
        start_y,
        outline="red",
        width=2,
    )


def on_mouse_drag(event):
    if current_rect is not None:
        canvas.coords(current_rect, start_x, start_y, event.x, event.y)


def on_mouse_up(event):
    global current_rect

    x1, y1 = start_x, start_y
    x2, y2 = event.x, event.y

    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    # 원본 이미지 좌표로 변환
    orig_x1 = int(x1 / scale)
    orig_y1 = int(y1 / scale)
    orig_x2 = int(x2 / scale)
    orig_y2 = int(y2 / scale)

    selections.append((orig_x1, orig_y1, orig_x2, orig_y2))

    idx = len(selections)

    if idx < 2:
        instruction.config(text="Drag a box around RIGHT eye")
    else:
        root.quit()


canvas.bind("<ButtonPress-1>", on_mouse_down)
canvas.bind("<B1-Motion>", on_mouse_drag)
canvas.bind("<ButtonRelease-1>", on_mouse_up)

root.mainloop()
root.destroy()

print("\nCopy this into cornea_baseline.py:\n")
print("EYE_BBOXES = {")
print(f'    "left_eye":  {selections[0]},')
print(f'    "right_eye": {selections[1]},')
print("}")