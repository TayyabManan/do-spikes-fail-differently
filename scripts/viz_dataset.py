"""Dataset visualization: what DVS128Gesture actually looks like.

Renders from the cached T=16 frame tensors on the volume (CPU, no GPU):
  1. dataset_grid.png  one test sample per class, 8 of 16 timesteps.
     Orange = ON events (brightness up), blue = OFF events (down).
  2. dataset_gestures.gif  the same 11 samples animated over all 16
     frames, tiled 4x3.

Class names come from gesture_mapping.csv in the dataset download.

Run:
  modal run scripts/viz_dataset.py
Fetch:
  modal volume get dvs128-data /eval/dataset_grid.png results/figures/dataset_grid.png
  modal volume get dvs128-data /eval/dataset_gestures.gif results/figures/dataset_gestures.gif
"""

import modal

app = modal.App("dvs128-viz")

vol = modal.Volume.from_name("dvs128-data", create_if_missing=True)
DATA = "/data"
ROOT = f"{DATA}/DVS128Gesture"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.0", "torchvision==0.19.0",
    "spikingjelly==0.0.0.0.14", "numpy<2", "matplotlib", "pillow",
)

# ON events drawn orange, OFF events blue (matches the report palette)
ON_RGB = (1.00, 0.55, 0.10)
OFF_RGB = (0.20, 0.47, 0.75)


def frame_to_rgb(frame_2hw, gain=0.5):
    """[2, 128, 128] counts -> [128, 128, 3] float RGB on black."""
    import numpy as np

    off = np.clip(frame_2hw[0] * gain, 0, 1)
    on = np.clip(frame_2hw[1] * gain, 0, 1)
    img = np.zeros((*frame_2hw.shape[1:], 3), dtype=np.float32)
    for ch in range(3):
        img[..., ch] = on * ON_RGB[ch] + off * OFF_RGB[ch]
    return np.clip(img, 0, 1)


def class_names():
    import csv
    import os

    path = f"{ROOT}/download/gesture_mapping.csv"
    if not os.path.exists(path):
        return [f"class {i}" for i in range(11)]
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if len(r) >= 2 and r[1].strip().isdigit()]
    # csv maps name -> label 1..11; SpikingJelly labels are 0..10
    names = {int(r[1]) - 1: r[0].strip() for r in rows}
    return [names.get(i, f"class {i}") for i in range(11)]


@app.function(image=image, volumes={DATA: vol}, timeout=1800, cpu=4)
def render():
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

    ds = DVS128Gesture(ROOT, train=False, data_type="frame",
                       frames_number=16, split_by="number")
    names = class_names()

    # first test sample of each class
    samples = {}
    for frame, label in ds:
        if int(label) not in samples:
            samples[int(label)] = np.asarray(frame, dtype=np.float32)
        if len(samples) == 11:
            break

    # ---- static grid: 11 classes x 8 timesteps ----
    ts = list(range(0, 16, 2))
    fig, axes = plt.subplots(11, len(ts), figsize=(len(ts) * 1.5, 11 * 1.55))
    for row in range(11):
        for col, t in enumerate(ts):
            ax = axes[row, col]
            ax.imshow(frame_to_rgb(samples[row][t]))
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t = {t}", fontsize=9)
            if col == 0:
                ax.set_ylabel(names[row], fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=4)
    fig.suptitle("DVS128Gesture test samples, one per class, 8 of 16 frames\n"
                 "orange = ON events, blue = OFF events", fontsize=11)
    fig.tight_layout(rect=[0.02, 0, 1, 0.96])
    os.makedirs(f"{DATA}/eval", exist_ok=True)
    fig.savefig(f"{DATA}/eval/dataset_grid.png", dpi=110,
                facecolor="white", bbox_inches="tight")
    print("wrote /eval/dataset_grid.png")

    # ---- animated gif: 4x3 tiling, 16 frames ----
    cell, pad = 128, 6
    cols, rows = 4, 3
    W = cols * cell + (cols + 1) * pad
    H = rows * (cell + 14) + (rows + 1) * pad
    gif_frames = []
    for t in range(16):
        canvas = np.zeros((H, W, 3), dtype=np.float32)
        for k in range(11):
            r, c = divmod(k, cols)
            y = pad + r * (cell + 14 + pad)
            x = pad + c * (cell + pad)
            canvas[y:y + cell, x:x + cell] = frame_to_rgb(samples[k][t])
        gif_frames.append(Image.fromarray((canvas * 255).astype(np.uint8)))
    gif_frames[0].save(f"{DATA}/eval/dataset_gestures.gif", save_all=True,
                       append_images=gif_frames[1:], duration=140, loop=0)
    print("wrote /eval/dataset_gestures.gif")
    vol.commit()


@app.local_entrypoint()
def main():
    render.remote()
