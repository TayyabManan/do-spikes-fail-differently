"""Step 5 input: dump per-sample test outputs for both models.

Loads the best SNN (checkpoints/best.pt) and best ANN
(checkpoints_ann/best.pt), runs the full DVS128Gesture test set
through each, and saves one npz to the volume:

  /eval/test_outputs.npz
    labels   [N]        ground-truth class ids
    snn_out  [N, 11]    SNN rate outputs, averaged over T
    ann_out  [N, 11]    ANN outputs, averaged over T

Raw outputs, no softmax: step 5 decides how to turn them into
confidences, so keep every option open here.

Run:
  modal run scripts/step5_dump_logits.py
Fetch:
  modal volume get dvs128-data /eval/test_outputs.npz results/data/test_outputs.npz

Prints test accuracy for both as a checksum: expect ~0.9306 (SNN)
and ~0.9653 (ANN). If those numbers differ, stop and tell me.
"""

import modal

app = modal.App("dvs128-dump")

vol = modal.Volume.from_name("dvs128-data", create_if_missing=True)
DATA = "/data"
ROOT = f"{DATA}/DVS128Gesture"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.0",
    "torchvision==0.19.0",
    "spikingjelly==0.0.0.0.14",
    "numpy<2",
    "tqdm",
)


def build_snn():
    """Copied verbatim from step 3."""
    import torch.nn as nn
    from spikingjelly.activation_based import layer, neuron, surrogate

    def lif():
        return neuron.LIFNode(
            surrogate_function=surrogate.ATan(), detach_reset=True
        )

    def block(cin, cout):
        return [
            layer.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(cout),
            lif(),
            layer.MaxPool2d(2, 2),
        ]

    return nn.Sequential(
        *block(2, 128), *block(128, 128), *block(128, 128),
        *block(128, 128), *block(128, 128),
        layer.Flatten(),
        layer.Dropout(0.5),
        layer.Linear(128 * 4 * 4, 512),
        lif(),
        layer.Dropout(0.5),
        layer.Linear(512, 110),
        lif(),
        layer.VotingLayer(10),
    )


def build_ann():
    """Copied verbatim from step 4 (voting is applied in forward)."""
    import torch.nn as nn

    def block(cin, cout):
        return [
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        ]

    return nn.Sequential(
        *block(2, 128), *block(128, 128), *block(128, 128),
        *block(128, 128), *block(128, 128),
        nn.Flatten(),
        nn.Dropout(0.5),
        nn.Linear(128 * 4 * 4, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 110),
        nn.ReLU(),
    )


@app.function(image=image, gpu="A10G", volumes={DATA: vol}, timeout=1800)
def dump(T: int = 16, batch: int = 16):
    import os

    import numpy as np
    import torch
    from spikingjelly.activation_based import functional
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
    from torch.utils.data import DataLoader

    device = "cuda"
    ds = DVS128Gesture(ROOT, train=False, data_type="frame",
                       frames_number=T, split_by="number")
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4)

    # ---- SNN ----
    snn = build_snn().to(device)
    functional.set_step_mode(snn, "m")
    ck = torch.load(f"{DATA}/checkpoints/best.pt", map_location=device)
    snn.load_state_dict(ck["net"])
    snn.eval()

    # ---- ANN ----
    ann = build_ann().to(device)
    ck = torch.load(f"{DATA}/checkpoints_ann/best.pt", map_location=device)
    ann.load_state_dict(ck["net"])
    ann.eval()

    labels, snn_out, ann_out = [], [], []
    with torch.no_grad():
        for frame, label in loader:
            frame = frame.to(device).float()
            labels.append(label.numpy())

            # SNN: [T, B, ...], rate readout, reset between batches
            out = snn(frame.transpose(0, 1)).mean(0)
            functional.reset_net(snn)
            snn_out.append(out.cpu().numpy())

            # ANN: fold T into batch, vote, average over T
            B, T_ = frame.shape[0], frame.shape[1]
            x = frame.reshape(B * T_, *frame.shape[2:])
            out = ann(x).view(B, T_, 11, 10).mean(3).mean(1)
            ann_out.append(out.cpu().numpy())

    labels = np.concatenate(labels)
    snn_out = np.concatenate(snn_out)
    ann_out = np.concatenate(ann_out)

    snn_acc = (snn_out.argmax(1) == labels).mean()
    ann_acc = (ann_out.argmax(1) == labels).mean()
    print(f"N = {len(labels)}")
    print(f"SNN test acc: {snn_acc:.4f}   (expect ~0.9306)")
    print(f"ANN test acc: {ann_acc:.4f}   (expect ~0.9653)")

    os.makedirs(f"{DATA}/eval", exist_ok=True)
    np.savez(f"{DATA}/eval/test_outputs.npz",
             labels=labels, snn_out=snn_out, ann_out=ann_out)
    vol.commit()
    print("wrote /eval/test_outputs.npz")


@app.local_entrypoint()
def main():
    dump.remote()
