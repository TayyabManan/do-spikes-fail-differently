"""Step 5 input: dump per-sample test outputs for both models.

Loads one checkpoint per model, runs the full DVS128Gesture test set
through each, and saves one npz to the volume:

  /eval/seed{S}/test_outputs.npz         (--ckpt last, the default)
  /eval/seed{S}/test_outputs_best.npz    (--ckpt best, reference only)
    labels     [N]        ground-truth class ids
    snn_out    [N, 11]    SNN rate outputs, averaged over T
    ann_out    [N, 11]    ANN outputs, averaged over T
    seed, ckpt, snn_epoch, ann_epoch     provenance scalars

Raw outputs, no softmax: step 5 decides how to turn them into
confidences, so keep every option open here.

Which checkpoint. last.pt is the fixed-budget epoch-64 weights and is
what the paper reports. best.pt is the epoch with the highest TEST
accuracy, i.e. selected on the test set, so it is optimistic (seed 0:
SNN 0.9306 at epoch 60 vs 0.9167 at epoch 63; the ANN's best equals its
last). It stays available for the appendix comparison only.

Checkpoint layout (steps 3 and 4): seed 0 lives in /checkpoints and
/checkpoints_ann, seed S > 0 in /checkpoints/seedS and /checkpoints_ann/seedS.

Run:
  modal run scripts/step5_dump_logits.py --seed 0
  modal run scripts/step5_dump_logits.py --seed 1
  modal run scripts/step5_dump_logits.py --seed 0 --ckpt best
Fetch:
  modal volume get dvs128-data /eval/seed0/test_outputs.npz results/data/seed0/test_outputs.npz

Checksum: the script re-reads each run's metrics.csv and compares the
accuracy it just measured against the test_acc logged at the checkpoint's
epoch. A gap above one sample (1/288) prints a WARNING. Stop and
investigate if you see one.

The original single-seed dump (/eval/test_outputs.npz: best.pt, SNN
0.9306, ANN 0.9653) is left untouched. The live demo still reads it.
"""

import modal

app = modal.App("dvs128-dump")

vol = modal.Volume.from_name("dvs128-data", create_if_missing=True)
DATA = "/data"
ROOT = f"{DATA}/DVS128Gesture"
CKPT_SNN = f"{DATA}/checkpoints"
CKPT_ANN = f"{DATA}/checkpoints_ann"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.0",
    "torchvision==0.19.0",
    "spikingjelly==0.0.0.0.14",
    "numpy<2",
    "tqdm",
)


def ckpt_dir(base: str, seed: int) -> str:
    """Mirror of steps 3/4: seed 0 is the base folder, others a subfolder."""
    return base if seed == 0 else f"{base}/seed{seed}"


def eval_dir(seed: int) -> str:
    return f"{DATA}/eval/seed{seed}"


def out_name(stem: str, ckpt: str) -> str:
    return f"{stem}.npz" if ckpt == "last" else f"{stem}_{ckpt}.npz"


def logged_test_acc(run_dir: str, epoch: int):
    """test_acc that training logged at this epoch, or None if unknown."""
    import csv
    import os

    path = f"{run_dir}/metrics.csv"
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["epoch"]) == epoch:
                return float(row["test_acc"])
    return None


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
def dump(T: int = 16, batch: int = 16, seed: int = 0, ckpt: str = "last"):
    import os

    import numpy as np
    import torch
    from spikingjelly.activation_based import functional
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
    from torch.utils.data import DataLoader

    assert ckpt in ("last", "best"), ckpt
    device = "cuda"
    ds = DVS128Gesture(ROOT, train=False, data_type="frame",
                       frames_number=T, split_by="number")
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4)

    snn_dir, ann_dir = ckpt_dir(CKPT_SNN, seed), ckpt_dir(CKPT_ANN, seed)

    # ---- SNN ----
    snn = build_snn().to(device)
    functional.set_step_mode(snn, "m")
    ck = torch.load(f"{snn_dir}/{ckpt}.pt", map_location=device)
    snn.load_state_dict(ck["net"])
    snn_epoch = int(ck["epoch"])
    snn.eval()

    # ---- ANN ----
    ann = build_ann().to(device)
    ck = torch.load(f"{ann_dir}/{ckpt}.pt", map_location=device)
    ann.load_state_dict(ck["net"])
    ann_epoch = int(ck["epoch"])
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
    n = len(labels)

    print(f"seed {seed}, checkpoint {ckpt}.pt, N = {n}")
    for name, out, run_dir, epoch in (("SNN", snn_out, snn_dir, snn_epoch),
                                      ("ANN", ann_out, ann_dir, ann_epoch)):
        acc = (out.argmax(1) == labels).mean()
        logged = logged_test_acc(run_dir, epoch)
        flag = ""
        if logged is not None and abs(acc - logged) > 1.0 / n + 1e-6:
            flag = "   WARNING: differs from metrics.csv by more than one sample"
        logged_s = "unknown" if logged is None else f"{logged:.4f}"
        print(f"{name} test acc: {acc:.4f}   (epoch {epoch}, "
              f"metrics.csv logged {logged_s}){flag}")

    os.makedirs(eval_dir(seed), exist_ok=True)
    path = f"{eval_dir(seed)}/{out_name('test_outputs', ckpt)}"
    np.savez(path, labels=labels, snn_out=snn_out, ann_out=ann_out,
             seed=seed, ckpt=ckpt, snn_epoch=snn_epoch, ann_epoch=ann_epoch)
    vol.commit()
    print(f"wrote {path.replace(DATA, '')}")


@app.local_entrypoint()
def main(seed: int = 0, ckpt: str = "last"):
    dump.remote(seed=seed, ckpt=ckpt)
