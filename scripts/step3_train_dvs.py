"""Step 3: train the tutorial-grade SNN on DVS128Gesture, on Modal.

Architecture: SpikingJelly's canonical DVSGesture net. 5 conv blocks
(Conv-BN-LIF-MaxPool), then FC-LIF-FC-LIF into an 11-class vote.
Every LIF uses the atan surrogate you tested in step 2.

Checkpointing: saved to a persistent Modal volume after EVERY epoch.
Rerunning the same command resumes from the last epoch automatically.
Nothing is lost to crashes, preemption, or closing your laptop
(use --detach for the laptop case).

One-time setup:
  pip install modal
  modal setup
  modal volume create dvs128-data
  modal volume put dvs128-data "D:\\datasets\\dvs128gesture\\download" /DVS128Gesture/download

Then (from the repo root):
  modal run scripts/step3_train_dvs.py --mode prepare   # CPU, once, ~30-60 min
  modal run --detach scripts/step3_train_dvs.py         # GPU, resumable, seed 0
  modal run --detach scripts/step3_train_dvs.py --seed 1   # any other seed

Seeds and checkpoints:
  seed 0 keeps the original layout: /checkpoints/{last,best}.pt, metrics.csv.
  seed S > 0 writes the same files to /checkpoints/seedS/.
  The seed controls init, shuffling and dropout (torch, numpy, random).
  cuDNN autotuning stays nondeterministic, so two runs of one seed are
  close but not bit-identical.
  best.pt is still written (highest test accuracy so far, ties -> latest),
  but it is selected ON THE TEST SET. Steps 5-7 evaluate last.pt, the
  fixed-budget epoch-64 checkpoint, by default. Do not report best.pt.

Watch progress in the Modal dashboard, or fetch results later:
  modal volume get dvs128-data /checkpoints/metrics.csv results/data/seed0/step3_metrics.csv
  modal volume get dvs128-data /checkpoints/seed1/metrics.csv results/data/seed1/step3_metrics.csv
"""

import modal

app = modal.App("dvs128-snn")

vol = modal.Volume.from_name("dvs128-data", create_if_missing=True)
DATA = "/data"
ROOT = f"{DATA}/DVS128Gesture"      # must contain download/DvsGesture.tar.gz etc.
CKPT = f"{DATA}/checkpoints"


def ckpt_dir(seed: int) -> str:
    """seed 0 keeps the original folder; other seeds get a subfolder."""
    return CKPT if seed == 0 else f"{CKPT}/seed{seed}"


image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.0",
    "torchvision==0.19.0",
    "spikingjelly==0.0.0.0.14",     # last stable PyPI release, known-good pair
    "numpy<2",
    "tqdm",
)


@app.function(image=image, volumes={DATA: vol}, timeout=4 * 3600, cpu=8)
def prepare():
    """Extract the tar and convert events to T=16 frames. Runs once.
    The converted frames live on the volume, so training never redoes this."""
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

    for train in (True, False):
        ds = DVS128Gesture(
            ROOT, train=train, data_type="frame",
            frames_number=16, split_by="number",
        )
        print(("train" if train else "test"), "samples:", len(ds))
    vol.commit()                     # persist the converted frames
    print("prepare done")


def build_net():
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
        *block(2, 128),              # 128x128 -> 64x64
        *block(128, 128),            # -> 32x32
        *block(128, 128),            # -> 16x16
        *block(128, 128),            # -> 8x8
        *block(128, 128),            # -> 4x4
        layer.Flatten(),
        layer.Dropout(0.5),
        layer.Linear(128 * 4 * 4, 512),
        lif(),
        layer.Dropout(0.5),
        layer.Linear(512, 110),
        lif(),
        layer.VotingLayer(10),       # 110 spiking outputs vote for 11 classes
    )


@app.function(image=image, gpu="A10G", volumes={DATA: vol}, timeout=6 * 3600)
def train_remote(epochs: int = 64, batch: int = 16, lr: float = 1e-3, T: int = 16,
                 seed: int = 0):
    import json
    import os
    import random
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    from spikingjelly.activation_based import functional
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
    from torch.utils.data import DataLoader

    device = "cuda"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    run_dir = ckpt_dir(seed)

    def make_loader(is_train):
        ds = DVS128Gesture(ROOT, train=is_train, data_type="frame",
                           frames_number=T, split_by="number")
        return DataLoader(ds, batch_size=batch, shuffle=is_train,
                          num_workers=4, pin_memory=True, drop_last=is_train)

    train_loader, test_loader = make_loader(True), make_loader(False)

    net = build_net().to(device)
    functional.set_step_mode(net, "m")           # process all T steps at once
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # ---- resume if a checkpoint exists ----
    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/run.json", "w") as f:
        json.dump({"model": "snn", "seed": seed, "epochs": epochs,
                   "batch": batch, "lr": lr, "T": T, "loss": "mse_onehot",
                   "optimizer": "adam", "schedule": "cosine"}, f, indent=1)
    last, best_path = f"{run_dir}/last.pt", f"{run_dir}/best.pt"
    start_epoch, best_acc = 0, 0.0
    if os.path.exists(last):
        ck = torch.load(last, map_location=device)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best_acc = ck["epoch"] + 1, ck["best_acc"]
        print(f"resumed from epoch {ck['epoch']}, best acc {best_acc:.4f}")

    def run_epoch(loader, training):
        net.train(training)
        total, correct, loss_sum = 0, 0, 0.0
        with torch.set_grad_enabled(training):
            for frame, label in loader:
                # frame: [B, T, 2, 128, 128] -> [T, B, 2, 128, 128]
                frame = frame.to(device).float().transpose(0, 1)
                label = label.to(device)
                out_fr = net(frame).mean(0)      # rate readout over T
                loss = F.mse_loss(out_fr, F.one_hot(label, 11).float())
                if training:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                functional.reset_net(net)        # the bug from step 2, avoided
                total += label.numel()
                correct += (out_fr.argmax(1) == label).sum().item()
                loss_sum += loss.item() * label.numel()
        return loss_sum / total, correct / total

    metrics_path = f"{run_dir}/metrics.csv"
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w") as f:
            f.write("epoch,train_loss,train_acc,test_loss,test_acc,epoch_time_s\n")

    te_acc = float("nan")                  # stays nan if the run was already complete
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(train_loader, True)
        te_loss, te_acc = run_epoch(test_loader, False)
        sched.step()
        epoch_time = time.time() - t0
        best_acc = max(best_acc, te_acc)
        print(f"epoch {epoch:3d}  train {tr_acc:.4f}  test {te_acc:.4f}"
              f"  best {best_acc:.4f}")

        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{tr_loss:.4f},{tr_acc:.4f},"
                    f"{te_loss:.4f},{te_acc:.4f},{epoch_time:.1f}\n")
        state = {"net": net.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "epoch": epoch,
                 "best_acc": best_acc}
        torch.save(state, last)
        if te_acc >= best_acc:             # test-selected: kept for reference only
            torch.save(state, best_path)
        vol.commit()                             # persist every epoch

    print(f"done. seed {seed}: last-epoch test acc {te_acc:.4f} (report this), "
          f"best test acc {best_acc:.4f} (test-selected, do not report)")


@app.local_entrypoint()
def main(mode: str = "train", epochs: int = 64, seed: int = 0):
    if mode == "prepare":
        prepare.remote()
    else:
        train_remote.remote(epochs=epochs, seed=seed)
