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
  modal run --detach scripts/step3_train_dvs.py         # GPU, resumable

Watch progress in the Modal dashboard, or fetch results later:
  modal volume get dvs128-data /checkpoints/metrics.csv results/data/step3_metrics.csv
"""

import modal

app = modal.App("dvs128-snn")

vol = modal.Volume.from_name("dvs128-data", create_if_missing=True)
DATA = "/data"
ROOT = f"{DATA}/DVS128Gesture"      # must contain download/DvsGesture.tar.gz etc.
CKPT = f"{DATA}/checkpoints"

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
def train_remote(epochs: int = 64, batch: int = 16, lr: float = 1e-3, T: int = 16):
    import os

    import torch
    import torch.nn.functional as F
    from spikingjelly.activation_based import functional
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
    from torch.utils.data import DataLoader

    device = "cuda"
    torch.manual_seed(0)

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
    os.makedirs(CKPT, exist_ok=True)
    last, best_path = f"{CKPT}/last.pt", f"{CKPT}/best.pt"
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

    metrics_path = f"{CKPT}/metrics.csv"
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w") as f:
            f.write("epoch,train_loss,train_acc,test_loss,test_acc\n")

    for epoch in range(start_epoch, epochs):
        tr_loss, tr_acc = run_epoch(train_loader, True)
        te_loss, te_acc = run_epoch(test_loader, False)
        sched.step()
        best_acc = max(best_acc, te_acc)
        print(f"epoch {epoch:3d}  train {tr_acc:.4f}  test {te_acc:.4f}"
              f"  best {best_acc:.4f}")

        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{tr_loss:.4f},{tr_acc:.4f},"
                    f"{te_loss:.4f},{te_acc:.4f}\n")
        state = {"net": net.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "epoch": epoch,
                 "best_acc": best_acc}
        torch.save(state, last)
        if te_acc >= best_acc:
            torch.save(state, best_path)
        vol.commit()                             # persist every epoch

    print(f"done. best test acc {best_acc:.4f}")


@app.local_entrypoint()
def main(mode: str = "train", epochs: int = 64):
    if mode == "prepare":
        prepare.remote()
    else:
        train_remote.remote(epochs=epochs)
