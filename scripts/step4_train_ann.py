"""Step 4: the ANN twin of the step 3 SNN, on Modal.

One variable changes: every LIF neuron becomes a ReLU. Everything
else is held identical to step 3: same conv stack, same parameter
count, same T=16 frames from the same volume, same loss, optimizer,
schedule, batch size, seed, and epochs.

Design choice, stated for the writeup: the ANN sees the same 16
frames and its logits are averaged over T, mirroring the SNN's rate
readout. The only thing removed is the membrane state carrying
information across timesteps. (The FOI paper instead collapsed all
events into one frame; our choice isolates the neuron model more
cleanly since the input pipeline stays byte-identical.)

Checkpoints go to a separate folder (checkpoints_ann) with the same
save-every-epoch and auto-resume behavior as step 3.

Run (frames already prepared by step 3, so no prepare pass):
  modal run --detach scripts/step4_train_ann.py            # seed 0
  modal run --detach scripts/step4_train_ann.py --seed 1   # any other seed

Seeds and checkpoints mirror step 3 exactly: seed 0 stays in
/checkpoints_ann/, seed S > 0 goes to /checkpoints_ann/seedS/. best.pt is
test-selected and kept for reference only; steps 5-7 evaluate last.pt.

Fetch results later:
  modal volume get dvs128-data /checkpoints_ann/metrics.csv results/data/seed0/step4_metrics.csv
  modal volume get dvs128-data /checkpoints_ann/seed1/metrics.csv results/data/seed1/step4_metrics.csv
"""

import modal

app = modal.App("dvs128-ann")

vol = modal.Volume.from_name("dvs128-data", create_if_missing=True)
DATA = "/data"
ROOT = f"{DATA}/DVS128Gesture"
CKPT = f"{DATA}/checkpoints_ann"          # separate from the SNN's folder


def ckpt_dir(seed: int) -> str:
    """seed 0 keeps the original folder; other seeds get a subfolder."""
    return CKPT if seed == 0 else f"{CKPT}/seed{seed}"


image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.0",
    "torchvision==0.19.0",
    "spikingjelly==0.0.0.0.14",           # only used for the dataset class
    "numpy<2",
    "tqdm",
)


def build_net():
    """Same stack as step 3's build_net, LIF swapped for ReLU.
    Parameter count is identical: conv, BN, and linear layers unchanged."""
    import torch.nn as nn

    def block(cin, cout):
        return [
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        ]

    return nn.Sequential(
        *block(2, 128),
        *block(128, 128),
        *block(128, 128),
        *block(128, 128),
        *block(128, 128),
        nn.Flatten(),
        nn.Dropout(0.5),
        nn.Linear(128 * 4 * 4, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 110),
        nn.ReLU(),
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
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/run.json", "w") as f:
        json.dump({"model": "ann", "seed": seed, "epochs": epochs,
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

    def forward(frame):
        # frame: [B, T, 2, 128, 128]. Fold T into the batch, run the
        # stateless net, unfold, vote (110 -> 11), average over T.
        B, T_ = frame.shape[0], frame.shape[1]
        x = frame.reshape(B * T_, *frame.shape[2:])
        out = net(x)                          # [B*T, 110]
        out = out.view(B, T_, 11, 10).mean(3) # voting layer, same as SNN
        return out.mean(1)                    # average over T = rate readout

    def run_epoch(loader, training):
        net.train(training)
        total, correct, loss_sum = 0, 0, 0.0
        with torch.set_grad_enabled(training):
            for frame, label in loader:
                frame = frame.to(device).float()
                label = label.to(device)
                out_fr = forward(frame)
                loss = F.mse_loss(out_fr, F.one_hot(label, 11).float())
                if training:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
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
        vol.commit()

    print(f"done. seed {seed}: last-epoch test acc {te_acc:.4f} (report this), "
          f"best test acc {best_acc:.4f} (test-selected, do not report)")


@app.local_entrypoint()
def main(epochs: int = 64, seed: int = 0):
    train_remote.remote(epochs=epochs, seed=seed)
