"""Step 6b, part 1 (GPU): temporal-correlation sweep of the noise corruption.

Why. Step 6 found the headline crossover: under Poisson background
noise the SNN keeps its accuracy and its confidence tracks it, while the
ANN collapses and stays confident. The report offers a hypothesis: a LIF
neuron is a leaky temporal integrator, uncorrelated noise pushes the
membrane briefly and leaks away, correlated signal accumulates. Step 6
cannot test that, because its noise is drawn independently per frame,
which is the one case the hypothesis favours.

Design. Keep the noise MARGINAL per frame identical (Poisson with rate
lam per pixel-bin-polarity, same lam grid as step 6) and vary only how
long a noise pattern persists along T:
  k = 1    a fresh Poisson field every frame (step 6's noise, background
           activity)
  k = 2, 4, 8   one field held for k consecutive frames
  k = 16   one field held for all 16 frames (a static hot-pixel pattern)
Every k adds the same expected number of events per frame. Only the
temporal correlation changes.

Prediction if the leak hypothesis is right: the SNN's advantage should
shrink as k grows, because persistent noise integrates like signal. The
ANN's frame-average also loses its averaging benefit as k grows, so the
quantity to read is the PAIRED SNN - ANN difference against k, not
either curve alone. If the difference is flat in k, the hypothesis is
wrong and something else (thresholding, rate coding) explains step 6.

Conditions: 4 lam x 5 k = 20, plus clean. 21 test passes per model, a
few minutes on the A10G. Seeded with fixed offsets, both models see
byte-identical inputs, and the seeds do not depend on the training seed.

Checkpoints: last.pt by default. Layout as in step 5.

Output: /eval/seed{S}/noise_sweep_outputs.npz with keys
  "{model}_lam{i}_k{k}"  i in 1..4, k in {1,2,4,8,16}    [N, 11] outputs
  "{model}_clean"                                          [N, 11]
  labels, lams, ks, seed, ckpt, snn_epoch, ann_epoch

Run:
  modal run scripts/step6b_noise_sweep.py --seed 0
Fetch:
  modal volume get dvs128-data /eval/seed0/noise_sweep_outputs.npz results/data/seed0/noise_sweep_outputs.npz
Analyse:
  python scripts/step6b_analysis.py results/data/seed0/noise_sweep_outputs.npz
"""

import modal

app = modal.App("dvs128-noise-sweep")

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

LAMS = [0.05, 0.1, 0.2, 0.5]      # same grid as step 6 "noise"
KS = [1, 2, 4, 8, 16]             # frames a noise field persists; 16 = static


def ckpt_dir(base: str, seed: int) -> str:
    return base if seed == 0 else f"{base}/seed{seed}"


def eval_dir(seed: int) -> str:
    return f"{DATA}/eval/seed{seed}"


def build_snn():
    import torch.nn as nn
    from spikingjelly.activation_based import layer, neuron, surrogate

    def lif():
        return neuron.LIFNode(surrogate_function=surrogate.ATan(),
                              detach_reset=True)

    def block(cin, cout):
        return [layer.Conv2d(cin, cout, 3, padding=1, bias=False),
                layer.BatchNorm2d(cout), lif(), layer.MaxPool2d(2, 2)]

    return nn.Sequential(
        *block(2, 128), *block(128, 128), *block(128, 128),
        *block(128, 128), *block(128, 128),
        layer.Flatten(), layer.Dropout(0.5),
        layer.Linear(128 * 4 * 4, 512), lif(),
        layer.Dropout(0.5), layer.Linear(512, 110), lif(),
        layer.VotingLayer(10),
    )


def build_ann():
    import torch.nn as nn

    def block(cin, cout):
        return [nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2, 2)]

    return nn.Sequential(
        *block(2, 128), *block(128, 128), *block(128, 128),
        *block(128, 128), *block(128, 128),
        nn.Flatten(), nn.Dropout(0.5),
        nn.Linear(128 * 4 * 4, 512), nn.ReLU(),
        nn.Dropout(0.5), nn.Linear(512, 110), nn.ReLU(),
    )


def block_noise(frame, lam, k, gen):
    """Add Poisson(lam) events whose pattern is held for k frames.

    frame: [B, T, 2, H, W] float counts on GPU. T must be divisible by k.
    k = 1 reproduces step 6's per-frame noise in distribution; k = T is
    one static field added to every frame.
    """
    import torch

    B, T, C, H, W = frame.shape
    assert T % k == 0, (T, k)
    field = torch.poisson(torch.full((B, T // k, C, H, W), float(lam),
                                     device=frame.device), generator=gen)
    return frame + field.repeat_interleave(k, dim=1)


@app.function(image=image, gpu="A10G", volumes={DATA: vol}, timeout=3600)
def run(T: int = 16, batch: int = 16, seed: int = 0, ckpt: str = "last"):
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

    snn = build_snn().to(device)
    functional.set_step_mode(snn, "m")
    ck = torch.load(f"{ckpt_dir(CKPT_SNN, seed)}/{ckpt}.pt", map_location=device)
    snn.load_state_dict(ck["net"])
    snn_epoch = int(ck["epoch"])
    snn.eval()
    ann = build_ann().to(device)
    ck = torch.load(f"{ckpt_dir(CKPT_ANN, seed)}/{ckpt}.pt", map_location=device)
    ann.load_state_dict(ck["net"])
    ann_epoch = int(ck["epoch"])
    ann.eval()
    print(f"seed {seed}, {ckpt}.pt: SNN epoch {snn_epoch}, ANN epoch {ann_epoch}")

    def snn_fwd(f):
        out = snn(f.transpose(0, 1)).mean(0)
        functional.reset_net(snn)
        return out

    def ann_fwd(f):
        B, T_ = f.shape[0], f.shape[1]
        return ann(f.reshape(B * T_, *f.shape[2:])) \
            .view(B, T_, 11, 10).mean(3).mean(1)

    conditions = [("clean", None, None)]
    for i, lam in enumerate(LAMS, start=1):
        for k in KS:
            conditions.append((f"lam{i}_k{k}", lam, k))

    results, labels_all = {}, None
    with torch.no_grad():
        for tag, lam, k in conditions:
            gen = torch.Generator(device=device)
            gen.manual_seed(3000 if lam is None else 3000 + LAMS.index(lam) * 31 + k)
            outs = {"snn": [], "ann": []}
            labels = []
            for frame, label in loader:
                frame = frame.to(device).float()
                if lam is not None:
                    frame = block_noise(frame, lam, k, gen)
                labels.append(label.numpy())
                outs["snn"].append(snn_fwd(frame).cpu().numpy())
                outs["ann"].append(ann_fwd(frame).cpu().numpy())
            labels_all = np.concatenate(labels)
            line = f"{tag:<12} lam={lam} k={k}"
            for m in ("snn", "ann"):
                results[f"{m}_{tag}"] = np.concatenate(outs[m])
                acc = (results[f"{m}_{tag}"].argmax(1) == labels_all).mean()
                line += f"  {m} acc={acc:.4f}"
            print(line)

    os.makedirs(eval_dir(seed), exist_ok=True)
    suffix = "" if ckpt == "last" else f"_{ckpt}"
    path = f"{eval_dir(seed)}/noise_sweep_outputs{suffix}.npz"
    np.savez(path, labels=labels_all, lams=np.array(LAMS), ks=np.array(KS),
             seed=seed, ckpt=ckpt, snn_epoch=snn_epoch, ann_epoch=ann_epoch,
             **results)
    vol.commit()
    print(f"wrote {path.replace(DATA, '')}")


@app.local_entrypoint()
def main(seed: int = 0, ckpt: str = "last"):
    run.remote(seed=seed, ckpt=ckpt)
