"""Step 6, part 1 (GPU): corruption stress test, dump outputs.

Runs the full test set through BOTH models under four event-camera
corruptions at four severities each, plus clean. 33 test passes total,
a few minutes on the A10G.

Corruptions, applied to the frame tensor [B, T, 2, 128, 128]:
  drop     binomial thinning: each event survives with prob 1-p.
           Models a failing / desensitized sensor. p in {.2 .4 .6 .8}
  noise    add Poisson background events everywhere, rate lam per
           pixel-bin-polarity, drawn independently per frame. Models
           background-activity noise. lam in {.05 .1 .2 .5}
           (step6b_noise_sweep.py varies the temporal correlation of
           this noise to test the leak hypothesis)
  occlude  zero a random square of side s (same square across T).
           Models partial blockage. s in {24 40 56 72}
  tshuffle permute the T axis within windows of size w. w in {2 4 8 16}.
           DESIGNED CONTROL: the ANN averages logits over T, so it is
           invariant to frame order by construction. Any degradation
           is therefore pure SNN temporal processing. If the SNN also
           stays flat, it never used timing either.

All corruptions are seeded with fixed per-corruption offsets, so both
models see identical corrupted inputs AND reruns are reproducible. The
corruption seeds do not depend on the training seed: every seed's models
see byte-identical corrupted test sets.
(An earlier version seeded with hash(kind), which Python randomizes
per process: results/data/archive/ keeps that run. Its within-run
SNN/ANN pairing was valid, but its exact numbers are not re-derivable.)

Checkpoints: last.pt by default (fixed-budget epoch 64), best.pt on
request for the appendix only. Layout as in step 5: seed 0 in
/checkpoints and /checkpoints_ann, seed S > 0 in the seedS subfolders.

Output: /eval/seed{S}/corruption_outputs.npz (or corruption_outputs_best.npz)
with key "{model}_{corr}_{sev_idx}" per condition, plus "labels",
severity metadata and provenance scalars (seed, ckpt, snn_epoch, ann_epoch).

Run:
  modal run scripts/step6_corruption.py --seed 0
  modal run scripts/step6_corruption.py --seed 1
Fetch:
  modal volume get dvs128-data /eval/seed0/corruption_outputs.npz results/data/seed0/corruption_outputs.npz

The original single-seed dump (/eval/corruption_outputs.npz, best.pt)
is left untouched. The live demo's published curves come from it.
"""

import modal

app = modal.App("dvs128-corrupt")

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
    return base if seed == 0 else f"{base}/seed{seed}"


def eval_dir(seed: int) -> str:
    return f"{DATA}/eval/seed{seed}"


def out_name(stem: str, ckpt: str) -> str:
    return f"{stem}.npz" if ckpt == "last" else f"{stem}_{ckpt}.npz"


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


SEVERITIES = {
    "drop": [0.2, 0.4, 0.6, 0.8],
    "noise": [0.05, 0.1, 0.2, 0.5],
    "occlude": [24, 40, 56, 72],
    "tshuffle": [2, 4, 8, 16],
}

# fixed seed offsets: hash(kind) is randomized per process, this is not
SEED_OFFSET = {"clean": 0, "drop": 1, "noise": 2, "occlude": 3, "tshuffle": 4}


def corrupt(frame, kind, sev, gen):
    """frame: [B, T, 2, 128, 128] float counts, on GPU. Seeded generator."""
    import torch

    if kind == "drop":
        return torch.binomial(frame, torch.full_like(frame, 1.0 - sev),
                              generator=gen)
    if kind == "noise":
        return frame + torch.poisson(torch.full_like(frame, float(sev)),
                                     generator=gen)
    if kind == "occlude":
        out = frame.clone()
        B, _, _, H, W = frame.shape
        s = int(sev)
        ys = torch.randint(0, H - s + 1, (B,), generator=gen, device=frame.device)
        xs = torch.randint(0, W - s + 1, (B,), generator=gen, device=frame.device)
        for b in range(B):
            out[b, :, :, ys[b]:ys[b] + s, xs[b]:xs[b] + s] = 0
        return out
    if kind == "tshuffle":
        out = frame.clone()
        T = frame.shape[1]
        w = int(sev)
        for start in range(0, T, w):
            end = min(start + w, T)
            perm = torch.randperm(end - start, generator=gen,
                                  device=frame.device) + start
            out[:, start:end] = frame[:, perm]
        return out
    raise ValueError(kind)


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

    conditions = [("clean", 0, None)]
    for kind, sevs in SEVERITIES.items():
        for i, s in enumerate(sevs, start=1):
            conditions.append((kind, i, s))

    results, labels_all = {}, []
    with torch.no_grad():
        for kind, sev_idx, sev in conditions:
            gen = torch.Generator(device=device)
            gen.manual_seed(1000 + sev_idx * 7 + SEED_OFFSET[kind])
            outs = {"snn": [], "ann": []}
            labels = []
            for frame, label in loader:
                frame = frame.to(device).float()
                if sev is not None:
                    frame = corrupt(frame, kind, sev, gen)
                labels.append(label.numpy())
                outs["snn"].append(snn_fwd(frame).cpu().numpy())
                outs["ann"].append(ann_fwd(frame).cpu().numpy())
            labels = np.concatenate(labels)
            labels_all = labels
            for m in ("snn", "ann"):
                key = f"{m}_{kind}_{sev_idx}"
                results[key] = np.concatenate(outs[m])
                acc = (results[key].argmax(1) == labels).mean()
                print(f"{key:<22} sev={sev}  acc={acc:.4f}")

    os.makedirs(eval_dir(seed), exist_ok=True)
    path = f"{eval_dir(seed)}/{out_name('corruption_outputs', ckpt)}"
    np.savez(path,
             labels=labels_all,
             severities=np.array(
                 [f"{k}:{','.join(str(s) for s in v)}"
                  for k, v in SEVERITIES.items()]),
             seed=seed, ckpt=ckpt, snn_epoch=snn_epoch, ann_epoch=ann_epoch,
             **results)
    vol.commit()
    print(f"wrote {path.replace(DATA, '')}")


@app.local_entrypoint()
def main(seed: int = 0, ckpt: str = "last"):
    run.remote(seed=seed, ckpt=ckpt)
