"""Step 11: the matched pair on CIFAR10-DVS, the second dataset. One Modal app.

Same recipe as DVS128Gesture, applied to both models: SpikingJelly's canonical
conv net, LIF vs ReLU as the only difference, T=16 frames split_by="number",
MSE against one-hot, Adam 1e-3, cosine schedule, batch 16, 64 epochs, dropout
0.5, one shared seed per pair. Only the output width changes: 100 spiking
outputs voting for 10 classes instead of 110 for 11.

Why this dataset. SpikingJelly loads it, the frames are 128x128 like
DVS128Gesture so every corruption transfers unchanged, the test split has
1000 samples (tighter CIs than 288), and its temporal structure is artificial
(static images moved in front of the sensor), so the temporal-shuffle control
should read near zero for the SNN by construction. A clean contrast.

Split. CIFAR10-DVS ships no train/test split. This script uses the same
deterministic per-class split SpikingJelly's split_to_train_test_set gives
with random_split=False (first 90% of each class in sorted file order for
training, the rest for test), computed from file names so nothing is loaded.
9000 train, 1000 test. Identical for both models and every seed.

Modes (all from the repo root, always python -m modal):
  python -m modal run --detach scripts/step11_cifar10dvs.py --mode prepare
      CPU. Downloads the ten figshare zips (~1 GB), extracts, converts to
      frames. Spawns and returns; takes an hour or two. Watch the dashboard.
      Resumable: each phase (download, extract, events_np, frames) leaves a
      done-marker under cifar10dvs/markers/ after an integrity count and a
      volume commit. Rerunning skips finished phases and rebuilds any phase
      that was cut off. Expect 10 classes x 1000 files at every stage.
  python -m modal run --detach scripts/step11_cifar10dvs.py --mode train --model snn --seed 0
  python -m modal run --detach scripts/step11_cifar10dvs.py --mode train --model ann --seed 0
      GPU, spawns and returns. About 7 min/epoch for the SNN (~8 h per run)
      and ~2 min/epoch for the ANN, from the DVS128 timings scaled by 7.7x
      the samples. Resumable at epoch granularity: last.pt is written
      atomically every epoch with optimizer, scheduler and RNG states, then
      committed. Modal retries a crashed or preempted container up to three
      times and the retry resumes from last.pt; a run that still dies resumes
      when relaunched with the same --model and --seed. A heartbeat lock
      (lock.json, stale after 45 min) refuses a second container on the same
      run; add --force to override after a crash you are sure about.
  python -m modal run scripts/step11_cifar10dvs.py --mode dump --seed 0
  python -m modal run scripts/step11_cifar10dvs.py --mode corrupt --seed 0
  python -m modal run scripts/step11_cifar10dvs.py --mode sweep --seed 0
  python -m modal run scripts/step11_cifar10dvs.py --mode energy --seed 0
      GPU, minutes each, blocking (leave the window open). last.pt by default.

Volume layout (dvs128-data):
  cifar10dvs/                 download/, extract/, events_np/, frames_number_16_split_by_number/
  checkpoints_c10/{snn,ann}/seedS/   last.pt, best.pt, metrics.csv, run.json
  eval_c10/seedS/             test_outputs.npz, corruption_outputs.npz,
                              noise_sweep_outputs.npz, energy.json, energy_report.txt
Fetch into results/data/c10/seedS/ with the same file names, plus
step3_metrics.csv (SNN) and step4_metrics.csv (ANN) so step10_multiseed.py
can read them with --root results/data/c10 --tag _c10 --name CIFAR10-DVS.

Corruptions, sweep and energy accounting are copied from steps 6, 6b and 7
with the class count changed. Corruption seeds are the same fixed offsets,
so both models and all seeds see byte-identical corrupted inputs.
"""

import modal

app = modal.App("cifar10dvs-pair")

vol = modal.Volume.from_name("dvs128-data", create_if_missing=True)
DATA = "/data"
ROOT = f"{DATA}/cifar10dvs"
CKPT = f"{DATA}/checkpoints_c10"
EVAL = f"{DATA}/eval_c10"
N_CLASSES = 10
VOTES = 10
TRAIN_RATIO = 0.9

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.0",
    "torchvision==0.19.0",
    "spikingjelly==0.0.0.0.14",
    "numpy<2",
    "tqdm",
)

E_MAC, E_AC = 4.6e-12, 0.9e-12   # joules, Horowitz 2014, 45 nm
LAYER_MACS = [
    ("conv1", 2 * 128 * 9 * 128 * 128),
    ("conv2", 128 * 128 * 9 * 64 * 64),
    ("conv3", 128 * 128 * 9 * 32 * 32),
    ("conv4", 128 * 128 * 9 * 16 * 16),
    ("conv5", 128 * 128 * 9 * 8 * 8),
    ("fc1",   2048 * 512),
    ("fc2",   512 * N_CLASSES * VOTES),
]

SEVERITIES = {
    "drop": [0.2, 0.4, 0.6, 0.8],
    "noise": [0.05, 0.1, 0.2, 0.5],
    "occlude": [24, 40, 56, 72],
    "tshuffle": [2, 4, 8, 16],
}
SEED_OFFSET = {"clean": 0, "drop": 1, "noise": 2, "occlude": 3, "tshuffle": 4}
LAMS = [0.05, 0.1, 0.2, 0.5]
KS = [1, 2, 4, 8, 16]


def run_dir(model: str, seed: int) -> str:
    return f"{CKPT}/{model}/seed{seed}"


def eval_dir(seed: int) -> str:
    return f"{EVAL}/seed{seed}"


def suffix(ckpt: str) -> str:
    return "" if ckpt == "last" else f"_{ckpt}"


# ---------------------------------------------------------------- models

def build_net(model: str):
    """SNN: SpikingJelly canonical net with 100 spiking outputs voting for 10.
    ANN: the same stack with every LIF replaced by ReLU; voting in forward."""
    import torch.nn as nn

    if model == "snn":
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
            layer.Dropout(0.5), layer.Linear(512, N_CLASSES * VOTES), lif(),
            layer.VotingLayer(VOTES),
        )
    if model == "ann":
        def block(cin, cout):
            return [nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                    nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2, 2)]

        return nn.Sequential(
            *block(2, 128), *block(128, 128), *block(128, 128),
            *block(128, 128), *block(128, 128),
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(512, N_CLASSES * VOTES), nn.ReLU(),
        )
    raise ValueError(model)


def make_forward(net, model):
    """Returns f(frame [B, T, 2, H, W]) -> [B, N_CLASSES], rate readout for both."""
    if model == "snn":
        from spikingjelly.activation_based import functional
        functional.set_step_mode(net, "m")

        def fwd(frame):
            out = net(frame.transpose(0, 1)).mean(0)
            functional.reset_net(net)
            return out
        return fwd

    def fwd(frame):
        B, T_ = frame.shape[0], frame.shape[1]
        out = net(frame.reshape(B * T_, *frame.shape[2:]))
        return out.view(B, T_, N_CLASSES, VOTES).mean(3).mean(1)
    return fwd


def load_model(model: str, seed: int, ckpt: str, device: str):
    import torch
    net = build_net(model).to(device)
    ck = torch.load(f"{run_dir(model, seed)}/{ckpt}.pt", map_location=device)
    net.load_state_dict(ck["net"])
    net.eval()
    return net, make_forward(net, model), int(ck["epoch"])


# ---------------------------------------------------------------- data

def load_dataset(T: int):
    from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
    return CIFAR10DVS(ROOT, data_type="frame", frames_number=T, split_by="number")


def split_indices(ds, train_ratio: float = TRAIN_RATIO):
    """Per-class deterministic split from file names: first ceil(ratio * n)
    of each class in sorted order to train, the rest to test. These are the
    indices SpikingJelly's split_to_train_test_set(random_split=False)
    returns, without loading a single frame."""
    import math
    from collections import defaultdict

    by_class = defaultdict(list)
    for i, (_, y) in enumerate(ds.samples):
        by_class[int(y)].append(i)
    train_idx, test_idx = [], []
    for y in sorted(by_class):
        idx = by_class[y]
        pos = math.ceil(len(idx) * train_ratio)
        train_idx += idx[:pos]
        test_idx += idx[pos:]
    return train_idx, test_idx


def make_loaders(T: int, batch: int, workers: int = 8):
    from torch.utils.data import DataLoader, Subset
    ds = load_dataset(T)
    tr, te = split_indices(ds)
    train_loader = DataLoader(Subset(ds, tr), batch_size=batch, shuffle=True,
                              num_workers=workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(Subset(ds, te), batch_size=batch, shuffle=False,
                             num_workers=workers, pin_memory=True)
    return train_loader, test_loader, len(tr), len(te)


def logged_test_acc(rd: str, epoch: int):
    import csv
    import os
    path = f"{rd}/metrics.csv"
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["epoch"]) == epoch:
                return float(row["test_acc"])
    return None


# ---------------------------------------------------------------- prepare

N_FILES = 10000          # 10 classes x 1000 recordings
N_PER_CLASS = 1000


def count_files(root: str, ext: str) -> dict:
    """{class_dir: number of files ending in ext}, one level down."""
    import os
    out = {}
    if not os.path.isdir(root):
        return out
    for c in sorted(os.listdir(root)):
        d = os.path.join(root, c)
        if os.path.isdir(d):
            out[c] = sum(1 for f in os.listdir(d) if f.endswith(ext))
    return out


def phase_ok(counts: dict) -> bool:
    return len(counts) == 10 and all(v == N_PER_CLASS for v in counts.values())


@app.function(image=image, volumes={DATA: vol}, timeout=8 * 3600, cpu=16, memory=16384,
              retries=modal.Retries(max_retries=2, initial_delay=60.0, backoff_coefficient=1.0))
def prepare(T: int = 16):
    """Download, extract, convert to events, convert to T frames. One phase at
    a time, each verified by file counts, marked done, and committed, so a
    cut-off run resumes at the phase it was in and never reuses a partial
    directory. SpikingJelly's own constructor would."""
    import os
    import shutil
    import time

    import numpy as np
    from spikingjelly import datasets as sjds
    from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
    from torchvision.datasets import utils

    # SpikingJelly 0.0.0.0.14 reads CIFAR10-DVS with np.bool, removed in numpy
    # 1.24; the image pins numpy<2. Without this every conversion fails silently.
    if not hasattr(np, "bool"):
        np.bool = np.bool_

    os.makedirs(ROOT, exist_ok=True)
    markers = f"{ROOT}/markers"
    os.makedirs(markers, exist_ok=True)
    download_root = f"{ROOT}/download"
    extract_root = f"{ROOT}/extract"
    events_root = f"{ROOT}/events_np"
    frames_root = f"{ROOT}/frames_number_{T}_split_by_number"

    def done(name):
        return os.path.exists(f"{markers}/{name}.done")

    def mark(name):
        with open(f"{markers}/{name}.done", "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        vol.commit()
        print(f"phase {name} done and committed")

    def fresh(path):
        if os.path.exists(path):
            print(f"removing partial [{path}]")
            shutil.rmtree(path)
        os.makedirs(path)

    # phase 1: download, per file, md5-checked (idempotent by construction)
    if not done("download"):
        os.makedirs(download_root, exist_ok=True)
        for file_name, url, md5 in CIFAR10DVS.resource_url_md5():
            fpath = os.path.join(download_root, file_name)
            if utils.check_integrity(fpath, md5):
                print(f"have [{file_name}]")
                continue
            if os.path.exists(fpath):
                os.remove(fpath)
            print(f"downloading [{file_name}]")
            utils.download_url(url=url, root=download_root, filename=file_name, md5=md5)
            vol.commit()
        assert all(utils.check_integrity(os.path.join(download_root, f), m)
                   for f, _, m in CIFAR10DVS.resource_url_md5()), "download incomplete"
        mark("download")

    # phase 2: extract the ten zips
    if not done("extract"):
        fresh(extract_root)
        CIFAR10DVS.extract_downloaded_files(download_root, extract_root)
        counts = count_files(extract_root, ".aedat")
        print("extract counts:", counts)
        assert phase_ok(counts), f"extract incomplete: {counts}"
        mark("extract")

    # phase 3: aedat -> events npz. Probe one file through both conversions
    # synchronously first: the bulk phases run in thread pools that swallow
    # exceptions, so this is where a real error becomes a visible traceback.
    if not done("events"):
        probe = f"{markers}/probe"
        fresh(probe)
        os.makedirs(f"{probe}/events")
        os.makedirs(f"{probe}/frames")
        first_class = sorted(os.listdir(extract_root))[0]
        first_file = sorted(os.listdir(os.path.join(extract_root, first_class)))[0]
        CIFAR10DVS.read_aedat_save_to_np(os.path.join(extract_root, first_class, first_file),
                                         f"{probe}/events/probe.npz")
        ev = np.load(f"{probe}/events/probe.npz")
        print(f"probe events ok: {first_file} -> {len(ev['t'])} events, "
              f"x in [{ev['x'].min()}, {ev['x'].max()}], y in [{ev['y'].min()}, {ev['y'].max()}]")
        sjds.integrate_events_file_to_frames_file_by_fixed_frames_number(
            CIFAR10DVS.load_events_np, f"{probe}/events/probe.npz", f"{probe}/frames",
            "number", T, 128, 128, True)
        fr = np.load(f"{probe}/frames/probe.npz")["frames"]
        assert tuple(fr.shape) == (T, 2, 128, 128), fr.shape
        print(f"probe frames ok: shape {tuple(fr.shape)}, mean count per cell {fr.mean():.4f}")
        shutil.rmtree(probe)
        fresh(events_root)
        CIFAR10DVS.create_events_np_files(extract_root, events_root)
        counts = count_files(events_root, ".npz")
        print("events_np counts:", counts)
        assert phase_ok(counts), f"events conversion incomplete: {counts}"
        mark("events")

    # phase 4: events -> T frames. The constructor finds download/extract/
    # events_np present and only builds the frame cache.
    if not done("frames"):
        if os.path.exists(frames_root):
            print(f"removing partial [{frames_root}]")
            shutil.rmtree(frames_root)
        load_dataset(T)
        counts = count_files(frames_root, ".npz")
        print("frames counts:", counts)
        assert phase_ok(counts), f"frame conversion incomplete: {counts}"
        mark("frames")

    # final check, the numbers training will see
    ds = load_dataset(T)
    tr, te = split_indices(ds)
    frame, label = ds[0]
    assert len(ds) == N_FILES and len(tr) == 9000 and len(te) == 1000, (len(ds), len(tr), len(te))
    assert tuple(frame.shape) == (T, 2, 128, 128), frame.shape
    print(f"CIFAR10-DVS ready: {len(ds)} samples, split {len(tr)} train / {len(te)} test, "
          f"sample 0 shape {tuple(frame.shape)} {frame.dtype}, label {label}")
    vol.commit()
    print("prepare done")


# ---------------------------------------------------------------- train

LOCK_STALE_S = 45 * 60


@app.function(image=image, gpu="A10G", volumes={DATA: vol}, timeout=24 * 3600, cpu=8,
              retries=modal.Retries(max_retries=3, initial_delay=60.0, backoff_coefficient=1.0))
def train(model: str, seed: int, epochs: int = 64, batch: int = 16, lr: float = 1e-3,
          T: int = 16, force: bool = False):
    import json
    import os
    import random
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F

    assert model in ("snn", "ann"), model
    device = "cuda"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rd = run_dir(model, seed)

    train_loader, test_loader, n_tr, n_te = make_loaders(T, batch)
    print(f"{model} seed {seed}: {n_tr} train / {n_te} test samples")

    net = build_net(model).to(device)
    fwd = make_forward(net, model)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    os.makedirs(rd, exist_ok=True)

    # one container per run: a fresh heartbeat means someone else is training this
    lock = f"{rd}/lock.json"
    if os.path.exists(lock) and not force:
        age = time.time() - os.path.getmtime(lock)
        if age < LOCK_STALE_S:
            raise RuntimeError(
                f"{lock} was touched {age / 60:.0f} min ago, another container may be "
                f"training {model} seed {seed}. Wait for it, or relaunch with --force.")

    def touch_lock():
        with open(lock, "w") as f:
            json.dump({"model": model, "seed": seed, "time": time.time()}, f)

    def atomic_save(obj, path):
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    touch_lock()
    with open(f"{rd}/run.json", "w") as f:
        json.dump({"dataset": "cifar10dvs", "model": model, "seed": seed,
                   "epochs": epochs, "batch": batch, "lr": lr, "T": T,
                   "loss": "mse_onehot", "optimizer": "adam", "schedule": "cosine",
                   "split": f"per-class first {TRAIN_RATIO:.0%} train, sorted order"},
                  f, indent=1)
    last, best_path = f"{rd}/last.pt", f"{rd}/best.pt"
    start_epoch, best_acc = 0, 0.0
    if os.path.exists(last):
        ck = torch.load(last, map_location=device)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best_acc = ck["epoch"] + 1, ck["best_acc"]
        if "rng" in ck:                       # continue the shuffle and dropout streams
            torch.set_rng_state(ck["rng"]["torch"].cpu())
            torch.cuda.set_rng_state(ck["rng"]["cuda"].cpu())
            np.random.set_state(ck["rng"]["numpy"])
            random.setstate(ck["rng"]["random"])
        print(f"resumed from epoch {ck['epoch']}, best acc {best_acc:.4f}")

    def run_epoch(loader, training):
        net.train(training)
        total, correct, loss_sum = 0, 0, 0.0
        with torch.set_grad_enabled(training):
            for frame, label in loader:
                frame = frame.to(device).float()
                label = label.to(device)
                out = fwd(frame)
                loss = F.mse_loss(out, F.one_hot(label, N_CLASSES).float())
                if training:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                total += label.numel()
                correct += (out.argmax(1) == label).sum().item()
                loss_sum += loss.item() * label.numel()
        return loss_sum / total, correct / total

    metrics_path = f"{rd}/metrics.csv"
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w") as f:
            f.write("epoch,train_loss,train_acc,test_loss,test_acc,epoch_time_s\n")

    te_acc = float("nan")
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(train_loader, True)
        te_loss, te_acc = run_epoch(test_loader, False)
        sched.step()
        epoch_time = time.time() - t0
        best_acc = max(best_acc, te_acc)
        print(f"epoch {epoch:3d}  train {tr_acc:.4f}  test {te_acc:.4f}"
              f"  best {best_acc:.4f}  {epoch_time:.0f}s")
        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{tr_loss:.4f},{tr_acc:.4f},"
                    f"{te_loss:.4f},{te_acc:.4f},{epoch_time:.1f}\n")
        state = {"net": net.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "epoch": epoch, "best_acc": best_acc,
                 "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
                         "numpy": np.random.get_state(), "random": random.getstate()}}
        atomic_save(state, last)
        if te_acc >= best_acc:             # test-selected: kept for reference only
            atomic_save(state, best_path)
        touch_lock()
        vol.commit()

    if os.path.exists(lock):
        os.remove(lock)
    vol.commit()
    print(f"done. {model} seed {seed}: last-epoch test acc {te_acc:.4f} (report this), "
          f"best {best_acc:.4f} (test-selected, do not report)")


# ---------------------------------------------------------------- corruptions

def corrupt(frame, kind, sev, gen):
    import torch

    if kind == "drop":
        return torch.binomial(frame, torch.full_like(frame, 1.0 - sev), generator=gen)
    if kind == "noise":
        return frame + torch.poisson(torch.full_like(frame, float(sev)), generator=gen)
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
            perm = torch.randperm(end - start, generator=gen, device=frame.device) + start
            out[:, start:end] = frame[:, perm]
        return out
    raise ValueError(kind)


def block_noise(frame, lam, k, gen):
    import torch
    B, T, C, H, W = frame.shape
    assert T % k == 0, (T, k)
    field = torch.poisson(torch.full((B, T // k, C, H, W), float(lam),
                                     device=frame.device), generator=gen)
    return frame + field.repeat_interleave(k, dim=1)


class Density:
    def __init__(self):
        self.value_sum, self.nonzero_sum, self.count = 0.0, 0.0, 0

    def hook(self, module, inp, out):
        self.value_sum += out.sum().item()
        self.nonzero_sum += (out != 0).sum().item()
        self.count += out.numel()

    @property
    def mean(self):
        return self.value_sum / self.count

    @property
    def nonzero(self):
        return self.nonzero_sum / self.count


# ---------------------------------------------------------------- evaluation

@app.function(image=image, gpu="A10G", volumes={DATA: vol}, timeout=2 * 3600, cpu=8)
def evaluate(mode: str, seed: int, ckpt: str = "last", T: int = 16, batch: int = 16):
    import json
    import os

    import numpy as np
    import torch

    assert mode in ("dump", "corrupt", "sweep", "energy"), mode
    assert ckpt in ("last", "best"), ckpt
    device = "cuda"
    _, loader, _, n_te = make_loaders(T, batch)
    snn, snn_fwd, snn_epoch = load_model("snn", seed, ckpt, device)
    ann, ann_fwd, ann_epoch = load_model("ann", seed, ckpt, device)
    fwds = {"snn": snn_fwd, "ann": ann_fwd}
    print(f"CIFAR10-DVS seed {seed}, {ckpt}.pt: SNN epoch {snn_epoch}, ANN epoch {ann_epoch}, "
          f"n = {n_te}")
    os.makedirs(eval_dir(seed), exist_ok=True)
    meta = dict(seed=seed, ckpt=ckpt, snn_epoch=snn_epoch, ann_epoch=ann_epoch,
                dataset="cifar10dvs")

    def run_conditions(conditions, seed_fn, transform):
        """conditions: list of (tag, params). Returns (labels, {f"{model}_{tag}": out})."""
        results, labels_all = {}, None
        with torch.no_grad():
            for tag, params in conditions:
                gen = torch.Generator(device=device)
                gen.manual_seed(seed_fn(tag, params))
                outs = {"snn": [], "ann": []}
                labels = []
                for frame, label in loader:
                    frame = frame.to(device).float()
                    if params is not None:
                        frame = transform(frame, params, gen)
                    labels.append(label.numpy())
                    for m in ("snn", "ann"):
                        outs[m].append(fwds[m](frame).cpu().numpy())
                labels_all = np.concatenate(labels)
                line = f"{tag:<16}"
                for m in ("snn", "ann"):
                    results[f"{m}_{tag}"] = np.concatenate(outs[m])
                    acc = (results[f"{m}_{tag}"].argmax(1) == labels_all).mean()
                    line += f"  {m} acc={acc:.4f}"
                print(line)
        return labels_all, results

    if mode == "dump":
        labels, res = run_conditions([("clean", None)], lambda t, p: 0, None)
        for m, epoch in (("snn", snn_epoch), ("ann", ann_epoch)):
            acc = (res[f"{m}_clean"].argmax(1) == labels).mean()
            logged = logged_test_acc(run_dir(m, seed), epoch)
            flag = ""
            if logged is not None and abs(acc - logged) > 1.0 / len(labels) + 1e-6:
                flag = "   WARNING: differs from metrics.csv by more than one sample"
            print(f"{m.upper()} test acc {acc:.4f} (epoch {epoch}, metrics.csv logged "
                  f"{'unknown' if logged is None else f'{logged:.4f}'}){flag}")
        path = f"{eval_dir(seed)}/test_outputs{suffix(ckpt)}.npz"
        np.savez(path, labels=labels, snn_out=res["snn_clean"], ann_out=res["ann_clean"], **meta)

    elif mode == "corrupt":
        conditions = [("clean_0", None)]
        for kind, sevs in SEVERITIES.items():
            for i, s in enumerate(sevs, start=1):
                conditions.append((f"{kind}_{i}", (kind, i, s)))

        def seed_fn(tag, p):
            return 1000 if p is None else 1000 + p[1] * 7 + SEED_OFFSET[p[0]]

        labels, res = run_conditions(conditions, seed_fn,
                                     lambda f, p, g: corrupt(f, p[0], p[2], g))
        path = f"{eval_dir(seed)}/corruption_outputs{suffix(ckpt)}.npz"
        np.savez(path, labels=labels,
                 severities=np.array([f"{k}:{','.join(str(s) for s in v)}"
                                      for k, v in SEVERITIES.items()]),
                 **meta, **res)

    elif mode == "sweep":
        conditions = [("clean", None)]
        for i, lam in enumerate(LAMS, start=1):
            for k in KS:
                conditions.append((f"lam{i}_k{k}", (lam, k)))

        def seed_fn(tag, p):
            return 3000 if p is None else 3000 + LAMS.index(p[0]) * 31 + p[1]

        labels, res = run_conditions(conditions, seed_fn,
                                     lambda f, p, g: block_noise(f, p[0], p[1], g))
        path = f"{eval_dir(seed)}/noise_sweep_outputs{suffix(ckpt)}.npz"
        np.savez(path, labels=labels, lams=np.array(LAMS), ks=np.array(KS), **meta, **res)

    else:  # energy
        import torch.nn as nn
        from spikingjelly.activation_based import layer, neuron

        lifs = [m for m in snn.modules() if isinstance(m, neuron.LIFNode)]
        snn_pools = [m for m in snn.modules() if isinstance(m, layer.MaxPool2d)]
        relus = [m for m in ann.modules() if isinstance(m, nn.ReLU)]
        ann_pools = [m for m in ann.modules() if isinstance(m, nn.MaxPool2d)]
        assert len(lifs) == 7 and len(relus) == 7 and len(snn_pools) == 5 and len(ann_pools) == 5
        lif_d, snn_pool_d = [Density() for _ in lifs], [Density() for _ in snn_pools]
        relu_d, ann_pool_d = [Density() for _ in relus], [Density() for _ in ann_pools]
        for ms, ds_ in ((lifs, lif_d), (snn_pools, snn_pool_d), (relus, relu_d),
                        (ann_pools, ann_pool_d)):
            for m, d in zip(ms, ds_):
                m.register_forward_hook(d.hook)
        input_d = Density()
        with torch.no_grad():
            for frame, _ in loader:
                frame = frame.to(device).float()
                input_d.hook(None, None, frame)
                snn_fwd(frame)
                ann_fwd(frame)

        lif_rates = [d.mean for d in lif_d]
        snn_pool_rates = [d.mean for d in snn_pool_d]
        ann_pool_nz = [d.nonzero for d in ann_pool_d]
        relu_nz = [d.nonzero for d in relu_d]
        snn_in = [None] + snn_pool_rates + [lif_rates[5]]
        ann_in = [None] + ann_pool_nz + [relu_nz[5]]

        rows = []
        tot = {"snn": 0.0, "ann_dense": 0.0, "ann_sparse": 0.0}
        for (name, macs), rs, ra in zip(LAYER_MACS, snn_in, ann_in):
            dense = macs * T * E_MAC
            if rs is None:
                snn_e, ann_sparse_e = dense, dense
            else:
                snn_e = macs * T * rs * E_AC
                ann_sparse_e = macs * T * ra * E_MAC
            tot["snn"] += snn_e
            tot["ann_dense"] += dense
            tot["ann_sparse"] += ann_sparse_e
            rows.append(dict(layer=name, macs_per_frame=macs, snn_in_density=rs,
                             ann_in_nonzero=ra, snn_uJ=snn_e * 1e6,
                             ann_dense_uJ=dense * 1e6, ann_sparse_uJ=ann_sparse_e * 1e6))
        dense_ops = sum(macs for _, macs in LAYER_MACS) * T
        in_nz = input_d.nonzero
        snn_eff_macs = LAYER_MACS[0][1] * T * in_nz
        snn_eff_acs = sum(macs * T * r for (_, macs), r in zip(LAYER_MACS[1:], snn_in[1:]))
        ann_eff_macs = LAYER_MACS[0][1] * T * in_nz + \
            sum(macs * T * r for (_, macs), r in zip(LAYER_MACS[1:], ann_in[1:]))
        snn_act = 1 - sum(d.value_sum for d in lif_d) / sum(d.count for d in lif_d)
        ann_act = 1 - sum(d.nonzero_sum for d in relu_d) / sum(d.count for d in relu_d)
        snn_l2 = sum(r["snn_uJ"] for r in rows[1:])
        ann_l2 = sum(r["ann_dense_uJ"] for r in rows[1:])

        lines = [f"CIFAR10-DVS seed {seed}, {ckpt}.pt (SNN epoch {snn_epoch}, ANN epoch "
                 f"{ann_epoch}), T={T}, n={n_te}",
                 f"{'layer':<7} {'MACs/frame':>14} {'SNN in':>8} {'ANN in':>8} "
                 f"{'SNN uJ':>9} {'ANN dense':>10} {'ANN sparse':>11}"]
        for r in rows:
            rs = "analog" if r["snn_in_density"] is None else f"{r['snn_in_density']:.4f}"
            ra = "analog" if r["ann_in_nonzero"] is None else f"{r['ann_in_nonzero']:.4f}"
            lines.append(f"{r['layer']:<7} {r['macs_per_frame']:>14,} {rs:>8} {ra:>8} "
                         f"{r['snn_uJ']:>9.3f} {r['ann_dense_uJ']:>10.3f} {r['ann_sparse_uJ']:>11.3f}")
        lines += ["-" * 72,
                  f"{'TOTAL':<7} {'':>14} {'':>8} {'':>8} {tot['snn'] * 1e6:>9.3f} "
                  f"{tot['ann_dense'] * 1e6:>10.3f} {tot['ann_sparse'] * 1e6:>11.3f}",
                  f"\nANN dense / SNN:  {tot['ann_dense'] / tot['snn']:.1f}x saving",
                  f"ANN sparse / SNN: {tot['ann_sparse'] / tot['snn']:.1f}x saving",
                  f"spiking layers only (conv1 excluded from both): {snn_l2:.1f} uJ vs "
                  f"{ann_l2:.1f} uJ dense, {ann_l2 / snn_l2:.0f}x",
                  f"conv1 share of the SNN bill: {rows[0]['snn_uJ'] / (tot['snn'] * 1e6):.1%}",
                  "mean LIF firing rates, layer order: " + " ".join(f"{r:.3f}" for r in lif_rates),
                  "post-pool spike densities, pool order: " + " ".join(f"{r:.3f}" for r in snn_pool_rates),
                  "ANN ReLU nonzero fractions, layer order: " + " ".join(f"{r:.3f}" for r in relu_nz),
                  "ANN post-pool nonzero fractions, pool order: " + " ".join(f"{r:.3f}" for r in ann_pool_nz),
                  f"activation sparsity: SNN {snn_act:.1%} of neuron-timesteps silent, "
                  f"ANN {ann_act:.1%} of ReLU outputs zero",
                  f"input frame: nonzero fraction {in_nz:.4f}, mean count {input_d.mean:.4f}",
                  "\nNeuroBench-style counts per inference (first layer at nonzero-input fraction "
                  "for both):",
                  f"  Dense synaptic ops   {dense_ops:,.0f}",
                  f"  SNN Effective_MACs   {snn_eff_macs:,.0f}   Effective_ACs {snn_eff_acs:,.0f}",
                  f"  ANN Effective_MACs   {ann_eff_macs:,.0f}   Effective_ACs 0",
                  "note: pJ table bills conv1 at full MAC cost for both models (conservative); "
                  f"E_MAC={E_MAC * 1e12}pJ E_AC={E_AC * 1e12}pJ (Horowitz 2014, 45nm); "
                  "accounting model, not measured power"]
        report = "\n".join(lines)
        print(report)
        summary = dict(
            dataset="cifar10dvs", seed=seed, ckpt=ckpt, snn_epoch=snn_epoch, ann_epoch=ann_epoch,
            T=T, n=n_te, E_MAC_pJ=E_MAC * 1e12, E_AC_pJ=E_AC * 1e12, layers=rows,
            snn_mJ=tot["snn"] * 1e3, ann_dense_mJ=tot["ann_dense"] * 1e3,
            ann_sparse_mJ=tot["ann_sparse"] * 1e3,
            ratio_dense=tot["ann_dense"] / tot["snn"], ratio_sparse=tot["ann_sparse"] / tot["snn"],
            spiking_layers_ratio=ann_l2 / snn_l2,
            conv1_share_snn=rows[0]["snn_uJ"] / (tot["snn"] * 1e6),
            lif_rates=lif_rates, snn_pool_densities=snn_pool_rates,
            ann_relu_nonzero=relu_nz, ann_pool_nonzero=ann_pool_nz,
            snn_activation_sparsity=snn_act, ann_activation_sparsity=ann_act,
            input_nonzero=in_nz, input_mean_count=input_d.mean,
            neurobench=dict(dense_ops=dense_ops, snn_effective_macs=snn_eff_macs,
                            snn_effective_acs=snn_eff_acs, ann_effective_macs=ann_eff_macs,
                            ann_effective_acs=0.0))
        with open(f"{eval_dir(seed)}/energy_report{suffix(ckpt)}.txt", "w") as f:
            f.write(report + "\n")
        with open(f"{eval_dir(seed)}/energy{suffix(ckpt)}.json", "w") as f:
            json.dump(summary, f, indent=1)
        path = f"{eval_dir(seed)}/energy{suffix(ckpt)}.json"

    vol.commit()
    print(f"wrote {path.replace(DATA, '')}")


# ---------------------------------------------------------------- entrypoint

@app.local_entrypoint()
def main(mode: str = "train", model: str = "snn", seed: int = 0, epochs: int = 64,
         ckpt: str = "last", wait: bool = False, force: bool = False):
    """prepare and train spawn and return (safe to close the window; watch the
    dashboard). The four evaluation modes block and print their results.

    Spawning only survives under `modal run --detach`: without it Modal stops
    the ephemeral app as soon as this function returns and cancels the call.
    So refuse to spawn unless --detach is on the command line."""
    import sys

    if mode in ("prepare", "train") and not wait and "--detach" not in sys.argv:
        print(f"ERROR: --mode {mode} spawns a long call and needs `modal run --detach`.\n"
              f"       Rerun as: python -m modal run --detach scripts/step11_cifar10dvs.py "
              f"--mode {mode}" + (f" --model {model} --seed {seed}" if mode == "train" else "")
              + "\n       (or add --wait to stream the log; then keep the window open)")
        return
    if mode == "prepare":
        if wait:
            prepare.remote()
        else:
            call = prepare.spawn()
            print(f"spawned prepare: function call {call.object_id}. Safe to close this window.")
    elif mode == "train":
        if wait:
            train.remote(model=model, seed=seed, epochs=epochs, force=force)
        else:
            call = train.spawn(model=model, seed=seed, epochs=epochs, force=force)
            print(f"spawned {model} seed {seed}: function call {call.object_id}. "
                  f"Safe to close this window. Logs: modal dashboard.")
    else:
        evaluate.remote(mode=mode, seed=seed, ckpt=ckpt)
