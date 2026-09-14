"""Step 12: membrane time-constant ablation on DVS128Gesture, SNN side only.

Why. The noise sweep (step 6b) says the SNN's robustness to event noise is a
temporal filter: large for uncorrelated noise, gone or reversed once the
noise field persists. The filter is the LIF leak, v <- v + (x - v) / tau, so
its strength should scale with tau. Training the SNN at several tau against
the SAME ANN twins tests that directly:
  tau = 1.1   nearly memoryless: v = 0.09 v + 0.91 x, a per-frame threshold unit
  tau = 2.0   the baseline (seeds 0-4 already trained by step 3)
  tau = 4.0   v = 0.75 v + 0.25 x
  tau = 8.0   v = 0.875 v + 0.125 x, a slow integrator
Prediction if the leak explains the result: the i.i.d.-noise advantage and
the persistent-noise penalty both grow with tau, and both shrink toward zero
at tau 1.1. If they do not move with tau, the mechanism is elsewhere
(thresholding, rate coding) and the paper must say so.

This is an SNN-side ablation. The ANN has no tau; every pair compares
SNN(tau, seed s) with the existing ANN(seed s) on byte-identical inputs. The
recipe is otherwise step 3's exactly: same net, MSE on one-hot, Adam 1e-3,
cosine, batch 16, 64 epochs, T=16, dropout 0.5, shared seed.

Modes (always python -m modal; --detach for train and eval, both spawn):
  python -m modal run --detach scripts/step12_tau_ablation.py --mode train --tau 4.0 --seed 0
      GPU, ~1 h, resumable exactly like step 11 (atomic checkpoints with RNG
      state, heartbeat lock, Modal retries, 6 h timeout).
  python -m modal run --detach scripts/step12_tau_ablation.py --mode eval --tau 4.0 --seed 0
      GPU, ~10 min. Runs dump, corrupt, sweep and energy in one container,
      last.pt, and writes the same five files steps 5-7 write.
Layout on the volume:
  checkpoints_tau/tau{tau}/seed{S}/   SNN only: last.pt, best.pt, metrics.csv, run.json
  eval_tau/tau{tau}/seed{S}/          test_outputs.npz, corruption_outputs.npz,
                                      noise_sweep_outputs.npz, energy.json, energy_report.txt
Fetch to results/data/tau/tau{tau}/seed{S}/ (same names, plus metrics.csv as
step3_metrics.csv). Baseline tau 2.0 is read from results/data/seed{S}/.
Analyse: python scripts/step12_tau_analysis.py

Corruptions, the sweep and the energy accounting are copied from steps 6,
6b and 7 with the same fixed seeds, so every tau sees byte-identical
corrupted inputs, the same ones the baseline saw.
"""

import modal

app = modal.App("dvs128-tau-ablation")

vol = modal.Volume.from_name("dvs128-data", create_if_missing=True)
DATA = "/data"
ROOT = f"{DATA}/DVS128Gesture"
CKPT_TAU = f"{DATA}/checkpoints_tau"
CKPT_ANN = f"{DATA}/checkpoints_ann"
EVAL_TAU = f"{DATA}/eval_tau"
N_CLASSES = 11
VOTES = 10
LOCK_STALE_S = 45 * 60

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.0",
    "torchvision==0.19.0",
    "spikingjelly==0.0.0.0.14",
    "numpy<2",
    "tqdm",
)

E_MAC, E_AC = 4.6e-12, 0.9e-12
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


def tau_tag(tau: float) -> str:
    return f"tau{tau:g}"


def snn_dir(tau: float, seed: int) -> str:
    return f"{CKPT_TAU}/{tau_tag(tau)}/seed{seed}"


def ann_dir(seed: int) -> str:
    """The DVS128 ANN layout of step 4: seed 0 at the base folder."""
    return CKPT_ANN if seed == 0 else f"{CKPT_ANN}/seed{seed}"


def eval_dir(tau: float, seed: int) -> str:
    return f"{EVAL_TAU}/{tau_tag(tau)}/seed{seed}"


# ---------------------------------------------------------------- models

def build_snn(tau: float):
    """Step 3's net with the LIF time constant exposed."""
    import torch.nn as nn
    from spikingjelly.activation_based import layer, neuron, surrogate

    def lif():
        return neuron.LIFNode(tau=float(tau), surrogate_function=surrogate.ATan(),
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
        nn.Dropout(0.5), nn.Linear(512, N_CLASSES * VOTES), nn.ReLU(),
    )


def snn_forward(net):
    from spikingjelly.activation_based import functional
    functional.set_step_mode(net, "m")

    def fwd(frame):
        out = net(frame.transpose(0, 1)).mean(0)
        functional.reset_net(net)
        return out
    return fwd


def ann_forward(net):
    def fwd(frame):
        B, T_ = frame.shape[0], frame.shape[1]
        out = net(frame.reshape(B * T_, *frame.shape[2:]))
        return out.view(B, T_, N_CLASSES, VOTES).mean(3).mean(1)
    return fwd


# ---------------------------------------------------------------- data

def make_loader(is_train: bool, T: int, batch: int):
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
    from torch.utils.data import DataLoader
    ds = DVS128Gesture(ROOT, train=is_train, data_type="frame",
                       frames_number=T, split_by="number")
    return DataLoader(ds, batch_size=batch, shuffle=is_train, num_workers=4,
                      pin_memory=True, drop_last=is_train)


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


# ---------------------------------------------------------------- train

@app.function(image=image, gpu="A10G", volumes={DATA: vol}, timeout=6 * 3600, cpu=4,
              retries=modal.Retries(max_retries=3, initial_delay=60.0, backoff_coefficient=1.0))
def train(tau: float, seed: int, epochs: int = 64, batch: int = 16, lr: float = 1e-3,
          T: int = 16, force: bool = False):
    import json
    import os
    import random
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F

    assert tau > 1.0, "SpikingJelly's LIF needs tau > 1"
    device = "cuda"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rd = snn_dir(tau, seed)

    train_loader, test_loader = make_loader(True, T, batch), make_loader(False, T, batch)
    net = build_snn(tau).to(device)
    fwd = snn_forward(net)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    os.makedirs(rd, exist_ok=True)
    lock = f"{rd}/lock.json"
    if os.path.exists(lock) and not force:
        age = time.time() - os.path.getmtime(lock)
        if age < LOCK_STALE_S:
            raise RuntimeError(f"{lock} was touched {age / 60:.0f} min ago, another container "
                               f"may be training tau {tau} seed {seed}. Wait, or --force.")

    def touch_lock():
        with open(lock, "w") as f:
            json.dump({"tau": tau, "seed": seed, "time": time.time()}, f)

    def atomic_save(obj, path):
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    touch_lock()
    with open(f"{rd}/run.json", "w") as f:
        json.dump({"dataset": "dvs128gesture", "model": "snn", "tau": tau, "seed": seed,
                   "epochs": epochs, "batch": batch, "lr": lr, "T": T, "loss": "mse_onehot",
                   "optimizer": "adam", "schedule": "cosine"}, f, indent=1)
    last, best_path = f"{rd}/last.pt", f"{rd}/best.pt"
    start_epoch, best_acc = 0, 0.0
    if os.path.exists(last):
        ck = torch.load(last, map_location=device)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best_acc = ck["epoch"] + 1, ck["best_acc"]
        if "rng" in ck:
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
        print(f"tau {tau} seed {seed} epoch {epoch:3d}  train {tr_acc:.4f}  test {te_acc:.4f}"
              f"  best {best_acc:.4f}  {epoch_time:.0f}s")
        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{tr_loss:.4f},{tr_acc:.4f},{te_loss:.4f},{te_acc:.4f},"
                    f"{epoch_time:.1f}\n")
        state = {"net": net.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "epoch": epoch, "best_acc": best_acc,
                 "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
                         "numpy": np.random.get_state(), "random": random.getstate()}}
        atomic_save(state, last)
        if te_acc >= best_acc:
            atomic_save(state, best_path)
        touch_lock()
        vol.commit()

    if os.path.exists(lock):
        os.remove(lock)
    vol.commit()
    print(f"done. tau {tau} seed {seed}: last-epoch test acc {te_acc:.4f} (report this), "
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

@app.function(image=image, gpu="A10G", volumes={DATA: vol}, timeout=2 * 3600, cpu=4,
              retries=modal.Retries(max_retries=1, initial_delay=60.0, backoff_coefficient=1.0))
def evaluate(tau: float, seed: int, ckpt: str = "last", T: int = 16, batch: int = 16):
    """dump + corrupt + sweep + energy for SNN(tau, seed) against ANN(seed)."""
    import json
    import os

    import numpy as np
    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import layer, neuron

    assert ckpt in ("last", "best"), ckpt
    device = "cuda"
    loader = make_loader(False, T, batch)

    snn = build_snn(tau).to(device)
    ck = torch.load(f"{snn_dir(tau, seed)}/{ckpt}.pt", map_location=device)
    snn.load_state_dict(ck["net"])
    snn_epoch = int(ck["epoch"])
    snn.eval()
    ann = build_ann().to(device)
    ck = torch.load(f"{ann_dir(seed)}/{ckpt}.pt", map_location=device)
    ann.load_state_dict(ck["net"])
    ann_epoch = int(ck["epoch"])
    ann.eval()
    fwds = {"snn": snn_forward(snn), "ann": ann_forward(ann)}
    out_dir = eval_dir(tau, seed)
    os.makedirs(out_dir, exist_ok=True)
    sfx = "" if ckpt == "last" else f"_{ckpt}"
    meta = dict(seed=seed, ckpt=ckpt, snn_epoch=snn_epoch, ann_epoch=ann_epoch,
                dataset="dvs128gesture", tau=tau)
    print(f"tau {tau} seed {seed}, {ckpt}.pt: SNN epoch {snn_epoch}, ANN epoch {ann_epoch}")

    def run_conditions(conditions, seed_fn, transform):
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

    # 1. dump, with the metrics.csv checksum
    labels, res = run_conditions([("clean", None)], lambda t, p: 0, None)
    for m, epoch, rd in (("snn", snn_epoch, snn_dir(tau, seed)), ("ann", ann_epoch, ann_dir(seed))):
        acc = (res[f"{m}_clean"].argmax(1) == labels).mean()
        logged = logged_test_acc(rd, epoch)
        flag = ""
        if logged is not None and abs(acc - logged) > 1.0 / len(labels) + 1e-6:
            flag = "   WARNING: differs from metrics.csv by more than one sample"
        print(f"{m.upper()} test acc {acc:.4f} (epoch {epoch}, metrics.csv logged "
              f"{'unknown' if logged is None else f'{logged:.4f}'}){flag}")
    np.savez(f"{out_dir}/test_outputs{sfx}.npz", labels=labels, snn_out=res["snn_clean"],
             ann_out=res["ann_clean"], **meta)
    vol.commit()
    print("wrote test_outputs")

    # 2. corruption suite, step 6 seeds
    conditions = [("clean_0", None)]
    for kind, sevs in SEVERITIES.items():
        for i, s in enumerate(sevs, start=1):
            conditions.append((f"{kind}_{i}", (kind, i, s)))
    labels, res = run_conditions(
        conditions, lambda t, p: 1000 if p is None else 1000 + p[1] * 7 + SEED_OFFSET[p[0]],
        lambda f, p, g: corrupt(f, p[0], p[2], g))
    np.savez(f"{out_dir}/corruption_outputs{sfx}.npz", labels=labels,
             severities=np.array([f"{k}:{','.join(str(s) for s in v)}"
                                  for k, v in SEVERITIES.items()]), **meta, **res)
    vol.commit()
    print("wrote corruption_outputs")

    # 3. noise temporal-correlation sweep, step 6b seeds
    conditions = [("clean", None)]
    for i, lam in enumerate(LAMS, start=1):
        for k in KS:
            conditions.append((f"lam{i}_k{k}", (lam, k)))
    labels, res = run_conditions(
        conditions, lambda t, p: 3000 if p is None else 3000 + LAMS.index(p[0]) * 31 + p[1],
        lambda f, p, g: block_noise(f, p[0], p[1], g))
    np.savez(f"{out_dir}/noise_sweep_outputs{sfx}.npz", labels=labels, lams=np.array(LAMS),
             ks=np.array(KS), **meta, **res)
    vol.commit()
    print("wrote noise_sweep_outputs")

    # 4. energy, step 7 accounting
    lifs = [m for m in snn.modules() if isinstance(m, neuron.LIFNode)]
    snn_pools = [m for m in snn.modules() if isinstance(m, layer.MaxPool2d)]
    relus = [m for m in ann.modules() if isinstance(m, nn.ReLU)]
    ann_pools = [m for m in ann.modules() if isinstance(m, nn.MaxPool2d)]
    lif_d, snn_pool_d = [Density() for _ in lifs], [Density() for _ in snn_pools]
    relu_d, ann_pool_d = [Density() for _ in relus], [Density() for _ in ann_pools]
    for ms, ds_ in ((lifs, lif_d), (snn_pools, snn_pool_d), (relus, relu_d), (ann_pools, ann_pool_d)):
        for m, d in zip(ms, ds_):
            m.register_forward_hook(d.hook)
    input_d = Density()
    with torch.no_grad():
        for frame, _ in loader:
            frame = frame.to(device).float()
            input_d.hook(None, None, frame)
            fwds["snn"](frame)
            fwds["ann"](frame)
    lif_rates = [d.mean for d in lif_d]
    snn_pool_rates = [d.mean for d in snn_pool_d]
    ann_pool_nz = [d.nonzero for d in ann_pool_d]
    relu_nz = [d.nonzero for d in relu_d]
    snn_in = [None] + snn_pool_rates + [lif_rates[5]]
    ann_in = [None] + ann_pool_nz + [relu_nz[5]]
    rows, tot = [], {"snn": 0.0, "ann_dense": 0.0, "ann_sparse": 0.0}
    for (name, macs), rs, ra in zip(LAYER_MACS, snn_in, ann_in):
        dense = macs * T * E_MAC
        snn_e = dense if rs is None else macs * T * rs * E_AC
        ann_sparse_e = dense if ra is None else macs * T * ra * E_MAC
        tot["snn"] += snn_e
        tot["ann_dense"] += dense
        tot["ann_sparse"] += ann_sparse_e
        rows.append(dict(layer=name, macs_per_frame=macs, snn_in_density=rs, ann_in_nonzero=ra,
                         snn_uJ=snn_e * 1e6, ann_dense_uJ=dense * 1e6, ann_sparse_uJ=ann_sparse_e * 1e6))
    snn_act = 1 - sum(d.value_sum for d in lif_d) / sum(d.count for d in lif_d)
    ann_act = 1 - sum(d.nonzero_sum for d in relu_d) / sum(d.count for d in relu_d)
    summary = dict(
        dataset="dvs128gesture", tau=tau, seed=seed, ckpt=ckpt, snn_epoch=snn_epoch,
        ann_epoch=ann_epoch, T=T, n=len(loader.dataset), E_MAC_pJ=E_MAC * 1e12, E_AC_pJ=E_AC * 1e12,
        layers=rows, snn_mJ=tot["snn"] * 1e3, ann_dense_mJ=tot["ann_dense"] * 1e3,
        ann_sparse_mJ=tot["ann_sparse"] * 1e3, ratio_dense=tot["ann_dense"] / tot["snn"],
        ratio_sparse=tot["ann_sparse"] / tot["snn"], lif_rates=lif_rates,
        snn_pool_densities=snn_pool_rates, ann_relu_nonzero=relu_nz, ann_pool_nonzero=ann_pool_nz,
        snn_activation_sparsity=snn_act, ann_activation_sparsity=ann_act,
        input_nonzero=input_d.nonzero, input_mean_count=input_d.mean)
    report = [f"tau {tau} seed {seed}, {ckpt}.pt: SNN {tot['snn'] * 1e3:.2f} mJ, ANN dense "
              f"{tot['ann_dense'] * 1e3:.2f} mJ ({summary['ratio_dense']:.1f}x), ANN sparse "
              f"{tot['ann_sparse'] * 1e3:.2f} mJ ({summary['ratio_sparse']:.1f}x)",
              "mean LIF firing rates, layer order: " + " ".join(f"{r:.3f}" for r in lif_rates),
              f"activation sparsity: SNN {snn_act:.1%} silent, ANN {ann_act:.1%} zero"]
    print("\n".join(report))
    with open(f"{out_dir}/energy_report{sfx}.txt", "w") as f:
        f.write("\n".join(report) + "\n")
    with open(f"{out_dir}/energy{sfx}.json", "w") as f:
        json.dump(summary, f, indent=1)
    vol.commit()
    print(f"wrote energy. all four outputs in eval_tau/{tau_tag(tau)}/seed{seed}/")


# ---------------------------------------------------------------- entrypoint

@app.local_entrypoint()
def main(mode: str = "train", tau: float = 4.0, seed: int = 0, epochs: int = 64,
         ckpt: str = "last", wait: bool = False, force: bool = False):
    """train and eval both spawn and return under `modal run --detach`; without
    --detach the app would stop and cancel the call, so the script refuses."""
    import sys

    if not wait and "--detach" not in sys.argv:
        print(f"ERROR: --mode {mode} spawns a long call and needs `modal run --detach`.\n"
              f"       Rerun as: python -m modal run --detach scripts/step12_tau_ablation.py "
              f"--mode {mode} --tau {tau} --seed {seed}\n"
              f"       (or add --wait to stream the log; then keep the window open)")
        return
    if mode == "train":
        if wait:
            train.remote(tau=tau, seed=seed, epochs=epochs, force=force)
        else:
            call = train.spawn(tau=tau, seed=seed, epochs=epochs, force=force)
            print(f"spawned train tau {tau} seed {seed}: {call.object_id}. Safe to close this window.")
    elif mode == "eval":
        if wait:
            evaluate.remote(tau=tau, seed=seed, ckpt=ckpt)
        else:
            call = evaluate.spawn(tau=tau, seed=seed, ckpt=ckpt)
            print(f"spawned eval tau {tau} seed {seed}: {call.object_id}. Safe to close this window. "
                  f"Check: python -m modal volume ls dvs128-data eval_tau/{tau_tag(tau)}/seed{seed}")
    else:
        raise SystemExit(f"unknown mode {mode}; use train or eval")
