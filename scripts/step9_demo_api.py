"""Step 9: live demo backend (Modal, CPU, scales to zero) + local dev server.

Serves both best checkpoints behind a small FastAPI app so the portfolio's
demo page (tayyabmanan.com/demo/spikes) can corrupt a test gesture and run
the SNN and its ANN twin on the identical corrupted input, live.

Endpoints (JSON; the X-Demo-Key header is required whenever DEMO_KEY is set):
  GET  /health   {"ok": true, "samples": 33, "warm": true}
  GET  /bank     class names and the bank samples (test id, label)
  POST /run      {"sample": 0..32, "kind": clean|drop|noise|occlude|tshuffle,
                  "level": 0..4, "seed": 0..999999}
                 returns, per model: scores, prediction, normalized confidence
                 (max/sum, never softmax), the cumulative readout over T,
                 timings; for the SNN also LIF firing rates and the
                 accounting-model energy of this sample; plus the corrupted
                 frames, 2x2 sum-pooled to 64x64, uint8, base64.

Bank: 33 test samples, 3 per class, chosen where BOTH models are right on
the clean input (results/data/test_outputs.npz), so every disagreement the
demo shows is caused by the corruption. Ids are test-set indices in
SpikingJelly's order (steps 5-7 load the test set with shuffle=False).

Corruptions are the step 6 functions verbatim. A live run is a fresh seeded
draw at batch 1, so its exact events differ from the published batched run.
The population numbers on the page come from the published dump.

One-time:
  modal secret create spikes-demo-key DEMO_KEY=<long random string>
  modal run scripts/step9_demo_api.py --mode prepare
  modal volume get --force dvs128-data /eval/demo/thumbs "<portfolio>/public/demo/spikes"
Deploy (prints the URL, ends in .modal.run):
  modal deploy scripts/step9_demo_api.py
  modal run scripts/step9_demo_api.py --mode smoke
  A warm container from the previous deploy keeps serving, with the OLD
  code and the OLD secret value, until it idles out (scaledown_window,
  5 min). After changing the secret or the code, either wait that long
  with no requests or `modal app stop dvs128-demo` before deploying.
  GET /version (ungated) shows which code and key fingerprint a container
  holds.
Local dev server (random weights unless results/checkpoints/ has the .pt files):
  python scripts/step9_demo_api.py --local --port 8009
"""

import base64
import os
import sys
import time

import modal

app = modal.App("dvs128-demo")

vol = modal.Volume.from_name("dvs128-data", create_if_missing=True)
DATA = "/data"
ROOT = f"{DATA}/DVS128Gesture"
# `modal volume get` writes a remote directory under the destination using
# the directory's own name, so /eval/demo/thumbs lands as <dest>/thumbs, the
# exact folder the page reads (public/demo/spikes/thumbs).
BANK_PATH = f"{DATA}/eval/demo/bank.npz"
THUMB_DIR = f"{DATA}/eval/demo/thumbs"

# CPU wheels: the image is a fraction of the CUDA one, so cold starts are
# short. Same pins as every other step; only the wheel index differs.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0", "torchvision==0.19.0",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install("spikingjelly==0.0.0.0.14", "numpy<2", "fastapi[standard]",
                 "pillow")
)

T = 16
N_CLASSES = 11
# CPU cores requested for the web container. Also the torch thread count:
# os.cpu_count() inside a Modal container reports the host's cores, and
# oversubscribing a 4-core quota makes every forward slower, not faster.
DEMO_CPUS = 4

# DVS128Gesture gesture_mapping.csv, labels 1..11 shifted to 0..10.
# prepare_bank() re-reads the csv on the volume and warns on any mismatch.
CLASS_NAMES = [
    "hand clapping", "right hand wave", "left hand wave",
    "right hand clockwise", "right hand counter clockwise",
    "left hand clockwise", "left hand counter clockwise",
    "forearm roll backward", "drums", "guitar", "random other gestures",
]

# Test-set indices: the first three samples of each class that BOTH models
# classify correctly on the clean input (from results/data/test_outputs.npz).
BANK_IDS = [
    0, 1, 2, 24, 25, 26, 48, 49, 50, 72, 73, 74, 96, 97, 98,
    120, 121, 122, 144, 145, 146, 168, 169, 170, 192, 193, 194,
    240, 242, 243, 264, 265, 266,
]
BANK_LABELS = [i // 3 for i in range(len(BANK_IDS))]

# ---- step 6, verbatim -------------------------------------------------------
SEVERITIES = {
    "drop": [0.2, 0.4, 0.6, 0.8],
    "noise": [0.05, 0.1, 0.2, 0.5],
    "occlude": [24, 40, 56, 72],
    "tshuffle": [2, 4, 8, 16],
}
SEED_OFFSET = {"clean": 0, "drop": 1, "noise": 2, "occlude": 3, "tshuffle": 4}
KINDS = ["clean", "drop", "noise", "occlude", "tshuffle"]

# ---- step 7, verbatim -------------------------------------------------------
E_MAC, E_AC = 4.6e-12, 0.9e-12   # joules, Horowitz 2014, 45 nm
LAYER_MACS = [
    ("conv1", 2 * 128 * 9 * 128 * 128),      # input: analog frames
    ("conv2", 128 * 128 * 9 * 64 * 64),      # input: LIF1 spikes (post-pool)
    ("conv3", 128 * 128 * 9 * 32 * 32),
    ("conv4", 128 * 128 * 9 * 16 * 16),
    ("conv5", 128 * 128 * 9 * 8 * 8),
    ("fc1",   2048 * 512),                   # input: LIF5 spikes (post-pool)
    ("fc2",   512 * 110),                    # input: LIF6 spikes
]


def build_snn():
    """Copied verbatim from step 3."""
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
    """Copied verbatim from step 4 (voting is applied in forward)."""
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


def corrupt(frame, kind, sev, gen):
    """Step 6 verbatim. frame: [B, T, 2, 128, 128] float counts."""
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
        T_ = frame.shape[1]
        w = int(sev)
        for start in range(0, T_, w):
            end = min(start + w, T_)
            perm = torch.randperm(end - start, generator=gen,
                                  device=frame.device) + start
            out[:, start:end] = frame[:, perm]
        return out
    raise ValueError(kind)


def run_seed(kind, level, test_id, seed):
    """Deterministic per (condition, sample, reroll). Step 6's base seed,
    plus the sample and reroll so no two cells share a stream."""
    return 1000 + level * 7 + SEED_OFFSET[kind] + 101 * test_id + 10007 * seed


def normalized_confidence(scores):
    """Confidence = max / sum of the raw rate outputs. Never softmax:
    these are MSE-trained rates, not logits (CLAUDE.md, step 5)."""
    s = float(scores.sum())
    if s <= 0:
        return 0.0
    return float(scores.max() / s)


class Engine:
    """Both models + the sample bank, in memory. One forward per model per
    request. CPU only; the whole thing fits in a couple of GB."""

    def __init__(self, snn, ann, frames, labels, ids, names):
        import threading

        import torch
        from spikingjelly.activation_based import layer, neuron

        # One run at a time. The LIF layers carry membrane state between
        # forward and reset, and the rate hooks write into shared lists, so
        # two overlapping requests (FastAPI runs sync handlers in a thread
        # pool) would corrupt each other's predictions and energy figures.
        self._lock = threading.Lock()
        self.snn, self.ann = snn.eval(), ann.eval()
        self.frames = frames            # [N, T, 2, 128, 128] uint16
        self.labels = [int(x) for x in labels]
        self.ids = [int(x) for x in ids]
        self.names = list(names)
        self.torch = torch

        # Firing-rate hooks, exactly as in step 7: mean LIF output (spikes per
        # neuron-timestep) and post-pool spike density (input to the next
        # layer). Overwritten on every forward; at batch 1 they are the
        # sample's own rates.
        self.lifs = [m for m in snn.modules() if isinstance(m, neuron.LIFNode)]
        self.pools = [m for m in snn.modules() if isinstance(m, layer.MaxPool2d)]
        self.lif_rate = [0.0] * len(self.lifs)
        self.lif_numel = [0] * len(self.lifs)
        self.pool_rate = [0.0] * len(self.pools)

        def lif_hook(i):
            def hook(module, inp, out):
                self.lif_rate[i] = out.mean().item()
                self.lif_numel[i] = out.numel()
            return hook

        def pool_hook(i):
            def hook(module, inp, out):
                self.pool_rate[i] = out.mean().item()
            return hook

        for i, m in enumerate(self.lifs):
            m.register_forward_hook(lif_hook(i))
        for i, m in enumerate(self.pools):
            m.register_forward_hook(pool_hook(i))

    def energy_mj(self):
        """Step 7 accounting for the sample that was just run. conv1 is billed
        at full MAC cost (analog input, conservative against the SNN)."""
        in_rates = [None] + self.pool_rate + [self.lif_rate[5]]
        snn_e = ann_e = 0.0
        for (name, macs), r in zip(LAYER_MACS, in_rates):
            ann_e += macs * T * E_MAC
            snn_e += macs * T * E_MAC if r is None else macs * T * r * E_AC
        return snn_e * 1e3, ann_e * 1e3

    @staticmethod
    def readout(per_step):
        """per_step: [T, 11] numpy. Cumulative mean = what the rate readout
        would report if it stopped after t steps."""
        import numpy as np

        cum = np.cumsum(per_step, axis=0) / np.arange(1, T + 1)[:, None]
        return {
            "per_step": np.round(cum, 4).tolist(),
            "step_pred": cum.argmax(1).tolist(),
            "step_conf": [round(normalized_confidence(row), 4) for row in cum],
        }

    def run(self, sample, kind, level, seed):
        with self._lock:
            return self._run(sample, kind, level, seed)

    def _run(self, sample, kind, level, seed):
        import numpy as np
        from spikingjelly.activation_based import functional

        torch = self.torch
        test_id, label = self.ids[sample], self.labels[sample]
        frame = torch.from_numpy(self.frames[sample].astype(np.float32))[None]

        severity = None
        if kind != "clean" and level > 0:
            severity = SEVERITIES[kind][level - 1]
            gen = torch.Generator()
            gen.manual_seed(run_seed(kind, level, test_id, seed))
            frame = corrupt(frame, kind, severity, gen)
        else:
            kind, level = "clean", 0

        with torch.no_grad():
            # SNN: [T, B, ...] in, [T, B, 11] rates out, reset between samples
            t0 = time.perf_counter()
            snn_steps = self.snn(frame.transpose(0, 1))[:, 0, :]
            functional.reset_net(self.snn)
            snn_ms = (time.perf_counter() - t0) * 1e3
            snn_rates = list(self.lif_rate)
            silent = 1.0 - (sum(r * n for r, n in zip(self.lif_rate, self.lif_numel))
                            / max(1, sum(self.lif_numel)))
            snn_e, ann_e = self.energy_mj()

            # ANN: fold T into the batch, vote, average over T
            t0 = time.perf_counter()
            x = frame[0]                                   # [T, 2, 128, 128]
            ann_steps = self.ann(x).view(T, N_CLASSES, 10).mean(2)
            ann_ms = (time.perf_counter() - t0) * 1e3

        snn_steps = snn_steps.numpy()
        ann_steps = ann_steps.numpy()

        def model_block(steps):
            scores = steps.mean(0)
            pred = int(scores.argmax())
            block = {
                "scores": np.round(scores, 4).tolist(),
                "pred": pred,
                "conf": round(normalized_confidence(scores), 4),
                "correct": pred == label,
            }
            block.update(self.readout(steps))
            return block

        snn_block = model_block(snn_steps)
        snn_block.update({
            "lif_rates": [round(r, 4) for r in snn_rates],
            "pool_density": [round(r, 4) for r in self.pool_rate],
            "silent_fraction": round(silent, 4),
            "energy_mj": round(snn_e, 3),
            "ms": round(snn_ms, 1),
        })
        ann_block = model_block(ann_steps)
        ann_block.update({"energy_mj": round(ann_e, 3), "ms": round(ann_ms, 1)})

        # Transport: 2x2 sum-pool to 64x64, clip to uint8. Only the picture
        # is downsampled; both models saw the full 128x128 tensor.
        f = frame[0].view(T, 2, 64, 2, 64, 2).sum((3, 5))
        f8 = f.clamp(0, 255).to(torch.uint8).contiguous().numpy()

        return {
            "sample": sample,
            "test_id": test_id,
            "label": label,
            "label_name": self.names[label],
            "kind": kind,
            "level": level,
            "severity": severity,
            "seed": seed,
            "T": T,
            "frames": {
                "shape": [T, 2, 64, 64],
                "dtype": "uint8",
                "max": int(f8.max()),
                "b64": base64.b64encode(f8.tobytes()).decode("ascii"),
            },
            "snn": snn_block,
            "ann": ann_block,
        }


def make_app(engine, key=None, require_key=False):
    """FastAPI app over an Engine. `key` (DEMO_KEY) gates every route; the
    portfolio's /api/spikes proxy adds it server-side so the URL can stay
    public without being free compute for anyone who finds it. With
    `require_key` (the deployed app) an empty key is a startup error rather
    than an open API; only the local dev server may run without one."""
    from typing import Literal, Optional

    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel, Field

    expected = (key or "").strip()
    if require_key and not expected:
        raise RuntimeError("DEMO_KEY is empty or unset: refusing to serve the API unauthenticated")

    # No docs, no schema endpoint: the routes are private to the proxy.
    web = FastAPI(title="dvs128-demo", docs_url=None, redoc_url=None, openapi_url=None)
    n_samples = len(engine.ids)

    class RunBody(BaseModel):
        sample: int = Field(ge=0, lt=n_samples)
        kind: Literal["clean", "drop", "noise", "occlude", "tshuffle"] = "clean"
        level: int = Field(default=0, ge=0, le=4)
        seed: int = Field(default=0, ge=0, le=999_999)

    # Compared after stripping whitespace on both sides: a secret pasted into
    # a dashboard box, or a header built from an env file, can pick up a
    # trailing newline or space that no one can see. Constant-time compare
    # on bytes, so response timing never leaks how many leading characters
    # matched and a non-ASCII header value is a 401, not a 500.
    import hmac

    expected_bytes = expected.encode("utf-8")

    def check(x_demo_key):
        supplied = (x_demo_key or "").strip().encode("utf-8")
        if expected and not hmac.compare_digest(supplied, expected_bytes):
            raise HTTPException(status_code=401, detail="bad or missing key")

    @web.get("/version")
    def version():
        """Ungated. A fingerprint of the key this container holds (first 8
        hex chars of its sha256, not reversible), so a key mismatch between
        the deployment and the proxy can be diagnosed without revealing
        either side's value."""
        import hashlib

        return {
            "code": "2026-09-08",
            "key_sha8": hashlib.sha256(expected.encode()).hexdigest()[:8] if expected else None,
        }

    @web.get("/health")
    def health(x_demo_key: Optional[str] = Header(default=None)):
        check(x_demo_key)
        return {"ok": True, "samples": n_samples, "warm": True}

    @web.get("/bank")
    def bank(x_demo_key: Optional[str] = Header(default=None)):
        check(x_demo_key)
        return {
            "classes": engine.names,
            "T": T,
            "severities": SEVERITIES,
            "samples": [
                {"index": i, "test_id": tid, "label": lab,
                 "label_name": engine.names[lab]}
                for i, (tid, lab) in enumerate(zip(engine.ids, engine.labels))
            ],
        }

    @web.post("/run")
    def run(body: RunBody, x_demo_key: Optional[str] = Header(default=None)):
        check(x_demo_key)
        return engine.run(body.sample, body.kind, body.level, body.seed)

    return web


def load_bank(path):
    import numpy as np

    d = np.load(path)
    return d["frames"], d["labels"], d["ids"], [str(n) for n in d["names"]]


def load_engine(snn_path, ann_path, bank_path, threads=DEMO_CPUS):
    import torch
    from spikingjelly.activation_based import functional

    torch.set_num_threads(max(1, threads))
    snn = build_snn()
    functional.set_step_mode(snn, "m")
    snn.load_state_dict(torch.load(snn_path, map_location="cpu")["net"])
    ann = build_ann()
    ann.load_state_dict(torch.load(ann_path, map_location="cpu")["net"])
    frames, labels, ids, names = load_bank(bank_path)
    return Engine(snn, ann, frames, labels, ids, names)


# ---- Modal: bank preparation --------------------------------------------------

@app.function(image=image, volumes={DATA: vol}, cpu=4, timeout=1800)
def prepare_bank():
    """Pull the 33 bank samples out of the cached test frames, save them as
    one npz, and render a 64x64 alpha-only thumbnail per sample (event
    density as alpha, so the page can ink it in either theme via CSS mask)."""
    import csv

    import numpy as np
    from PIL import Image
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

    ds = DVS128Gesture(ROOT, train=False, data_type="frame",
                       frames_number=T, split_by="number")
    print("test samples:", len(ds))

    names = list(CLASS_NAMES)
    csv_path = f"{ROOT}/download/gesture_mapping.csv"
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            rows = [r for r in csv.reader(f)
                    if len(r) >= 2 and r[1].strip().isdigit()]
        csv_names = {int(r[1]) - 1: r[0].strip().replace("_", " ")
                     for r in rows}
        for i in range(N_CLASSES):
            if csv_names.get(i) and csv_names[i] != names[i]:
                print(f"WARNING class {i}: csv says {csv_names[i]!r}, "
                      f"CLASS_NAMES says {names[i]!r}; using the csv")
                names[i] = csv_names[i]

    frames, labels = [], []
    for tid in BANK_IDS:
        frame, label = ds[tid]
        frames.append(np.asarray(frame, dtype=np.float32))
        labels.append(int(label))
    frames = np.stack(frames)
    assert labels == BANK_LABELS, f"label order changed: {labels}"
    print("frames", frames.shape, "max count", frames.max(),
          "mean density", round(float((frames > 0).mean()), 4))
    frames_u16 = np.clip(frames, 0, 65535).astype(np.uint16)

    os.makedirs(os.path.dirname(BANK_PATH), exist_ok=True)
    np.savez_compressed(BANK_PATH, frames=frames_u16,
                        labels=np.array(labels), ids=np.array(BANK_IDS),
                        names=np.array(names))

    os.makedirs(THUMB_DIR, exist_ok=True)
    for k in range(len(BANK_IDS)):
        density = frames[k].sum((0, 1))                         # [128, 128]
        density = density.reshape(64, 2, 64, 2).sum((1, 3))     # [64, 64]
        scale = max(1e-6, float(np.percentile(density, 99)))
        alpha = np.clip(density / scale, 0, 1) ** 0.6
        la = np.zeros((64, 64, 2), dtype=np.uint8)
        la[..., 1] = (alpha * 255).astype(np.uint8)
        Image.fromarray(la, mode="LA").save(f"{THUMB_DIR}/{k}.png")
    vol.commit()
    print(f"wrote {BANK_PATH} and {len(BANK_IDS)} thumbnails to {THUMB_DIR}")


# ---- Modal: the web app ---------------------------------------------------------

@app.cls(
    image=image,
    volumes={DATA: vol},
    secrets=[modal.Secret.from_name("spikes-demo-key")],
    cpu=DEMO_CPUS,
    memory=3072,
    scaledown_window=300,        # idle five minutes, then scale to zero
    max_containers=2,
    enable_memory_snapshot=True,  # models + bank restored from a snapshot
)
@modal.concurrent(max_inputs=2)
class Demo:
    @modal.enter(snap=True)
    def load(self):
        t0 = time.perf_counter()
        self.engine = load_engine(f"{DATA}/checkpoints/best.pt",
                                  f"{DATA}/checkpoints_ann/best.pt", BANK_PATH)
        print(f"loaded both models + {len(self.engine.ids)} samples "
              f"in {time.perf_counter() - t0:.1f}s")

    @modal.asgi_app()
    def web(self):
        return make_app(self.engine, os.environ.get("DEMO_KEY"), require_key=True)

    @modal.method()
    def smoke(self):
        out = self.engine.run(0, "noise", 3, 0)
        return {k: out[k] for k in ("label_name", "kind", "severity")} | {
            "snn": {k: out["snn"][k] for k in ("pred", "conf", "correct",
                                               "silent_fraction", "energy_mj", "ms")},
            "ann": {k: out["ann"][k] for k in ("pred", "conf", "correct",
                                               "energy_mj", "ms")},
        }


@app.local_entrypoint()
def main(mode: str = "smoke"):
    if mode == "prepare":
        prepare_bank.remote()
    elif mode == "smoke":
        print(Demo().smoke.remote())
    else:
        raise SystemExit(f"unknown mode {mode!r}: use prepare or smoke")


# ---- local dev server -------------------------------------------------------------

def _synthetic_bank(rng):
    """Sparse Poisson background plus a moving blob per class, so the local
    page has something gesture-shaped to draw. Not data; dev only."""
    import numpy as np

    n = len(BANK_IDS)
    frames = np.zeros((n, T, 2, 128, 128), dtype=np.float32)
    yy, xx = np.mgrid[0:128, 0:128]
    for k in range(n):
        c = BANK_LABELS[k]
        for t in range(T):
            ang = 2 * np.pi * (t / T) * (1 + c / 6)
            cy, cx = 64 + 30 * np.sin(ang + c), 64 + 30 * np.cos(ang * 0.7 + k)
            blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 9.0 ** 2))
            frames[k, t, 0] = rng.poisson(blob * 1.2)
            frames[k, t, 1] = rng.poisson(blob * 1.6)
            frames[k, t] += rng.poisson(0.004, size=(2, 128, 128))
    return frames.astype(np.uint16)


def _local(port, thumbs_out=None):
    import numpy as np
    import torch
    import uvicorn
    from spikingjelly.activation_based import functional

    here = os.path.dirname(os.path.abspath(__file__))
    ck_dir = os.path.join(here, "..", "results", "checkpoints")
    snn_path = os.path.join(ck_dir, "snn_best.pt")
    ann_path = os.path.join(ck_dir, "ann_best.pt")
    bank_local = os.path.join(here, "..", "results", "data", "demo_bank.npz")

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    torch.manual_seed(0)
    snn = build_snn()
    functional.set_step_mode(snn, "m")
    ann = build_ann()
    if os.path.exists(snn_path) and os.path.exists(ann_path):
        snn.load_state_dict(torch.load(snn_path, map_location="cpu")["net"])
        ann.load_state_dict(torch.load(ann_path, map_location="cpu")["net"])
        print("local: real checkpoints")
    else:
        print("local: RANDOM weights (no results/checkpoints/*_best.pt)")

    if os.path.exists(bank_local):
        frames, labels, ids, names = load_bank(bank_local)
        print("local: real bank", frames.shape)
    else:
        rng = np.random.default_rng(0)
        frames, labels, ids, names = (_synthetic_bank(rng), BANK_LABELS,
                                      BANK_IDS, CLASS_NAMES)
        print("local: SYNTHETIC bank", frames.shape)

    if thumbs_out:
        from PIL import Image

        os.makedirs(thumbs_out, exist_ok=True)
        for k in range(len(ids)):
            density = frames[k].astype(np.float32).sum((0, 1))
            density = density.reshape(64, 2, 64, 2).sum((1, 3))
            scale = max(1e-6, float(np.percentile(density, 99)))
            alpha = np.clip(density / scale, 0, 1) ** 0.6
            la = np.zeros((64, 64, 2), dtype=np.uint8)
            la[..., 1] = (alpha * 255).astype(np.uint8)
            Image.fromarray(la, mode="LA").save(os.path.join(thumbs_out, f"{k}.png"))
        print(f"local: wrote {len(ids)} thumbnails to {thumbs_out}")

    engine = Engine(snn, ann, frames, labels, ids, names)
    web = make_app(engine, os.environ.get("DEMO_KEY"))
    uvicorn.run(web, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__" and "--local" in sys.argv:
    argv = sys.argv
    port = int(argv[argv.index("--port") + 1]) if "--port" in argv else 8009
    thumbs = argv[argv.index("--thumbs-out") + 1] if "--thumbs-out" in argv else None
    _local(port, thumbs)
