"""Step 7 (GPU, ~2 min): firing rates, activation sparsity, and the energy
accounting table, for one seed and one checkpoint.

What this does:
  1. Runs the test set through the SNN with hooks on every LIF layer
     (mean firing rate, spikes per neuron per timestep) and after every
     MaxPool (spike density of the map the next layer actually reads).
  2. Runs the same test set through the ANN twin with hooks on every
     ReLU and MaxPool, measuring the fraction of NONZERO activations.
     ReLU outputs are sparse too, and a zero-skipping accelerator can
     exploit that, so the ANN gets the same sparsity-aware treatment
     as the SNN in a second column. Dampfhoffer et al. (2023) and the
     NeuroBench harness both insist on this.
  3. Computes the energy accounting used across the SNN literature:
       ANN dense:   every layer does multiply-accumulates (MACs) on every
                    input. Energy = MACs x T x E_MAC.
       ANN sparse:  layer l pays MACs x T x nonzero_density(input) x E_MAC.
                    conv1 sees analog frames: full MAC cost, same as SNN.
       SNN:         layer l pays MACs x T x spike_density(input) x E_AC,
                    accumulates only. conv1 sees analog frames, so it is
                    billed at full MAC cost, which is conservative AGAINST
                    the SNN and dominates its bill.
     Input density for conv2-conv5 and fc1 is measured AFTER the preceding
     MaxPool: max over a 2x2 window of binary spikes is an OR, so the
     pooled map is denser than the LIF rate. fc2's input is LIF6 direct.
     E_MAC = 4.6 pJ, E_AC = 0.9 pJ (Horowitz, ISSCC 2014, 45nm).
  4. Reports NeuroBench-style per-inference counts (Yik et al., 2025):
     Dense synaptic ops (no sparsity), Effective_MACs (MACs whose input
     activation is nonzero), Effective_ACs (accumulates triggered by
     spikes), and ActivationSparsity. The first layer is counted at its
     nonzero-input fraction for BOTH models in this block, which is the
     NeuroBench convention, unlike the conservative pJ table above.

Honesty note for the writeup: this is an accounting model, not a
wall-power measurement. On a GPU both models cost similar watts;
the AC/sparsity savings are only realized on neuromorphic or
event-driven hardware. Say this in the report, the FOI paper does.

Checkpoints: last.pt by default (fixed-budget epoch 64). Layout as in
step 5: seed 0 in /checkpoints and /checkpoints_ann, seed S > 0 in the
seedS subfolders.

Run:    modal run scripts/step7_energy.py --seed 0
Output: printed table + /eval/seed{S}/energy_report.txt and energy.json
        (energy_report_best.txt / energy_best.json for --ckpt best)
Fetch:  modal volume get dvs128-data /eval/seed0/energy.json results/data/seed0/energy.json
        modal volume get dvs128-data /eval/seed0/energy_report.txt results/data/seed0/energy_report.txt

The original single-seed report (/eval/energy_report.txt, best.pt,
3.30 mJ vs 61.90 mJ, 18.7x) is left untouched.
"""

import modal

app = modal.App("dvs128-energy")

vol = modal.Volume.from_name("dvs128-data", create_if_missing=True)
DATA = "/data"
ROOT = f"{DATA}/DVS128Gesture"
CKPT_SNN = f"{DATA}/checkpoints"
CKPT_ANN = f"{DATA}/checkpoints_ann"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.0", "torchvision==0.19.0",
    "spikingjelly==0.0.0.0.14", "numpy<2", "tqdm",
)

E_MAC, E_AC = 4.6e-12, 0.9e-12   # joules, Horowitz 2014, 45 nm

# MACs per frame for each computing layer of this architecture
LAYER_MACS = [
    ("conv1", 2 * 128 * 9 * 128 * 128),      # input: analog frames
    ("conv2", 128 * 128 * 9 * 64 * 64),      # input: pool1 (LIF1 spikes / ReLU1)
    ("conv3", 128 * 128 * 9 * 32 * 32),      # input: pool2
    ("conv4", 128 * 128 * 9 * 16 * 16),      # input: pool3
    ("conv5", 128 * 128 * 9 * 8 * 8),        # input: pool4
    ("fc1",   2048 * 512),                   # input: pool5
    ("fc2",   512 * 110),                    # input: LIF6 / ReLU6
]


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


class Density:
    """Accumulates mean value and nonzero fraction of a hooked tensor."""

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


@app.function(image=image, gpu="A10G", volumes={DATA: vol}, timeout=1800)
def run(T: int = 16, batch: int = 16, seed: int = 0, ckpt: str = "last"):
    import json
    import os

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import functional, layer, neuron
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

    # SNN hooks: every LIF (rate) and every MaxPool (density of the map
    # the next layer reads). Pooled spikes are still 0/1, so mean = density.
    lifs = [m for m in snn.modules() if isinstance(m, neuron.LIFNode)]
    snn_pools = [m for m in snn.modules() if isinstance(m, layer.MaxPool2d)]
    lif_d = [Density() for _ in lifs]
    snn_pool_d = [Density() for _ in snn_pools]
    for m, d in zip(lifs, lif_d):
        m.register_forward_hook(d.hook)
    for m, d in zip(snn_pools, snn_pool_d):
        m.register_forward_hook(d.hook)

    # ANN hooks: every ReLU and every MaxPool, nonzero fraction.
    relus = [m for m in ann.modules() if isinstance(m, nn.ReLU)]
    ann_pools = [m for m in ann.modules() if isinstance(m, nn.MaxPool2d)]
    relu_d = [Density() for _ in relus]
    ann_pool_d = [Density() for _ in ann_pools]
    for m, d in zip(relus, relu_d):
        m.register_forward_hook(d.hook)
    for m, d in zip(ann_pools, ann_pool_d):
        m.register_forward_hook(d.hook)
    assert len(lifs) == 7 and len(relus) == 7 and len(snn_pools) == 5 \
        and len(ann_pools) == 5

    input_d = Density()
    with torch.no_grad():
        for frame, _ in loader:
            frame = frame.to(device).float()
            input_d.hook(None, None, frame)
            snn(frame.transpose(0, 1))
            functional.reset_net(snn)
            B, T_ = frame.shape[0], frame.shape[1]
            ann(frame.reshape(B * T_, *frame.shape[2:]))

    lif_rates = [d.mean for d in lif_d]
    snn_pool_rates = [d.mean for d in snn_pool_d]
    ann_pool_nz = [d.nonzero for d in ann_pool_d]
    relu_nz = [d.nonzero for d in relu_d]

    # input density seen by each layer, layer order of LAYER_MACS.
    # conv1: analog frames (None = full MAC cost in the pJ table).
    snn_in = [None] + snn_pool_rates + [lif_rates[5]]
    ann_in = [None] + ann_pool_nz + [relu_nz[5]]

    rows = []
    tot = {"snn": 0.0, "ann_dense": 0.0, "ann_sparse": 0.0}
    for (name, macs), rs, ra in zip(LAYER_MACS, snn_in, ann_in):
        dense = macs * T * E_MAC
        if rs is None:
            snn_e, ann_sparse_e = dense, dense           # analog input, full cost
        else:
            snn_e = macs * T * rs * E_AC
            ann_sparse_e = macs * T * ra * E_MAC
        tot["snn"] += snn_e
        tot["ann_dense"] += dense
        tot["ann_sparse"] += ann_sparse_e
        rows.append(dict(layer=name, macs_per_frame=macs,
                         snn_in_density=rs, ann_in_nonzero=ra,
                         snn_uJ=snn_e * 1e6, ann_dense_uJ=dense * 1e6,
                         ann_sparse_uJ=ann_sparse_e * 1e6))

    # NeuroBench-style per-inference counts. First layer at its nonzero
    # input fraction for both models (NeuroBench convention).
    dense_ops = sum(macs for _, macs in LAYER_MACS) * T
    in_nz = input_d.nonzero
    snn_eff_macs = LAYER_MACS[0][1] * T * in_nz
    snn_eff_acs = sum(macs * T * r for (_, macs), r in zip(LAYER_MACS[1:], snn_in[1:]))
    ann_eff_macs = LAYER_MACS[0][1] * T * in_nz + \
        sum(macs * T * r for (_, macs), r in zip(LAYER_MACS[1:], ann_in[1:]))
    snn_act_sparsity = 1 - sum(d.value_sum for d in lif_d) / sum(d.count for d in lif_d)
    ann_act_sparsity = 1 - sum(d.nonzero_sum for d in relu_d) / sum(d.count for d in relu_d)

    lines = [f"seed {seed}, {ckpt}.pt (SNN epoch {snn_epoch}, ANN epoch {ann_epoch}), "
             f"T={T}, n={len(ds)}",
             f"{'layer':<7} {'MACs/frame':>14} {'SNN in':>8} {'ANN in':>8} "
             f"{'SNN uJ':>9} {'ANN dense':>10} {'ANN sparse':>11}"]
    for r in rows:
        rs = "analog" if r["snn_in_density"] is None else f"{r['snn_in_density']:.4f}"
        ra = "analog" if r["ann_in_nonzero"] is None else f"{r['ann_in_nonzero']:.4f}"
        lines.append(f"{r['layer']:<7} {r['macs_per_frame']:>14,} {rs:>8} {ra:>8} "
                     f"{r['snn_uJ']:>9.3f} {r['ann_dense_uJ']:>10.3f} "
                     f"{r['ann_sparse_uJ']:>11.3f}")
    lines.append("-" * 72)
    lines.append(f"{'TOTAL':<7} {'':>14} {'':>8} {'':>8} {tot['snn'] * 1e6:>9.3f} "
                 f"{tot['ann_dense'] * 1e6:>10.3f} {tot['ann_sparse'] * 1e6:>11.3f}")
    lines.append(f"\nANN dense / SNN:  {tot['ann_dense'] / tot['snn']:.1f}x saving "
                 f"(the usual literature accounting)")
    lines.append(f"ANN sparse / SNN: {tot['ann_sparse'] / tot['snn']:.1f}x saving "
                 f"(zero-skipping ANN, same treatment for both)")
    snn_l2 = sum(r["snn_uJ"] for r in rows[1:])
    ann_l2 = sum(r["ann_dense_uJ"] for r in rows[1:])
    lines.append(f"spiking layers only (conv1 excluded from both): "
                 f"{snn_l2:.1f} uJ vs {ann_l2:.1f} uJ dense, {ann_l2 / snn_l2:.0f}x")
    lines.append(f"conv1 share of the SNN bill: {rows[0]['snn_uJ'] / (tot['snn'] * 1e6):.1%}")
    lines.append("mean LIF firing rates, layer order: "
                 + " ".join(f"{r:.3f}" for r in lif_rates))
    lines.append("post-pool spike densities, pool order: "
                 + " ".join(f"{r:.3f}" for r in snn_pool_rates))
    lines.append("ANN ReLU nonzero fractions, layer order: "
                 + " ".join(f"{r:.3f}" for r in relu_nz))
    lines.append("ANN post-pool nonzero fractions, pool order: "
                 + " ".join(f"{r:.3f}" for r in ann_pool_nz))
    lines.append(f"activation sparsity: SNN {snn_act_sparsity:.1%} of neuron-timesteps "
                 f"silent, ANN {ann_act_sparsity:.1%} of ReLU outputs zero "
                 f"(both weighted per neuron-timestep)")
    lines.append(f"input frame: nonzero fraction {in_nz:.4f}, mean count {input_d.mean:.4f}")
    lines.append("\nNeuroBench-style counts per inference (first layer at nonzero-input "
                 "fraction for both):")
    lines.append(f"  Dense synaptic ops   {dense_ops:,.0f}")
    lines.append(f"  SNN Effective_MACs   {snn_eff_macs:,.0f}   Effective_ACs {snn_eff_acs:,.0f}")
    lines.append(f"  ANN Effective_MACs   {ann_eff_macs:,.0f}   Effective_ACs 0")
    lines.append(f"note: pJ table bills conv1 at full MAC cost for both models "
                 f"(conservative); E_MAC={E_MAC * 1e12}pJ E_AC={E_AC * 1e12}pJ "
                 f"(Horowitz 2014, 45nm); accounting model, not measured power")

    report = "\n".join(lines)
    print(report)

    summary = dict(
        seed=seed, ckpt=ckpt, snn_epoch=snn_epoch, ann_epoch=ann_epoch, T=T,
        n=len(ds), E_MAC_pJ=E_MAC * 1e12, E_AC_pJ=E_AC * 1e12,
        layers=rows,
        snn_mJ=tot["snn"] * 1e3, ann_dense_mJ=tot["ann_dense"] * 1e3,
        ann_sparse_mJ=tot["ann_sparse"] * 1e3,
        ratio_dense=tot["ann_dense"] / tot["snn"],
        ratio_sparse=tot["ann_sparse"] / tot["snn"],
        spiking_layers_ratio=ann_l2 / snn_l2,
        conv1_share_snn=rows[0]["snn_uJ"] / (tot["snn"] * 1e6),
        lif_rates=lif_rates, snn_pool_densities=snn_pool_rates,
        ann_relu_nonzero=relu_nz, ann_pool_nonzero=ann_pool_nz,
        snn_activation_sparsity=snn_act_sparsity,
        ann_activation_sparsity=ann_act_sparsity,
        input_nonzero=in_nz, input_mean_count=input_d.mean,
        neurobench=dict(dense_ops=dense_ops, snn_effective_macs=snn_eff_macs,
                        snn_effective_acs=snn_eff_acs,
                        ann_effective_macs=ann_eff_macs, ann_effective_acs=0.0),
    )
    os.makedirs(eval_dir(seed), exist_ok=True)
    suffix = "" if ckpt == "last" else f"_{ckpt}"
    with open(f"{eval_dir(seed)}/energy_report{suffix}.txt", "w") as f:
        f.write(report + "\n")
    with open(f"{eval_dir(seed)}/energy{suffix}.json", "w") as f:
        json.dump(summary, f, indent=1)
    vol.commit()
    print(f"\nwrote /eval/seed{seed}/energy_report{suffix}.txt and energy{suffix}.json")


@app.local_entrypoint()
def main(seed: int = 0, ckpt: str = "last"):
    run.remote(seed=seed, ckpt=ckpt)
