"""Step 7 (GPU, ~2 min): firing rates and the energy accounting table.

What this does:
  1. Runs the test set through the SNN with hooks on every LIF layer,
     measuring mean firing rate (spikes per neuron per timestep).
  2. Computes the standard energy accounting used across the SNN
     literature (and the FOI paper):
       ANN: every layer does multiply-accumulates (MACs) on every
            input. Energy = MACs x T x E_MAC.
       SNN: a layer only computes when a presynaptic spike arrives,
            and spikes trigger accumulates (ACs), not multiplies.
            Energy per layer = MACs x T x spike_density(input) x E_AC.
            Input density for conv2-conv5 and fc1 is measured AFTER
            the preceding MaxPool: max over a 2x2 window of binary
            spikes is an OR, so the pooled map is denser than the LIF
            rate. (An earlier version billed the pre-pool LIF rate,
            which understated SNN energy.) fc2's input is LIF6 direct.
            Layer 1 sees analog frames, so it is counted at full MAC
            cost, which is conservative AGAINST the SNN.
     E_MAC = 4.6 pJ, E_AC = 0.9 pJ (Horowitz, ISSCC 2014, 45nm).

Honesty note for the writeup: this is an accounting model, not a
wall-power measurement. On a GPU both models cost similar watts;
the AC/sparsity savings are only realized on neuromorphic or
event-driven hardware. Say this in the report, the FOI paper does.

Run:    modal run scripts/step7_energy.py
Output: printed table + /eval/energy_report.txt on the volume.
"""

import modal

app = modal.App("dvs128-energy")

vol = modal.Volume.from_name("dvs128-data", create_if_missing=True)
DATA = "/data"
ROOT = f"{DATA}/DVS128Gesture"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.0", "torchvision==0.19.0",
    "spikingjelly==0.0.0.0.14", "numpy<2", "tqdm",
)

E_MAC, E_AC = 4.6e-12, 0.9e-12   # joules, Horowitz 2014, 45 nm

# MACs per frame for each computing layer of this architecture
LAYER_MACS = [
    ("conv1", 2 * 128 * 9 * 128 * 128),      # input: analog frames
    ("conv2", 128 * 128 * 9 * 64 * 64),      # input: LIF1 spikes
    ("conv3", 128 * 128 * 9 * 32 * 32),      # input: LIF2 spikes
    ("conv4", 128 * 128 * 9 * 16 * 16),      # input: LIF3 spikes
    ("conv5", 128 * 128 * 9 * 8 * 8),        # input: LIF4 spikes
    ("fc1",   2048 * 512),                   # input: LIF5 spikes
    ("fc2",   512 * 110),                    # input: LIF6 spikes
]


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


@app.function(image=image, gpu="A10G", volumes={DATA: vol}, timeout=1800)
def run(T: int = 16, batch: int = 16):
    import os

    import torch
    from spikingjelly.activation_based import functional, layer, neuron
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
    from torch.utils.data import DataLoader

    device = "cuda"
    ds = DVS128Gesture(ROOT, train=False, data_type="frame",
                       frames_number=T, split_by="number")
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4)

    net = build_snn().to(device)
    functional.set_step_mode(net, "m")
    net.load_state_dict(torch.load(f"{DATA}/checkpoints/best.pt",
                                   map_location=device)["net"])
    net.eval()

    # hooks: record mean firing rate of every LIF, and spike density
    # after every MaxPool (the actual input to the next conv, and to
    # fc1 after the last pool). Pooled spikes are still 0/1, so the
    # mean of the pooled map IS its density.
    lifs = [m for m in net.modules() if isinstance(m, neuron.LIFNode)]
    pools = [m for m in net.modules() if isinstance(m, layer.MaxPool2d)]
    lif_sums, lif_counts = [0.0] * len(lifs), [0] * len(lifs)
    pool_sums, pool_counts = [0.0] * len(pools), [0] * len(pools)

    def make_hook(sums, counts, i):
        def hook(module, inp, out):
            sums[i] += out.mean().item() * out.numel()
            counts[i] += out.numel()
        return hook

    for i, m in enumerate(lifs):
        m.register_forward_hook(make_hook(lif_sums, lif_counts, i))
    for i, m in enumerate(pools):
        m.register_forward_hook(make_hook(pool_sums, pool_counts, i))

    mean_frame = 0.0
    n_frames = 0
    with torch.no_grad():
        for frame, _ in loader:
            frame = frame.to(device).float()
            mean_frame += frame.mean().item() * frame.numel()
            n_frames += frame.numel()
            net(frame.transpose(0, 1))
            functional.reset_net(net)

    rates = [s / c for s, c in zip(lif_sums, lif_counts)]
    pool_rates = [s / c for s, c in zip(pool_sums, pool_counts)]
    input_density = (mean_frame / n_frames)

    # energy per sample
    lines = []
    lines.append(f"{'layer':<7} {'MACs/frame':>14} {'input rate':>11} "
                 f"{'SNN uJ':>9} {'ANN uJ':>9}")
    snn_total = ann_total = 0.0
    # input rate per layer: conv1 analog; conv2-conv5 and fc1 see the
    # post-pool spike map; fc2 sees LIF6 directly (no pool between).
    in_rates = [None] + pool_rates + [rates[5]]
    for (name, macs), r in zip(LAYER_MACS, in_rates):
        ann_e = macs * T * E_MAC
        if r is None:
            snn_e = macs * T * E_MAC       # analog input, full MAC cost
            r_str = "analog"
        else:
            snn_e = macs * T * r * E_AC
            r_str = f"{r:.4f}"
        snn_total += snn_e
        ann_total += ann_e
        lines.append(f"{name:<7} {macs:>14,} {r_str:>11} "
                     f"{snn_e * 1e6:>9.3f} {ann_e * 1e6:>9.3f}")

    lines.append("-" * 55)
    lines.append(f"{'TOTAL':<7} {'':>14} {'':>11} "
                 f"{snn_total * 1e6:>9.3f} {ann_total * 1e6:>9.3f}")
    lines.append(f"\nSNN / ANN energy ratio: {snn_total / ann_total:.3f} "
                 f"({ann_total / snn_total:.1f}x saving)")
    lines.append(f"mean LIF firing rates, layer order: "
                 + " ".join(f"{r:.3f}" for r in rates))
    lines.append(f"post-pool spike densities, pool order: "
                 + " ".join(f"{r:.3f}" for r in pool_rates))
    weighted_rate = sum(lif_sums) / sum(lif_counts)
    lines.append(f"overall spike sparsity: {1 - weighted_rate:.1%} of "
                 f"neuron-timesteps are silent (per neuron-timestep, weighted)")
    lines.append(f"input frame density (nonzero mass per cell): "
                 f"{input_density:.4f}")
    lines.append(f"note: conv1 counted at full MAC cost for the SNN "
                 f"(conservative); E_MAC={E_MAC*1e12}pJ E_AC={E_AC*1e12}pJ "
                 f"(Horowitz 2014, 45nm)")

    report = "\n".join(lines)
    print(report)
    os.makedirs(f"{DATA}/eval", exist_ok=True)
    with open(f"{DATA}/eval/energy_report.txt", "w") as f:
        f.write(report + "\n")
    vol.commit()
    print("\nwrote /eval/energy_report.txt")


@app.local_entrypoint()
def main():
    run.remote()
