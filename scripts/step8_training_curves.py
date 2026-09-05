"""Step 8 figure: training curves of the matched pair, from metrics.csv.

Input: results/data/step3_metrics.csv (SNN), step4_metrics.csv (ANN),
one row per epoch: epoch,train_loss,train_acc,test_loss,test_acc.

Usage (from the repo root):
    python scripts/step8_training_curves.py
Writes results/figures/training_curves_dvs.png.
(step2's surrogate figure keeps the name training_curves.png.)
"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILES = {
    "snn": "results/data/step3_metrics.csv",
    "ann": "results/data/step4_metrics.csv",
}
COLORS = {"snn": "tab:blue", "ann": "tab:orange"}


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = {k: [float(r[k]) for r in rows] for k in rows[0]}
    return cols


def main():
    data = {m: load(p) for m, p in FILES.items()}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for m, d in data.items():
        c = COLORS[m]
        axes[0].plot(d["epoch"], d["test_acc"], color=c, lw=2,
                     label=f"{m.upper()} test")
        axes[0].plot(d["epoch"], d["train_acc"], color=c, lw=1, ls="--",
                     alpha=0.6, label=f"{m.upper()} train")
        axes[1].plot(d["epoch"], d["test_loss"], color=c, lw=2,
                     label=f"{m.upper()} test")
        axes[1].plot(d["epoch"], d["train_loss"], color=c, lw=1, ls="--",
                     alpha=0.6, label=f"{m.upper()} train")

    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0.4, 1.02)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("MSE loss")
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Training curves, matched pair on DVS128Gesture "
                 "(identical schedule, seed 0, 64 epochs)")
    fig.tight_layout()
    fig.savefig("results/figures/training_curves_dvs.png", dpi=150)
    print("wrote results/figures/training_curves_dvs.png")

    for m, d in data.items():
        print(f"{m}: final train_acc {d['train_acc'][-1]:.4f}, "
              f"final test_acc {d['test_acc'][-1]:.4f}, "
              f"best test_acc {max(d['test_acc']):.4f}")


if __name__ == "__main__":
    main()
