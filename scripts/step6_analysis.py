"""Step 6, part 2 (CPU): accuracy and calibration vs corruption severity.

Usage (from the repo root):
    python scripts/step6_analysis.py results/data/corruption_outputs.npz
Writes results/figures/corruption_curves.png and prints the key
paired comparisons.
Confidence mapping matches step 5: normalized scores, not softmax.
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N_BINS = 15
N_BOOT = 3000
RNG = np.random.default_rng(0)
CORRS = ["drop", "noise", "occlude", "tshuffle"]


def confidences(out):
    s = out.sum(1, keepdims=True)
    s[s == 0] = 1e-9
    return (out / s).max(1), out.argmax(1)


def ece(conf, correct, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def boot_ci(fn, n, reps=N_BOOT):
    vals = [fn(RNG.integers(0, n, n)) for _ in range(reps)]
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def main(path):
    d = np.load(path)
    labels = d["labels"]
    n = len(labels)
    sev_labels = {s.split(":")[0]: ["clean"] + s.split(":")[1].split(",")
                  for s in d["severities"]}

    stats = {}  # (model, corr, sev_idx) -> dict
    for m in ("snn", "ann"):
        for c in CORRS:
            for s in range(5):
                key = f"{m}_clean_0" if s == 0 else f"{m}_{c}_{s}"
                conf, pred = confidences(d[key])
                corr_arr = (pred == labels).astype(float)
                acc = corr_arr.mean()
                alo, ahi = boot_ci(lambda i: corr_arr[i].mean(), n)
                e = ece(conf, corr_arr)
                elo, ehi = boot_ci(lambda i: ece(conf[i], corr_arr[i]), n)
                stats[(m, c, s)] = dict(acc=acc, alo=alo, ahi=ahi,
                                        ece=e, elo=elo, ehi=ehi,
                                        conf=conf, corr=corr_arr,
                                        mconf=conf.mean())

    # figure: 2 rows (accuracy, ECE) x 4 corruptions
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex="col")
    colors = {"snn": "tab:blue", "ann": "tab:orange"}
    x = np.arange(5)
    for j, c in enumerate(CORRS):
        for m in ("snn", "ann"):
            st = [stats[(m, c, s)] for s in range(5)]
            for row, (lo_k, hi_k, v_k) in enumerate(
                    [("alo", "ahi", "acc"), ("elo", "ehi", "ece")]):
                v = [t[v_k] for t in st]
                lo = [t[v_k] - t[lo_k] for t in st]
                hi = [t[hi_k] - t[v_k] for t in st]
                axes[row, j].errorbar(x, v, yerr=[lo, hi], marker="o",
                                      capsize=3, label=m.upper(),
                                      color=colors[m])
        axes[0, j].set_title(c)
        axes[1, j].set_xticks(x, sev_labels[c])
        axes[1, j].set_xlabel("severity")
        axes[0, j].set_ylim(0.3, 1.0)
        axes[1, j].set_ylim(0, 0.45)
    axes[0, 0].set_ylabel("accuracy")
    axes[1, 0].set_ylabel("ECE")
    axes[0, 0].legend()
    fig.suptitle("SNN vs ANN under corruption, DVS128Gesture test "
                 f"(n={n}, bars = bootstrap 95% CI)")
    fig.tight_layout()
    fig.savefig("results/figures/corruption_curves.png", dpi=150)
    print("wrote results/figures/corruption_curves.png\n")

    # key paired comparisons: SNN - ANN accuracy at each condition
    print("paired SNN - ANN accuracy difference [95% CI], per condition:")
    for c in CORRS:
        line = f"  {c:<9}"
        for s in range(1, 5):
            sc, ac = stats[("snn", c, s)]["corr"], stats[("ann", c, s)]["corr"]
            diff = sc.mean() - ac.mean()
            lo, hi = boot_ci(lambda i: sc[i].mean() - ac[i].mean(), n)
            star = "*" if (lo > 0 or hi < 0) else " "
            line += f"  {diff:+.3f} [{lo:+.3f},{hi:+.3f}]{star}"
        print(line)
    print("  (* = CI excludes zero)")

    # does confidence track failure? mean confidence vs accuracy per severity
    print("\nmean confidence minus accuracy (positive = overconfident):")
    for c in CORRS:
        for m in ("snn", "ann"):
            gaps = [stats[(m, c, s)]["mconf"] - stats[(m, c, s)]["acc"]
                    for s in range(5)]
            print(f"  {m} {c:<9} " + " ".join(f"{g:+.3f}" for g in gaps))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "results/data/corruption_outputs.npz")
