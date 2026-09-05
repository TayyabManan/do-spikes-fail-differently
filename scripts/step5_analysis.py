"""Step 5: reliability analysis of the SNN vs ANN twins. Runs on CPU.

Input: test_outputs.npz from step5_dump_logits.py (labels, snn_out, ann_out).

Confidence mapping, stated for the writeup: both models were trained
with MSE against one-hot targets, so their outputs are already
probability-like scores (row sums ~0.93-0.99 here). Confidence is the
normalized score of the predicted class: conf = max(out) / sum(out).
Softmax would be wrong here: these are rates in [0,1.3], not logits,
and softmax on that range squashes every confidence toward 1/11,
manufacturing fake underconfidence.

Reports, each with bootstrap 95% CIs (n=288 makes point estimates
untrustworthy on their own):
  1. Accuracy per model, and the PAIRED gap with McNemar exact test
  2. ECE, 15 equal-width bins
  3. Mean confidence vs accuracy (direction of miscalibration)
  4. Reliability diagram, both models side by side -> reliability.png

Usage (from the repo root):
    python scripts/step5_analysis.py results/data/test_outputs.npz
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N_BINS = 15
N_BOOT = 5000
RNG = np.random.default_rng(0)


def confidences(out):
    s = out.sum(1, keepdims=True)
    s[s == 0] = 1e-9
    p = out / s
    return p.max(1), out.argmax(1)


def ece(conf, correct, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def boot_ci(stat_fn, n, reps=N_BOOT):
    """Percentile CI. stat_fn takes an index array over the test set."""
    vals = [stat_fn(RNG.integers(0, n, n)) for _ in range(reps)]
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def main(path):
    d = np.load(path)
    labels = d["labels"]
    n = len(labels)
    models = {}
    for name in ["snn", "ann"]:
        conf, pred = confidences(d[f"{name}_out"])
        models[name] = dict(conf=conf, correct=(pred == labels).astype(float))

    print(f"N = {n}, bins = {N_BINS}, bootstrap reps = {N_BOOT}\n")

    for name, m in models.items():
        acc = m["correct"].mean()
        alo, ahi = boot_ci(lambda i: m["correct"][i].mean(), n)
        e = ece(m["conf"], m["correct"])
        elo, ehi = boot_ci(lambda i: ece(m["conf"][i], m["correct"][i]), n)
        mc = m["conf"].mean()
        print(f"{name.upper()}:")
        print(f"  accuracy   {acc:.4f}  [{alo:.4f}, {ahi:.4f}]")
        print(f"  ECE        {e:.4f}  [{elo:.4f}, {ehi:.4f}]")
        print(f"  mean conf  {mc:.4f}  ({'over' if mc > acc else 'under'}"
              f"confident by {abs(mc - acc):.4f})\n")

    # paired accuracy comparison
    s, a = models["snn"]["correct"], models["ann"]["correct"]
    gap = a.mean() - s.mean()
    glo, ghi = boot_ci(lambda i: a[i].mean() - s[i].mean(), n)
    b = int(((s == 1) & (a == 0)).sum())   # SNN right, ANN wrong
    c = int(((s == 0) & (a == 1)).sum())   # ANN right, SNN wrong
    from math import comb
    k, tot = min(b, c), b + c
    p_mcnemar = 2 * sum(comb(tot, i) for i in range(k + 1)) / 2**tot
    p_mcnemar = min(1.0, p_mcnemar)
    print(f"PAIRED: ANN - SNN accuracy gap {gap:+.4f}  [{glo:+.4f}, {ghi:+.4f}]")
    print(f"  discordant: SNN-only-right {b}, ANN-only-right {c}, "
          f"McNemar exact p = {p_mcnemar:.4f}")

    # ECE difference, paired
    dlo, dhi = boot_ci(
        lambda i: ece(models["ann"]["conf"][i], a[i])
        - ece(models["snn"]["conf"][i], s[i]), n)
    print(f"  ANN - SNN ECE difference CI: [{dlo:+.4f}, {dhi:+.4f}]"
          f"  ({'excludes' if dlo > 0 or dhi < 0 else 'includes'} zero)")

    # reliability diagram
    edges = np.linspace(0, 1, N_BINS + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (name, m) in zip(axes, models.items()):
        accs, cnts = [], []
        for i in range(N_BINS):
            msk = (m["conf"] > edges[i]) & (m["conf"] <= edges[i + 1])
            accs.append(m["correct"][msk].mean() if msk.sum() else np.nan)
            cnts.append(int(msk.sum()))
        ax.plot([0, 1], [0, 1], "--", c="gray", lw=1, label="perfect")
        ax.bar(mids, accs, width=1 / N_BINS * 0.9, alpha=0.7,
               label="observed accuracy")
        for x, cnt, acc_v in zip(mids, cnts, accs):
            if cnt:
                ax.text(x, 0.02, str(cnt), ha="center", fontsize=7)
        e = ece(m["conf"], m["correct"])
        ax.set_title(f"{name.upper()}  (ECE = {e:.3f})")
        ax.set_xlabel("confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="upper left")
    fig.suptitle("Reliability diagrams, DVS128Gesture test set "
                 f"(n={n}; bin counts shown at bar base)")
    fig.tight_layout()
    fig.savefig("results/figures/reliability.png", dpi=150)
    print("\nwrote results/figures/reliability.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/data/test_outputs.npz")
