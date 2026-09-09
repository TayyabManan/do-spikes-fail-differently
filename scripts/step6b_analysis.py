"""Step 6b, part 2 (CPU): does the SNN's noise advantage depend on the
temporal correlation of the noise?

Input: noise_sweep_outputs.npz from step6b_noise_sweep.py.
Usage (from the repo root):
    python scripts/step6b_analysis.py results/data/seed0/noise_sweep_outputs.npz
Writes results/figures/noise_correlation_sweep.png (or the --out name)
and a JSON summary next to the input npz.

Confidence mapping matches step 5: normalized scores, not softmax.

The test of the leak hypothesis is the paired SNN - ANN accuracy
difference as a function of k (frames a noise field persists). A
shrinking difference from k=1 to k=16 supports "the LIF filters
temporally uncorrelated input". The script reports, per lam:
  diff(k) with bootstrap 95% CIs, and
  diff(k=1) - diff(k=16), the shrinkage, with its bootstrap CI.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N_BINS = 15
N_BOOT = 3000
RNG = np.random.default_rng(0)


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
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main(path, fig_path=None):
    d = np.load(path)
    labels = d["labels"]
    n = len(labels)
    lams = [float(x) for x in d["lams"]]
    ks = [int(x) for x in d["ks"]]
    seed = int(d["seed"]) if "seed" in d.files else None
    ckpt = str(d["ckpt"]) if "ckpt" in d.files else "?"

    def cond(m, i, k):
        return d["%s_clean" % m] if i is None else d["%s_lam%d_k%d" % (m, i, k)]

    stats = {}
    for m in ("snn", "ann"):
        for i in [None] + list(range(1, len(lams) + 1)):
            for k in ([None] if i is None else ks):
                conf, pred = confidences(cond(m, i, k))
                corr = (pred == labels).astype(float)
                stats[(m, i, k)] = dict(
                    acc=float(corr.mean()), corr=corr, conf=conf,
                    mconf=float(conf.mean()), ece=float(ece(conf, corr)))

    print(f"seed {seed}, {ckpt}.pt, n = {n}, bootstrap reps = {N_BOOT}")
    print(f"clean: SNN {stats[('snn', None, None)]['acc']:.4f}  "
          f"ANN {stats[('ann', None, None)]['acc']:.4f}\n")

    summary = dict(seed=seed, ckpt=ckpt, n=n, lams=lams, ks=ks, per_lam=[])
    header = "k=" + "  ".join(f"{k:>16d}" for k in ks)
    for i, lam in enumerate(lams, start=1):
        print(f"lam = {lam}")
        print(f"  {'':<22}{header}")
        row = dict(lam=lam, k=ks, snn_acc=[], ann_acc=[], diff=[], diff_ci=[],
                   snn_gap=[], ann_gap=[], snn_ece=[], ann_ece=[])
        for m in ("snn", "ann"):
            accs = [stats[(m, i, k)]["acc"] for k in ks]
            row[f"{m}_acc"] = accs
            row[f"{m}_gap"] = [stats[(m, i, k)]["mconf"] - stats[(m, i, k)]["acc"]
                               for k in ks]
            row[f"{m}_ece"] = [stats[(m, i, k)]["ece"] for k in ks]
            print(f"  {m + ' accuracy':<22}" + "  ".join(f"{a:>16.4f}" for a in accs))
        diffs, cis = [], []
        for k in ks:
            sc, ac = stats[("snn", i, k)]["corr"], stats[("ann", i, k)]["corr"]
            diffs.append(float(sc.mean() - ac.mean()))
            cis.append(boot_ci(lambda idx: sc[idx].mean() - ac[idx].mean(), n))
        row["diff"], row["diff_ci"] = diffs, cis
        print(f"  {'SNN - ANN (paired)':<22}" + "  ".join(
            f"{dv:+.3f} [{lo:+.3f},{hi:+.3f}]" for dv, (lo, hi) in zip(diffs, cis)))
        for m in ("snn", "ann"):
            print(f"  {m + ' conf - acc':<22}" + "  ".join(
                f"{g:>16.3f}" for g in row[f"{m}_gap"]))
        # shrinkage of the paired advantage from k=1 to k=16
        s1, a1 = stats[("snn", i, ks[0])]["corr"], stats[("ann", i, ks[0])]["corr"]
        sK, aK = stats[("snn", i, ks[-1])]["corr"], stats[("ann", i, ks[-1])]["corr"]
        shrink = float((s1.mean() - a1.mean()) - (sK.mean() - aK.mean()))
        lo, hi = boot_ci(lambda idx: (s1[idx].mean() - a1[idx].mean())
                         - (sK[idx].mean() - aK[idx].mean()), n)
        star = "*" if (lo > 0 or hi < 0) else ""
        row["shrink_k1_minus_kmax"] = dict(value=shrink, ci=[lo, hi])
        print(f"  advantage shrinkage, diff(k={ks[0]}) - diff(k={ks[-1]}): "
              f"{shrink:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}\n")
        summary["per_lam"].append(row)
    print("(* = CI excludes zero. Positive shrinkage = the SNN advantage is "
          "smaller for persistent noise, as the leak hypothesis predicts.)")

    # figure: 3 rows (accuracy, paired diff, conf - acc) x len(lams) columns
    fig, axes = plt.subplots(3, len(lams), figsize=(3.6 * len(lams), 8.5),
                             sharex=True)
    colors = {"snn": "tab:blue", "ann": "tab:orange"}
    x = np.arange(len(ks))
    for j, (lam, row) in enumerate(zip(lams, summary["per_lam"])):
        for m in ("snn", "ann"):
            axes[0, j].plot(x, row[f"{m}_acc"], marker="o", color=colors[m],
                            label=m.upper())
            axes[2, j].plot(x, row[f"{m}_gap"], marker="o", color=colors[m],
                            label=m.upper())
        lo = [dv - c[0] for dv, c in zip(row["diff"], row["diff_ci"])]
        hi = [c[1] - dv for dv, c in zip(row["diff"], row["diff_ci"])]
        axes[1, j].errorbar(x, row["diff"], yerr=[lo, hi], marker="o",
                            capsize=3, color="tab:green")
        axes[1, j].axhline(0, color="gray", lw=1, ls="--")
        axes[2, j].axhline(0, color="gray", lw=1, ls="--")
        axes[0, j].set_title(f"noise rate lam = {lam}")
        axes[0, j].set_ylim(0, 1.0)
        axes[1, j].set_ylim(-0.2, 0.5)
        axes[2, j].set_ylim(-0.15, 0.6)
        axes[2, j].set_xticks(x, [str(k) for k in ks])
        axes[2, j].set_xlabel("k = frames a noise field persists")
    axes[0, 0].set_ylabel("accuracy")
    axes[1, 0].set_ylabel("SNN - ANN accuracy (paired)")
    axes[2, 0].set_ylabel("mean conf - acc")
    axes[0, 0].legend(loc="lower left")
    fig.suptitle("Noise temporal-correlation sweep, DVS128Gesture test "
                 f"(seed {seed}, {ckpt}.pt, n={n}; k=1 is step 6's noise, "
                 "k=16 a static field; bars = bootstrap 95% CI)")
    fig.tight_layout()
    fig_path = fig_path or "results/figures/noise_correlation_sweep.png"
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"\nwrote {fig_path}")

    out_json = os.path.join(os.path.dirname(path) or ".", "noise_sweep_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "results/data/seed0/noise_sweep_outputs.npz",
         sys.argv[2] if len(sys.argv) > 2 else None)
