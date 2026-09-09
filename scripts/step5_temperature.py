"""Step 5, part 2 (CPU): temperature scaling as the calibration baseline.

Why. The clean calibration tie (step 5) and the confidence-tracking
result under noise (step 6) are measured on the models' native
confidence, the normalized score max(out)/sum(out). A reviewer will ask
what a standard post-hoc fix does to both models. This script answers
with temperature scaling (Guo et al., 2017): p = softmax(out / T), T
fitted by NLL. The softmax objection in step 5 is about T = 1 on rate
outputs; with T fitted (it comes out well below 1) this is ordinary
temperature scaling.

No validation split exists for DVS128Gesture and both models sit at
~100% train accuracy, so T is fitted on the test set by two-fold
CROSS-FITTING: split the 288 samples into two stratified halves, fit T
on one half, apply it to the other, swap. Every reported confidence
comes from a temperature fitted on samples it was not part of. The
bootstrap re-runs the whole procedure (resample, then cross-fit) with
the fold assignment fixed per original index, so a resampled duplicate
never lands on both sides of a fit.

Under corruption the temperatures fitted on CLEAN data are applied,
which is the deployment situation (Ovadia et al., 2019: post-hoc
calibration on in-distribution data does not survive shift). An
"oracle" column also refits T on the corrupted outputs themselves, the
best any single temperature could do if the shift were known.

Inputs: test_outputs.npz and corruption_outputs.npz for one seed.
Usage (from the repo root):
    python scripts/step5_temperature.py results/data/seed0/test_outputs.npz \
        results/data/seed0/corruption_outputs.npz
Writes results/figures/temperature_scaling.png (or the third argument)
and temperature_scaling.json next to the test_outputs file.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N_BINS = 15
N_BOOT_CLEAN = 2000
N_BOOT_CORR = 1000
RNG = np.random.default_rng(0)
CORRS = ["drop", "noise", "occlude", "tshuffle"]
LOG_T_LO, LOG_T_HI = np.log(1e-3), np.log(1e2)   # search bracket for T


def norm_conf(out):
    s = out.sum(1, keepdims=True)
    s[s == 0] = 1e-9
    return (out / s).max(1)


def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def ts_conf(out, temps):
    """max softmax(out / T) with a per-sample temperature vector."""
    return np.exp(log_softmax(out / temps[:, None])).max(1)


def nll_at(out, labels, log_t):
    logp = log_softmax(out / np.exp(log_t))
    return -logp[np.arange(len(labels)), labels].mean()


def fit_temperature(out, labels, iters=60):
    """Golden-section search on log T. The NLL of softmax(out / T) is
    convex in 1/T, so the bracketed minimum is the global one."""
    g = (np.sqrt(5) - 1) / 2
    a, b = LOG_T_LO, LOG_T_HI
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll_at(out, labels, c), nll_at(out, labels, d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll_at(out, labels, c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll_at(out, labels, d)
    return float(np.exp((a + b) / 2))


def ece(conf, correct, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def stratified_folds(labels, rng):
    """0/1 fold id per sample, balanced within every class."""
    folds = np.empty(len(labels), dtype=int)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % 2
    return folds


def crossfit(out, labels, folds):
    """Per-sample temperature fitted on the OTHER fold. Returns (temps, fitted)."""
    temps = np.empty(len(labels))
    fitted = {}
    for f in (0, 1):
        fitted[f] = fit_temperature(out[folds != f], labels[folds != f])
        temps[folds == f] = fitted[f]
    return temps, fitted


def boot(fn, n, reps):
    vals = [fn(RNG.integers(0, n, n)) for _ in range(reps)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main(test_path, corr_path, fig_path=None):
    t = np.load(test_path)
    labels = t["labels"]
    n = len(labels)
    seed = int(t["seed"]) if "seed" in t.files else None
    ckpt = str(t["ckpt"]) if "ckpt" in t.files else "?"
    folds = stratified_folds(labels, np.random.default_rng(0))
    outs = {"snn": t["snn_out"], "ann": t["ann_out"]}

    print(f"seed {seed}, {ckpt}.pt, n = {n}, two-fold cross-fitted temperature "
          f"scaling, {N_BOOT_CLEAN} clean / {N_BOOT_CORR} corruption bootstrap reps\n")

    summary = dict(seed=seed, ckpt=ckpt, n=n, clean={}, corruption={})
    temps = {}
    for m in ("snn", "ann"):
        out = outs[m]
        correct = (out.argmax(1) == labels).astype(float)
        temps[m], fitted = crossfit(out, labels, folds)
        c_norm, c_ts = norm_conf(out), ts_conf(out, temps[m])
        e_norm, e_ts = ece(c_norm, correct), ece(c_ts, correct)

        def boot_ece_ts(idx, out=out, correct=correct):
            tt, _ = crossfit(out[idx], labels[idx], folds[idx])
            return ece(ts_conf(out[idx], tt), correct[idx])

        lo_n, hi_n = boot(lambda i: ece(c_norm[i], correct[i]), n, N_BOOT_CLEAN)
        lo_t, hi_t = boot(boot_ece_ts, n, N_BOOT_CLEAN)
        acc = correct.mean()
        summary["clean"][m] = dict(
            acc=float(acc), fitted_T=fitted,
            norm=dict(ece=float(e_norm), ece_ci=[lo_n, hi_n],
                      mean_conf=float(c_norm.mean()), gap=float(c_norm.mean() - acc)),
            ts=dict(ece=float(e_ts), ece_ci=[lo_t, hi_t],
                    mean_conf=float(c_ts.mean()), gap=float(c_ts.mean() - acc)))
        print(f"{m.upper()} clean: acc {acc:.4f}, fitted T fold0 {fitted[0]:.3f} "
              f"fold1 {fitted[1]:.3f}")
        print(f"  normalized score: ECE {e_norm:.4f} [{lo_n:.4f}, {hi_n:.4f}], "
              f"conf - acc {c_norm.mean() - acc:+.4f}")
        print(f"  temp. scaling:    ECE {e_ts:.4f} [{lo_t:.4f}, {hi_t:.4f}], "
              f"conf - acc {c_ts.mean() - acc:+.4f}")
    # paired ECE difference after TS, ANN - SNN
    def paired_ts(idx):
        vals = []
        for m in ("snn", "ann"):
            out = outs[m]
            correct = (out.argmax(1) == labels).astype(float)
            tt, _ = crossfit(out[idx], labels[idx], folds[idx])
            vals.append(ece(ts_conf(out[idx], tt), correct[idx]))
        return vals[1] - vals[0]
    lo, hi = boot(paired_ts, n, N_BOOT_CLEAN // 2)
    summary["clean"]["ann_minus_snn_ece_ts_ci"] = [lo, hi]
    print(f"\nANN - SNN ECE difference after TS: [{lo:+.4f}, {hi:+.4f}] "
          f"({'excludes' if lo > 0 or hi < 0 else 'includes'} zero)\n")

    # ---- corruption: clean-fitted temperatures applied to shifted outputs ----
    c = np.load(corr_path)
    assert np.array_equal(c["labels"], labels), "label order differs between dumps"
    sev_labels = {s.split(":")[0]: s.split(":")[1].split(",") for s in c["severities"]}
    # bootstrap temperatures once per resample, reuse across conditions
    boot_idx = [RNG.integers(0, n, n) for _ in range(N_BOOT_CORR)]
    boot_temps = {m: [crossfit(outs[m][idx], labels[idx], folds[idx])[0]
                      for idx in boot_idx] for m in ("snn", "ann")}

    print("corruption, per condition: acc | conf - acc (norm / TS-clean / TS-oracle) "
          "| ECE (norm / TS-clean [CI] / TS-oracle)")
    for corr in CORRS:
        summary["corruption"][corr] = dict(severities=sev_labels[corr], snn=[], ann=[])
        for m in ("snn", "ann"):
            for s in range(1, 5):
                out = c[f"{m}_{corr}_{s}"]
                correct = (out.argmax(1) == labels).astype(float)
                acc = correct.mean()
                cn = norm_conf(out)
                ct = ts_conf(out, temps[m])
                oracle_t, _ = crossfit(out, labels, folds)
                co = ts_conf(out, oracle_t)
                e_n, e_t, e_o = ece(cn, correct), ece(ct, correct), ece(co, correct)
                vals = [ece(ts_conf(out[idx], bt), correct[idx])
                        for idx, bt in zip(boot_idx, boot_temps[m])]
                # bt is already in resampled order (position j = sample idx[j])
                lo_t, hi_t = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
                lo_n, hi_n = boot(lambda i: ece(cn[i], correct[i]), n, N_BOOT_CORR)
                rec = dict(sev=sev_labels[corr][s - 1], acc=float(acc),
                           norm=dict(ece=float(e_n), ece_ci=[lo_n, hi_n],
                                     gap=float(cn.mean() - acc)),
                           ts_clean=dict(ece=float(e_t), ece_ci=[lo_t, hi_t],
                                         gap=float(ct.mean() - acc)),
                           ts_oracle=dict(ece=float(e_o), gap=float(co.mean() - acc)))
                summary["corruption"][corr][m].append(rec)
                print(f"  {m} {corr:<9}{sev_labels[corr][s - 1]:>5}: acc {acc:.3f} | "
                      f"{cn.mean() - acc:+.3f} / {ct.mean() - acc:+.3f} / "
                      f"{co.mean() - acc:+.3f} | {e_n:.3f} / {e_t:.3f} "
                      f"[{lo_t:.3f},{hi_t:.3f}] / {e_o:.3f}")
        print()

    # ---- figure: conf - acc and ECE vs severity, norm vs TS-clean ----
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex="col")
    colors = {"snn": "tab:blue", "ann": "tab:orange"}
    x = np.arange(5)
    for j, corr in enumerate(CORRS):
        for m in ("snn", "ann"):
            cl = summary["clean"][m]
            recs = summary["corruption"][corr][m]
            for style, key, label in (("-", "norm", "normalized score"),
                                      ("--", "ts_clean", "temp. scaling (clean fit)")):
                gaps = [cl[key if key == "norm" else "ts"]["gap"]] + [r[key]["gap"] for r in recs]
                eces = [cl[key if key == "norm" else "ts"]["ece"]] + [r[key]["ece"] for r in recs]
                axes[0, j].plot(x, gaps, style, marker="o", color=colors[m],
                                label=f"{m.upper()}, {label}")
                axes[1, j].plot(x, eces, style, marker="o", color=colors[m])
        axes[0, j].axhline(0, color="gray", lw=1, ls=":")
        axes[0, j].set_title(corr)
        axes[1, j].set_xticks(x, ["clean"] + sev_labels[corr])
        axes[1, j].set_xlabel("severity")
        axes[0, j].set_ylim(-0.15, 0.6)
        axes[1, j].set_ylim(0, 0.6)
    axes[0, 0].set_ylabel("mean confidence - accuracy")
    axes[1, 0].set_ylabel("ECE")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Temperature scaling fitted on clean data, applied under corruption "
                 f"(seed {seed}, {ckpt}.pt, n={n}, two-fold cross-fit)")
    fig.tight_layout()
    fig_path = fig_path or "results/figures/temperature_scaling.png"
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"wrote {fig_path}")

    out_json = os.path.join(os.path.dirname(test_path) or ".", "temperature_scaling.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0] if len(args) > 0 else "results/data/seed0/test_outputs.npz",
         args[1] if len(args) > 1 else "results/data/seed0/corruption_outputs.npz",
         args[2] if len(args) > 2 else None)
