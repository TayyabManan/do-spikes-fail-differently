"""Step 12, analysis (CPU): does the noise result scale with the LIF leak?

Reads, per tau in TAUS and seed in SEEDS:
  tau 2.0 (baseline)   results/data/seed{S}/            (steps 5, 6, 6b, 7)
  other tau            results/data/tau/tau{tau}/seed{S}/  (step 12 eval)
each holding test_outputs.npz, corruption_outputs.npz, noise_sweep_outputs.npz
and energy.json. Missing taus are skipped with a note.

Reports, per tau, mean over seeds with a hierarchical bootstrap (seeds, then
samples) on every paired quantity:
  clean SNN accuracy and ECE, the ANN - SNN gap
  SNN accuracy under full temporal shuffle minus its own clean accuracy
  paired SNN - ANN accuracy under i.i.d. noise (sweep k=1) per rate
  the same under a static noise field (k=16), and the shrinkage
  both noise differences relative to each model's own clean accuracy,
    (SNN noise - SNN clean) - (ANN noise - ANN clean), which removes the
    clean-accuracy offset that varies with tau
  SNN confidence minus accuracy under static noise
  SNN firing rate and the dense energy ratio
The leak hypothesis predicts monotone trends in tau for the noise rows:
advantage at k=1 up, advantage at k=16 down, shrinkage up.

Usage (from the repo root):
    python scripts/step12_tau_analysis.py
Writes results/data/tau/tau_ablation_summary.md and .json, and
results/figures/tau_ablation.png.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TAUS = [1.1, 2.0, 4.0, 8.0]
SEEDS = [0, 1, 2]
N_BINS = 15
N_HBOOT = 3000
RNG = np.random.default_rng(0)


def tau_dir(tau, seed):
    if tau == 2.0:
        return f"results/data/seed{seed}"
    return f"results/data/tau/tau{tau:g}/seed{seed}"


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


def hier(pairs, reps=N_HBOOT):
    """pairs: per seed (a, b) arrays; statistic = mean over seeds of mean(a - b).
    Resample seeds, then samples (same indices for a and b)."""
    S, n = len(pairs), len(pairs[0][0])
    vals = []
    for _ in range(reps):
        sr = RNG.integers(0, S, S)
        idx = RNG.integers(0, n, n)
        vals.append(np.mean([pairs[j][0][idx].mean() - pairs[j][1][idx].mean() for j in sr]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load(tau, seed):
    d = tau_dir(tau, seed)
    if not os.path.exists(os.path.join(d, "noise_sweep_outputs.npz")):
        return None
    t = np.load(os.path.join(d, "test_outputs.npz"))
    c = np.load(os.path.join(d, "corruption_outputs.npz"))
    w = np.load(os.path.join(d, "noise_sweep_outputs.npz"))
    e_path = os.path.join(d, "energy.json")
    e = json.load(open(e_path)) if os.path.exists(e_path) else None
    labels = t["labels"]

    def corr(out):
        return (out.argmax(1) == labels).astype(float)

    conf_s, _ = confidences(t["snn_out"])
    conf_a, _ = confidences(t["ann_out"])
    rec = dict(
        tau=tau, seed=seed, labels=labels,
        snn_clean=corr(t["snn_out"]), ann_clean=corr(t["ann_out"]),
        snn_ece=float(ece(conf_s, corr(t["snn_out"]))), ann_ece=float(ece(conf_a, corr(t["ann_out"]))),
        snn_shuffle16=corr(c["snn_tshuffle_4"]), snn_clean_c=corr(c["snn_clean_0"]),
        lams=[float(x) for x in w["lams"]], ks=[int(x) for x in w["ks"]],
        sweep={}, energy=e,
        sweep_snn_clean=corr(w["snn_clean"]), sweep_ann_clean=corr(w["ann_clean"]))
    for i in range(1, len(rec["lams"]) + 1):
        for k in rec["ks"]:
            cs, ps = confidences(w[f"snn_lam{i}_k{k}"])
            ca, pa = confidences(w[f"ann_lam{i}_k{k}"])
            rec["sweep"][(i, k)] = dict(snn=(ps == labels).astype(float), ann=(pa == labels).astype(float),
                                        snn_gap=float(cs.mean() - (ps == labels).mean()),
                                        ann_gap=float(ca.mean() - (pa == labels).mean()))
    return rec


def main():
    recs = {tau: [r for r in (load(tau, s) for s in SEEDS) if r is not None] for tau in TAUS}
    have = [tau for tau in TAUS if recs[tau]]
    if not have:
        sys.exit("no tau results found under results/data/tau/ or results/data/seed*/")
    for tau in TAUS:
        if not recs[tau]:
            print(f"tau {tau:g}: no results yet, skipped")
    lams, ks = recs[have[0]][0]["lams"], recs[have[0]][0]["ks"]
    k1, kK = ks[0], ks[-1]

    rows = {}
    md = ["# LIF time-constant ablation on DVS128Gesture (SNN side; ANN twins unchanged)", "",
          "v <- v + (x - v) / tau. tau 2.0 is the baseline from steps 3-7. Mean over seeds, "
          "hierarchical bootstrap 95% CI (seeds, then samples). * = CI excludes zero.", ""]
    md += ["## Clean", "", "| tau | seeds | SNN acc (range) | ANN - SNN gap | SNN ECE | ANN ECE | "
           "SNN shuffle(16) - clean | SNN silent | dense energy ratio |", "|---|---|---|---|---|---|---|---|---|"]
    for tau in have:
        rs = recs[tau]
        S = len(rs)
        sa = [r["snn_clean"].mean() for r in rs]
        gap = hier([(r["ann_clean"], r["snn_clean"]) for r in rs])
        gap_m = float(np.mean([r["ann_clean"].mean() - r["snn_clean"].mean() for r in rs]))
        sh = hier([(r["snn_shuffle16"], r["snn_clean_c"]) for r in rs])
        sh_m = float(np.mean([r["snn_shuffle16"].mean() - r["snn_clean_c"].mean() for r in rs]))
        sil = [r["energy"]["snn_activation_sparsity"] for r in rs if r["energy"]]
        ratio = [r["energy"]["ratio_dense"] for r in rs if r["energy"]]
        row = dict(tau=tau, n_seeds=S, snn_acc=float(np.mean(sa)), snn_acc_min=float(min(sa)),
                   snn_acc_max=float(max(sa)), gap=gap_m, gap_ci=list(gap),
                   snn_ece=float(np.mean([r["snn_ece"] for r in rs])),
                   ann_ece=float(np.mean([r["ann_ece"] for r in rs])),
                   shuffle=sh_m, shuffle_ci=list(sh),
                   silent=float(np.mean(sil)) if sil else None,
                   ratio_dense=float(np.mean(ratio)) if ratio else None, sweep={})
        star_g = "*" if (gap[0] > 0 or gap[1] < 0) else ""
        star_s = "*" if (sh[0] > 0 or sh[1] < 0) else ""
        sil_s = "" if row["silent"] is None else f"{row['silent']:.1%}"
        rat_s = "" if row["ratio_dense"] is None else f"{row['ratio_dense']:.1f}x"
        md.append(f"| {tau:g} | {S} | {row['snn_acc']:.3f} ({row['snn_acc_min']:.3f}-{row['snn_acc_max']:.3f}) "
                  f"| {gap_m:+.3f} [{gap[0]:+.3f}, {gap[1]:+.3f}]{star_g} | {row['snn_ece']:.3f} | {row['ann_ece']:.3f} "
                  f"| {sh_m:+.3f} [{sh[0]:+.3f}, {sh[1]:+.3f}]{star_s} | {sil_s} | {rat_s} |")
        rows[tau] = row

    md += ["", "## Noise: paired SNN - ANN accuracy, i.i.d. field (k=1) vs static field (k=16)", ""]
    for i, lam in enumerate(lams, start=1):
        md += [f"### lam = {lam}", "",
               "| tau | diff k=1 | CI | diff k=16 | CI | shrinkage | CI | SNN conf-acc k=16 | ANN conf-acc k=16 |",
               "|---|---|---|---|---|---|---|---|---|"]
        for tau in have:
            rs = recs[tau]
            p1 = [(r["sweep"][(i, k1)]["snn"], r["sweep"][(i, k1)]["ann"]) for r in rs]
            pK = [(r["sweep"][(i, kK)]["snn"], r["sweep"][(i, kK)]["ann"]) for r in rs]
            d1, dK = hier(p1), hier(pK)
            d1_m = float(np.mean([a.mean() - b.mean() for a, b in p1]))
            dK_m = float(np.mean([a.mean() - b.mean() for a, b in pK]))
            # shrinkage: hierarchical over seeds and samples of diff(k1) - diff(kK)
            S, n = len(rs), len(rs[0]["labels"])
            vals = []
            for _ in range(N_HBOOT):
                sr = RNG.integers(0, S, S)
                idx = RNG.integers(0, n, n)
                vals.append(np.mean([(p1[j][0][idx].mean() - p1[j][1][idx].mean())
                                     - (pK[j][0][idx].mean() - pK[j][1][idx].mean()) for j in sr]))
            shr = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
            shr_m = d1_m - dK_m
            gs = float(np.mean([r["sweep"][(i, kK)]["snn_gap"] for r in rs]))
            ga = float(np.mean([r["sweep"][(i, kK)]["ann_gap"] for r in rs]))
            c_s = [r["sweep_snn_clean"] for r in rs]
            c_a = [r["sweep_ann_clean"] for r in rs]
            rel1 = [(p1[j][0] - c_s[j], p1[j][1] - c_a[j]) for j in range(S)]
            relK = [(pK[j][0] - c_s[j], pK[j][1] - c_a[j]) for j in range(S)]
            r1, rK = hier(rel1), hier(relK)
            r1_m = float(np.mean([a.mean() - b.mean() for a, b in rel1]))
            rK_m = float(np.mean([a.mean() - b.mean() for a, b in relK]))
            rows[tau]["sweep"][str(lam)] = dict(diff_k1=d1_m, diff_k1_ci=list(d1), diff_kK=dK_m,
                                                diff_kK_ci=list(dK), shrink=shr_m, shrink_ci=list(shr),
                                                snn_gap_kK=gs, ann_gap_kK=ga,
                                                rel_k1=r1_m, rel_k1_ci=list(r1),
                                                rel_kK=rK_m, rel_kK_ci=list(rK))
            st = lambda ci: "*" if (ci[0] > 0 or ci[1] < 0) else ""
            md.append(f"| {tau:g} | {d1_m:+.3f} | [{d1[0]:+.3f}, {d1[1]:+.3f}]{st(d1)} | {dK_m:+.3f} "
                      f"| [{dK[0]:+.3f}, {dK[1]:+.3f}]{st(dK)} | {shr_m:+.3f} | [{shr[0]:+.3f}, {shr[1]:+.3f}]{st(shr)} "
                      f"| {gs:+.3f} | {ga:+.3f} |")
        md.append("")

    md += ["## Noise-induced change relative to own clean accuracy", "",
           "(SNN noise - SNN clean) - (ANN noise - ANN clean). Positive = the SNN loses less "
           "accuracy than the ANN. This removes the clean-accuracy offset, which varies with tau. "
           "The shrinkage column above is already offset-free.", ""]
    for lam in lams:
        md += [f"### lam = {lam}", "", f"| tau | k=1 | CI | k={kK} | CI |", "|---|---|---|---|---|"]
        for tau in have:
            q = rows[tau]["sweep"][str(lam)]
            s1 = "*" if (q["rel_k1_ci"][0] > 0 or q["rel_k1_ci"][1] < 0) else ""
            sK = "*" if (q["rel_kK_ci"][0] > 0 or q["rel_kK_ci"][1] < 0) else ""
            md.append(f"| {tau:g} | {q['rel_k1']:+.3f} | [{q['rel_k1_ci'][0]:+.3f}, {q['rel_k1_ci'][1]:+.3f}]{s1} "
                      f"| {q['rel_kK']:+.3f} | [{q['rel_kK_ci'][0]:+.3f}, {q['rel_kK_ci'][1]:+.3f}]{sK} |")
        md.append("")

    text = "\n".join(md)
    print(text)
    os.makedirs("results/data/tau", exist_ok=True)
    with open("results/data/tau/tau_ablation_summary.md", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open("results/data/tau/tau_ablation_summary.json", "w") as f:
        json.dump({str(t): rows[t] for t in have}, f, indent=1)
    print("\nwrote results/data/tau/tau_ablation_summary.md and .json")

    # ---- figure: six panels against tau ----
    x = np.arange(len(have))
    xt = [f"{t:g}" for t in have]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    lam_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(lams)))

    ax = axes[0, 0]
    for j, tau in enumerate(have):
        for r in recs[tau]:
            ax.plot(j, r["snn_clean"].mean(), "o", color="tab:blue", alpha=0.4, ms=4)
    ax.plot(x, [rows[t]["snn_acc"] for t in have], "-o", color="tab:blue", label="SNN")
    ann_ref = float(np.mean([r["ann_clean"].mean() for t in have for r in recs[t]]))
    ax.axhline(ann_ref, color="tab:orange", ls="--", label=f"ANN twins (mean {ann_ref:.3f})")
    ax.set_title("clean accuracy")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, [rows[t]["shuffle"] for t in have], "-o", color="tab:blue")
    ax.errorbar(x, [rows[t]["shuffle"] for t in have],
                yerr=[[rows[t]["shuffle"] - rows[t]["shuffle_ci"][0] for t in have],
                      [rows[t]["shuffle_ci"][1] - rows[t]["shuffle"] for t in have]],
                fmt="none", ecolor="tab:blue", capsize=3)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_title("SNN: full shuffle minus own clean")

    ax = axes[0, 2]
    ax.plot(x, [rows[t]["silent"] if rows[t]["silent"] is not None else np.nan for t in have],
            "-o", color="tab:blue")
    ax.set_title("SNN activation sparsity (fraction silent)")

    for panel, key, title in ((axes[1, 0], "diff_k1", "SNN - ANN, i.i.d. noise (k=1)"),
                              (axes[1, 1], "diff_kK", f"SNN - ANN, static noise (k={kK})"),
                              (axes[1, 2], "shrink", f"shrinkage diff(k=1) - diff(k={kK})"),
                              (axes[2, 0], "rel_k1", "vs own clean: i.i.d. noise (k=1)"),
                              (axes[2, 1], "rel_kK", f"vs own clean: static noise (k={kK})")):
        for c, lam in zip(lam_colors, lams):
            v = [rows[t]["sweep"][str(lam)][key] for t in have]
            ci = [rows[t]["sweep"][str(lam)][key + "_ci"] for t in have]
            panel.errorbar(x, v, yerr=[[a - b[0] for a, b in zip(v, ci)], [b[1] - a for a, b in zip(v, ci)]],
                           marker="o", capsize=3, color=c, label=f"lam {lam}")
        panel.axhline(0, color="gray", ls="--", lw=1)
        panel.set_title(title)
    axes[1, 0].legend(fontsize=8)
    ax = axes[2, 2]
    for c, lam in zip(lam_colors, lams):
        ax.plot(x, [rows[t]["sweep"][str(lam)]["snn_gap_kK"] for t in have], "-o", color=c,
                label=f"SNN lam {lam}")
        ax.axhline(rows[have[0]]["sweep"][str(lam)]["ann_gap_kK"], color=c, ls=":", lw=1.2)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_title(f"SNN conf - acc, static noise (k={kK}); dotted = ANN")
    for ax in axes.flat:
        ax.set_xticks(x, xt)
        ax.set_xlabel("tau (LIF time constant)")
    fig.suptitle("LIF time-constant ablation, DVS128Gesture, SNN side; ANN twins unchanged "
                 "(bars = hierarchical bootstrap 95% CI over seeds and samples)")
    fig.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    fig.savefig("results/figures/tau_ablation.png", dpi=150)
    print("wrote results/figures/tau_ablation.png")


if __name__ == "__main__":
    main()
