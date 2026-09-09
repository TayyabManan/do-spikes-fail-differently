"""Step 10 (CPU): aggregate the matched pair over training seeds.

Reads, for every results/data/seed{S}/ that has a test_outputs.npz:
  test_outputs.npz          clean outputs (step 5, last.pt)
  corruption_outputs.npz    step 6 (optional but expected)
  energy.json               step 7 (optional)
  step3_metrics.csv, step4_metrics.csv   training curves (optional)

Reports three layers of uncertainty, kept separate on purpose:
  1. per seed: the bootstrap-over-samples CIs and McNemar test of step 5,
     so every seed stands on its own;
  2. across seeds: mean, SD and range of each point estimate, and a
     t-interval on the paired SNN - ANN gap (n = number of seeds);
  3. hierarchical bootstrap for the corruption paired differences:
     resample seeds with replacement, then test samples with replacement
     (the same sample indices for both models, keeping the pairing), and
     take the mean over the resampled seeds. This folds seed variance and
     sample variance into one interval. With few seeds it is coarse. Say so.

Usage (from the repo root):
    python scripts/step10_multiseed.py            # all results/data/seed*/
    python scripts/step10_multiseed.py --legacy   # dry run on the original
                                                  # single-seed best.pt dumps
Writes results/data/multiseed_summary.json, results/data/multiseed_summary.md,
results/figures/corruption_curves_multiseed.png and
results/figures/accuracy_by_seed.png.
"""

import csv
import glob
import json
import os
import re
import sys
from math import comb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N_BINS = 15
N_BOOT = 3000
N_HBOOT = 3000
RNG = np.random.default_rng(0)
CORRS = ["drop", "noise", "occlude", "tshuffle"]
T975 = {1: float("nan"), 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


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


def mcnemar_exact(b, c):
    k, tot = min(b, c), b + c
    if tot == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(tot, i) for i in range(k + 1)) / 2 ** tot)


def read_metrics(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0] if rows[0][k] != ""}


def load_seed(d, legacy=False):
    """Everything this script needs from one seed directory."""
    if legacy:
        t = np.load("results/data/test_outputs.npz")
        c_path = "results/data/corruption_outputs.npz"
        m3, m4 = "results/data/step3_metrics.csv", "results/data/step4_metrics.csv"
        e_path, seed, ckpt = "results/data/energy.json", 0, "best (legacy)"
    else:
        t = np.load(os.path.join(d, "test_outputs.npz"))
        c_path = os.path.join(d, "corruption_outputs.npz")
        m3, m4 = os.path.join(d, "step3_metrics.csv"), os.path.join(d, "step4_metrics.csv")
        e_path = os.path.join(d, "energy.json")
        seed = int(t["seed"]) if "seed" in t.files else int(re.findall(r"seed(\d+)", d)[0])
        ckpt = str(t["ckpt"]) if "ckpt" in t.files else "?"
        if seed == 0 and not os.path.exists(m3):        # seed 0 csvs may still sit at the top level
            m3, m4 = "results/data/step3_metrics.csv", "results/data/step4_metrics.csv"
    rec = dict(seed=seed, ckpt=ckpt, labels=t["labels"],
               out={"snn": t["snn_out"], "ann": t["ann_out"]},
               corr=np.load(c_path) if os.path.exists(c_path) else None,
               energy=json.load(open(e_path)) if os.path.exists(e_path) else None,
               metrics={"snn": read_metrics(m3), "ann": read_metrics(m4)})
    return rec


def per_seed_stats(rec):
    labels = rec["labels"]
    n = len(labels)
    st = dict(seed=rec["seed"], ckpt=rec["ckpt"], n=n)
    corr = {}
    for m in ("snn", "ann"):
        conf, pred = confidences(rec["out"][m])
        cr = (pred == labels).astype(float)
        corr[m] = cr
        st[f"{m}_acc"] = float(cr.mean())
        st[f"{m}_acc_ci"] = boot_ci(lambda i, cr=cr: cr[i].mean(), n)
        st[f"{m}_ece"] = float(ece(conf, cr))
        st[f"{m}_ece_ci"] = boot_ci(lambda i, conf=conf, cr=cr: ece(conf[i], cr[i]), n)
        st[f"{m}_mean_conf"] = float(conf.mean())
    s, a = corr["snn"], corr["ann"]
    st["gap"] = float(a.mean() - s.mean())                      # ANN - SNN
    st["gap_ci"] = boot_ci(lambda i: a[i].mean() - s[i].mean(), n)
    b, c = int(((s == 1) & (a == 0)).sum()), int(((s == 0) & (a == 1)).sum())
    st["discordant"] = [b, c]
    st["mcnemar_p"] = mcnemar_exact(b, c)
    conf_s, _ = confidences(rec["out"]["snn"])
    conf_a, _ = confidences(rec["out"]["ann"])
    st["ece_diff_ci"] = boot_ci(lambda i: ece(conf_a[i], a[i]) - ece(conf_s[i], s[i]), n)
    # training-curve facts
    for m in ("snn", "ann"):
        mt = rec["metrics"][m]
        if mt:
            st[f"{m}_last_epoch_acc"] = mt["test_acc"][-1]
            st[f"{m}_best_epoch_acc"] = max(mt["test_acc"])
            st[f"{m}_final_train_acc"] = mt["train_acc"][-1]
            if "epoch_time_s" in mt:
                st[f"{m}_epoch_time_s"] = float(np.mean(mt["epoch_time_s"]))
    if rec["energy"]:
        e = rec["energy"]
        st["energy"] = {k: e[k] for k in ("snn_mJ", "ann_dense_mJ", "ann_sparse_mJ",
                                          "ratio_dense", "ratio_sparse",
                                          "snn_activation_sparsity",
                                          "ann_activation_sparsity") if k in e}
    # corruption conditions
    if rec["corr"] is not None:
        d = rec["corr"]
        st["sev_labels"] = {x.split(":")[0]: x.split(":")[1].split(",") for x in d["severities"]}
        st["corr"] = {}
        for cname in CORRS:
            for sev in range(5):
                cond = {}
                for m in ("snn", "ann"):
                    key = f"{m}_clean_0" if sev == 0 else f"{m}_{cname}_{sev}"
                    conf, pred = confidences(d[key])
                    cr = (pred == labels).astype(float)
                    cond[m] = dict(acc=float(cr.mean()), mconf=float(conf.mean()),
                                   ece=float(ece(conf, cr)), corr=cr)
                st["corr"][(cname, sev)] = cond
    return st


def summarize(vals):
    v = np.asarray(vals, dtype=float)
    k = len(v)
    out = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if k > 1 else float("nan"),
               min=float(v.min()), max=float(v.max()), n_seeds=k)
    if k > 1:
        h = T975.get(k, 1.96) * out["sd"] / np.sqrt(k)
        out["t_ci"] = [out["mean"] - h, out["mean"] + h]
    return out


def hier_boot(corr_by_seed, reps=N_HBOOT):
    """corr_by_seed: list over seeds of (snn_correct, ann_correct) arrays.
    Resample seeds, then samples (same indices for both models)."""
    S, n = len(corr_by_seed), len(corr_by_seed[0][0])
    vals = []
    for _ in range(reps):
        seeds = RNG.integers(0, S, S)
        idx = RNG.integers(0, n, n)
        vals.append(np.mean([corr_by_seed[s][0][idx].mean() - corr_by_seed[s][1][idx].mean()
                             for s in seeds]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fmt_ci(ci, sign=False):
    f = "{:+.3f}" if sign else "{:.3f}"
    return f"[{f.format(ci[0])}, {f.format(ci[1])}]"


def main(argv):
    legacy = "--legacy" in argv
    if legacy:
        recs = [load_seed(None, legacy=True)]
        print("DRY RUN on the original single-seed best.pt dumps "
              "(results/data/*.npz). Not for the paper.\n")
    else:
        dirs = sorted(d for d in glob.glob("results/data/seed*")
                      if os.path.exists(os.path.join(d, "test_outputs.npz")))
        if not dirs:
            sys.exit("no results/data/seed*/test_outputs.npz found "
                     "(fetch the step 5 dumps first, or use --legacy for a dry run)")
        recs = [load_seed(d) for d in dirs]
    stats = [per_seed_stats(r) for r in recs]
    S = len(stats)
    n = stats[0]["n"]
    ckpts = sorted({s["ckpt"] for s in stats})
    md = [f"# Matched pair over {S} seed{'s' if S != 1 else ''} "
          f"(checkpoint: {', '.join(ckpts)}; n = {n} test samples)", ""]

    # ---------------- per-seed table ----------------
    md += ["## Per seed (bootstrap 95% CI over test samples)", "",
           "| seed | SNN acc | ANN acc | ANN - SNN gap | McNemar p (SNN-only right / ANN-only right) | SNN ECE | ANN ECE | ANN - SNN ECE CI |",
           "|---|---|---|---|---|---|---|---|"]
    for s in stats:
        md.append(f"| {s['seed']} | {s['snn_acc']:.4f} {fmt_ci(s['snn_acc_ci'])} "
                  f"| {s['ann_acc']:.4f} {fmt_ci(s['ann_acc_ci'])} "
                  f"| {s['gap']:+.4f} {fmt_ci(s['gap_ci'], True)} "
                  f"| {s['mcnemar_p']:.3f} ({s['discordant'][0]} / {s['discordant'][1]}) "
                  f"| {s['snn_ece']:.4f} {fmt_ci(s['snn_ece_ci'])} "
                  f"| {s['ann_ece']:.4f} {fmt_ci(s['ann_ece_ci'])} "
                  f"| {fmt_ci(s['ece_diff_ci'], True)} |")
    md.append("")

    # ---------------- across seeds ----------------
    agg = {}
    for key, label in (("snn_acc", "SNN accuracy"), ("ann_acc", "ANN accuracy"),
                       ("gap", "ANN - SNN accuracy gap"), ("snn_ece", "SNN ECE"),
                       ("ann_ece", "ANN ECE"), ("snn_mean_conf", "SNN mean confidence"),
                       ("ann_mean_conf", "ANN mean confidence")):
        agg[key] = summarize([s[key] for s in stats])
        agg[key]["label"] = label
    md += ["## Across seeds (mean, SD, range; t-interval where n_seeds > 1)", "",
           "| quantity | mean | SD | min | max | 95% t-CI |", "|---|---|---|---|---|---|"]
    for key, a in agg.items():
        ci = fmt_ci(a["t_ci"]) if "t_ci" in a else "n/a"
        md.append(f"| {a['label']} | {a['mean']:.4f} | {a['sd']:.4f} | {a['min']:.4f} "
                  f"| {a['max']:.4f} | {ci} |")
    sig = sum(1 for s in stats if s["mcnemar_p"] < 0.05)
    md += ["", f"McNemar p < 0.05 in {sig} of {S} seeds. ECE difference CI excludes zero in "
           f"{sum(1 for s in stats if s['ece_diff_ci'][0] > 0 or s['ece_diff_ci'][1] < 0)} of {S}.", ""]

    # ---------------- checkpoint selection effect ----------------
    if all(f"{m}_last_epoch_acc" in s for s in stats for m in ("snn", "ann")):
        md += ["## Test-set checkpoint selection (from metrics.csv)", "",
               "| seed | SNN last | SNN best | SNN inflation | ANN last | ANN best | ANN inflation | epoch s (SNN / ANN) |",
               "|---|---|---|---|---|---|---|---|"]
        for s in stats:
            et = (f"{s.get('snn_epoch_time_s', float('nan')):.0f} / "
                  f"{s.get('ann_epoch_time_s', float('nan')):.0f}")
            md.append(f"| {s['seed']} | {s['snn_last_epoch_acc']:.4f} | {s['snn_best_epoch_acc']:.4f} "
                      f"| {s['snn_best_epoch_acc'] - s['snn_last_epoch_acc']:+.4f} "
                      f"| {s['ann_last_epoch_acc']:.4f} | {s['ann_best_epoch_acc']:.4f} "
                      f"| {s['ann_best_epoch_acc'] - s['ann_last_epoch_acc']:+.4f} | {et} |")
        md.append("")

    # ---------------- energy ----------------
    if all(s.get("energy") for s in stats):
        md += ["## Energy accounting (per seed, last.pt)", "",
               "| seed | SNN mJ | ANN dense mJ | ANN sparse mJ | dense ratio | sparse ratio | SNN silent | ANN zero |",
               "|---|---|---|---|---|---|---|---|"]
        for s in stats:
            e = s["energy"]
            md.append(f"| {s['seed']} | {e['snn_mJ']:.2f} | {e['ann_dense_mJ']:.2f} | {e['ann_sparse_mJ']:.2f} "
                      f"| {e['ratio_dense']:.1f}x | {e['ratio_sparse']:.1f}x "
                      f"| {e['snn_activation_sparsity']:.1%} | {e['ann_activation_sparsity']:.1%} |")
        for key in ("snn_mJ", "ann_dense_mJ", "ann_sparse_mJ", "ratio_dense", "ratio_sparse"):
            agg[f"energy_{key}"] = summarize([s["energy"][key] for s in stats])
            agg[f"energy_{key}"]["label"] = f"energy {key}"
        md.append("")

    # ---------------- corruption ----------------
    have_corr = all("corr" in s for s in stats)
    corr_summary = {}
    if have_corr:
        sev_labels = stats[0]["sev_labels"]
        md += ["## Corruption: paired SNN - ANN accuracy difference", "",
               "Per condition: mean over seeds of the paired difference, hierarchical "
               "bootstrap 95% CI (seeds, then samples), and how many seeds have a "
               "per-seed sample-bootstrap CI excluding zero.", ""]
        for cname in CORRS:
            md += [f"### {cname}", "",
                   "| severity | SNN acc mean (range) | ANN acc mean (range) | SNN - ANN | hier. boot CI | seeds sig. | SNN conf-acc | ANN conf-acc |",
                   "|---|---|---|---|---|---|---|---|"]
            corr_summary[cname] = []
            for sev in range(5):
                sa = [s["corr"][(cname, sev)]["snn"]["acc"] for s in stats]
                aa = [s["corr"][(cname, sev)]["ann"]["acc"] for s in stats]
                diffs = [x - y for x, y in zip(sa, aa)]
                pairs = [(s["corr"][(cname, sev)]["snn"]["corr"],
                          s["corr"][(cname, sev)]["ann"]["corr"]) for s in stats]
                hci = hier_boot(pairs)
                nsig = 0
                for sc, ac in pairs:
                    lo, hi = boot_ci(lambda i, sc=sc, ac=ac: sc[i].mean() - ac[i].mean(), n)
                    nsig += int(lo > 0 or hi < 0)
                gs = float(np.mean([s["corr"][(cname, sev)]["snn"]["mconf"]
                                    - s["corr"][(cname, sev)]["snn"]["acc"] for s in stats]))
                ga = float(np.mean([s["corr"][(cname, sev)]["ann"]["mconf"]
                                    - s["corr"][(cname, sev)]["ann"]["acc"] for s in stats]))
                rec = dict(sev="clean" if sev == 0 else sev_labels[cname][sev - 1],
                           snn_acc=summarize(sa), ann_acc=summarize(aa),
                           diff_mean=float(np.mean(diffs)), diff_hier_ci=list(hci),
                           seeds_significant=nsig, snn_gap=gs, ann_gap=ga,
                           snn_ece=summarize([s["corr"][(cname, sev)]["snn"]["ece"] for s in stats]),
                           ann_ece=summarize([s["corr"][(cname, sev)]["ann"]["ece"] for s in stats]))
                corr_summary[cname].append(rec)
                star = "*" if (hci[0] > 0 or hci[1] < 0) else ""
                md.append(f"| {rec['sev']} | {rec['snn_acc']['mean']:.3f} ({rec['snn_acc']['min']:.3f}-{rec['snn_acc']['max']:.3f}) "
                          f"| {rec['ann_acc']['mean']:.3f} ({rec['ann_acc']['min']:.3f}-{rec['ann_acc']['max']:.3f}) "
                          f"| {rec['diff_mean']:+.3f} | {fmt_ci(hci, True)}{star} | {nsig}/{S} "
                          f"| {gs:+.3f} | {ga:+.3f} |")
            md.append("")
        md.append("(* = hierarchical bootstrap CI excludes zero)")
        md.append("")

    # ---------------- console + files ----------------
    text = "\n".join(md)
    print(text)
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    tag = "_legacy" if legacy else ""
    with open(f"results/data/multiseed_summary{tag}.md", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    out = dict(n_seeds=S, seeds=[s["seed"] for s in stats], ckpts=ckpts, n=n,
               per_seed=[{k: v for k, v in s.items() if k not in ("corr", "sev_labels")}
                         for s in stats],
               across_seeds=agg, corruption=corr_summary)
    with open(f"results/data/multiseed_summary{tag}.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nwrote results/data/multiseed_summary{tag}.md and .json")

    # figure 1: accuracy by seed
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    xs = np.arange(S)
    for m, off, col in (("snn", -0.12, "tab:blue"), ("ann", 0.12, "tab:orange")):
        v = [s[f"{m}_acc"] for s in stats]
        lo = [s[f"{m}_acc"] - s[f"{m}_acc_ci"][0] for s in stats]
        hi = [s[f"{m}_acc_ci"][1] - s[f"{m}_acc"] for s in stats]
        ax.errorbar(xs + off, v, yerr=[lo, hi], fmt="o", capsize=3, color=col,
                    label=f"{m.upper()} (mean {np.mean(v):.3f})")
        ax.axhline(np.mean(v), color=col, lw=1, ls=":", alpha=0.7)
    ax.set_xticks(xs, [f"seed {s['seed']}" for s in stats])
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.8, 1.0)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"Clean accuracy per seed, {', '.join(ckpts)} (bars = bootstrap 95% CI, n={n})",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(f"results/figures/accuracy_by_seed{tag}.png", dpi=150)
    print(f"wrote results/figures/accuracy_by_seed{tag}.png")

    # figure 2: corruption curves with seed spread + paired difference
    if have_corr:
        fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex="col")
        colors = {"snn": "tab:blue", "ann": "tab:orange"}
        x = np.arange(5)
        for j, cname in enumerate(CORRS):
            for m in ("snn", "ann"):
                per_seed = np.array([[s["corr"][(cname, sev)][m]["acc"] for sev in range(5)]
                                     for s in stats])
                for row in per_seed:
                    axes[0, j].plot(x, row, color=colors[m], lw=0.8, alpha=0.35)
                axes[0, j].plot(x, per_seed.mean(0), color=colors[m], lw=2.2, marker="o",
                                label=f"{m.upper()} mean of {S}")
            recs = corr_summary[cname]
            dm = [r["diff_mean"] for r in recs]
            lo = [r["diff_mean"] - r["diff_hier_ci"][0] for r in recs]
            hi = [r["diff_hier_ci"][1] - r["diff_mean"] for r in recs]
            axes[1, j].errorbar(x, dm, yerr=[lo, hi], marker="o", capsize=3, color="tab:green")
            axes[1, j].axhline(0, color="gray", lw=1, ls="--")
            axes[0, j].set_title(cname)
            axes[0, j].set_ylim(0.3, 1.0)
            axes[1, j].set_ylim(-0.2, 0.45)
            axes[1, j].set_xticks(x, ["clean"] + sev_labels[cname])
            axes[1, j].set_xlabel("severity")
        axes[0, 0].set_ylabel("accuracy (thin = seeds)")
        axes[1, 0].set_ylabel("SNN - ANN, paired (hier. boot 95% CI)")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"SNN vs ANN under corruption over {S} seed{'s' if S != 1 else ''}, "
                     f"{', '.join(ckpts)}, DVS128Gesture test (n={n})")
        fig.tight_layout()
        fig.savefig(f"results/figures/corruption_curves_multiseed{tag}.png", dpi=150)
        print(f"wrote results/figures/corruption_curves_multiseed{tag}.png")


if __name__ == "__main__":
    main(sys.argv[1:])
