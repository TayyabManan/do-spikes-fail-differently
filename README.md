# Do Spikes Fail Differently?

A matched-pair study of calibration and robustness in event-based vision.
A spiking neural network (SNN) and an artificial neural network (ANN) twin
share architecture, parameter count, input frames, loss, optimizer,
schedule, batch size, epochs, T=16 frames and random seed. Each leaky
integrate-and-fire neuron in the SNN is a ReLU in the ANN. Nothing else
differs. The study measures what spiking costs and what it buys: accuracy,
calibration, robustness under corruption, sensitivity to temporal
structure, and energy under an accounting model.

Two datasets: DVS128 Gesture (5 seeds, 288 test recordings) and CIFAR10-DVS
(3 seeds, 1000 test recordings). Plus a membrane time-constant ablation on
DVS128 Gesture (tau 1.1, 2, 4 and 8, seeds 0 to 2).

Semester project. Miniature of the master's thesis "Do Spikes Fail
Differently? Reliability of Parameter-Efficient Spiking Event-Language
Models for Open-Vocabulary Recognition".

## Paper

The manuscript submitted to NICE 2027 is not in this repository. Every
number in it comes from the three result summaries under `results/data/`
listed below.

## Findings

Means over seeds with last-epoch weights. Each per-seed metric carries a
bootstrap 95% CI over test samples. Paired SNN minus ANN differences carry
a hierarchical bootstrap CI over seeds and samples. Sources:
`results/data/multiseed_summary.md`,
`results/data/c10/multiseed_summary_c10.md` and
`results/data/tau/tau_ablation_summary.md`.

| Axis | DVS128 Gesture, 5 seeds | CIFAR10-DVS, 3 seeds |
|---|---|---|
| Clean accuracy | SNN 0.926, ANN 0.958. Gap 0.032 [+0.016, +0.048]. McNemar significant in 3 of 5 seeds. | SNN 0.724, ANN 0.730. Gap 0.006 [-0.014, +0.026]. Not significant in any seed. |
| Calibration | Both underconfident. ECE 0.047 (SNN) vs 0.039 (ANN). | Both overconfident. ECE 0.046 (SNN) vs 0.078 (ANN). |
| After temperature scaling | Paired ECE difference excludes zero in 0 of 5 seeds. A tie. | 0 of 3 seeds. A tie. |
| Background noise, drawn independently for each frame | SNN leads by +0.094, +0.173 and +0.140 at rates 0.1, 0.2 and 0.5, in every seed. The ANN grows overconfident, +0.230 at rate 0.5; the SNN stays within 0.05 of its accuracy. | SNN trails by 0.031 at rate 0.05 and leads by up to +0.202 at 0.2. At 0.5 the SNN is the more overconfident model, +0.358 vs +0.153. |
| The same noise field held for k frames | The advantage shrinks at high rates. At rate 0.5 it reverses: +0.135 at k=1, -0.211 [-0.317, -0.101] at k=16. | The penalty is positive at every rate. At rate 0.1: +0.047 at k=1, -0.100 [-0.152, -0.043] at k=16. |
| Event drop, occlusion | The ANN keeps its clean lead. | Event drop hurts the SNN more: -0.055, -0.109 and -0.074 at p = 0.4, 0.6 and 0.8. Occlusion moves the difference by less than 0.03 on both datasets. |
| Temporal shuffle, a control | The ANN is invariant by construction. SNN -0.001 [-0.015, +0.012]: no measurable use of frame order. | SNN -0.021 [-0.038, -0.002]: it uses frame order. |
| Energy, accounting model | SNN 3.32 mJ vs dense ANN 61.90 mJ per inference, 18.7x. 14.7x once the ANN is credited for its zero activations. 98.1% of SNN neuron time steps are silent. | 17.1x dense, 13.0x with zero credit. |

Time-constant ablation, DVS128 Gesture, seeds 0 to 2. Clean accuracy falls
from 0.935 at tau 1.1 to 0.846 at tau 8. The advantage at moderate noise
(rate 0.1) is about +0.14 at every tau, so it does not need integration
across frames. At rate 0.5 both the advantage and the persistence penalty
vanish at tau 1.1 (-0.024 and +0.009, neither significant) and appear from
tau 2 on. The penalty peaks at tau 4, so a pure low-pass account does not
fit either.

Checkpoint selection: picking the epoch that scores best on the test set
would have raised accuracy by up to 0.017 on DVS128 Gesture. Everything
above uses the fixed-budget epoch-64 checkpoint.

One line: in these matched pairs spiking costs a little clean accuracy,
ties on calibration after the standard correction, degrades less under
noise that is independent across frames, and degrades more, and more
confidently, when the same noise persists across frames.

Caveats: one architecture, one neuron model, frames rather than raw
events, 5 and 3 seeds, MSE loss for both models, temperature fitted on the
test set by two-fold cross-fitting because neither dataset has a
validation split, and energy from an accounting model (Horowitz 2014 pJ
costs), not from measured power.

## Live demo and single-seed write-up

- Live demo: [tayyabmanan.com/demo/spikes](https://tayyabmanan.com/demo/spikes).
  Pick a test recording, damage it, and watch both trained models answer
  the same damaged input. It runs the seed-0 checkpoints on Modal through
  `scripts/step9_demo_api.py`.
- Project page: [tayyabmanan.com/projects/do-spikes-fail-differently](https://tayyabmanan.com/projects/do-spikes-fail-differently).
- Write-up: [tayyabmanan.com/blog/do-spikes-fail-differently-snn-vs-ann](https://tayyabmanan.com/blog/do-spikes-fail-differently-snn-vs-ann).

These three date from 2026-09-08 and show seed 0 with the checkpoint that
scored best on the test set. Two of their claims did not survive five
seeds with last-epoch weights: the DVS128 Gesture SNN's sensitivity to
frame order, and an SNN lead at the lowest noise rate. The paper and the
summaries above supersede them.

## Repository layout

```
scripts/                  numbered pipeline. Modal scripts run on an A10G, the rest run locally on CPU.
  step1_lif.py              hand-rolled LIF neuron, teaching script
  step2_surrogate.py        surrogate gradients, 3 shapes x 5 seeds
  step3_train_dvs.py        SNN training (Modal), --seed N
  step4_train_ann.py        ANN twin training (Modal), --seed N
  step5_dump_logits.py      clean test outputs (Modal)
  step5_analysis.py         accuracy, ECE, reliability diagrams, paired tests
  step5_temperature.py      temperature scaling by two-fold cross-fitting
  step6_corruption.py       corruption suite: noise, event drop, occlusion, shuffle (Modal)
  step6_analysis.py         corruption curves with CIs
  step6b_noise_sweep.py     noise persistence sweep, k in {1, 2, 4, 8, 16} (Modal)
  step6b_analysis.py        persistence curves with CIs
  step7_energy.py           firing rates, sparsity, energy table (Modal)
  step8_training_curves.py  training curves from both metrics files
  step9_demo_api.py         live demo backend (Modal) and a local server
  step9_demo_export.py      published numbers for the portfolio demo
  step10_multiseed.py       aggregates results/data/seed*/ into the summaries
  step11_cifar10dvs.py      CIFAR10-DVS: prepare, train, dump, corrupt, sweep, energy (Modal)
  step12_tau_ablation.py    time-constant ablation, train and eval (Modal)
  step12_tau_analysis.py    ablation summary and figure
  viz_dataset.py            dataset grid and animated gestures
results/data/
  seedS/                    DVS128 Gesture dumps and training metrics for seed S
  c10/seedS/                CIFAR10-DVS dumps for seed S
  tau/tau{tau}/seedS/       ablation dumps
  multiseed_summary.*       DVS128 Gesture summary, json and md
  c10/multiseed_summary_c10.*  and  tau/tau_ablation_summary.*
  *.npz at the top level    legacy seed-0 best.pt dumps that the live demo reads
results/figures/            generated figures
```

## Reproducing

Training and evaluation run on Modal (A10G). Analysis runs locally. Run
everything from the repo root. `--seed N` selects the seed, default 0.
Modal volume paths take no leading slash.

```
# one-time Modal setup
pip install modal
modal setup
modal volume create dvs128-data
modal volume put dvs128-data <path-to>/DVS128Gesture/download DVS128Gesture/download
modal run scripts/step3_train_dvs.py --mode prepare              # frame cache, once

# DVS128 Gesture, per seed. Training spawns and returns; do not hold a --wait window open.
modal run --detach scripts/step3_train_dvs.py --seed 1            # SNN
modal run --detach scripts/step4_train_ann.py --seed 1            # ANN twin
modal run scripts/step5_dump_logits.py --seed 1
modal run scripts/step6_corruption.py --seed 1
modal run scripts/step6b_noise_sweep.py --seed 1
modal run scripts/step7_energy.py --seed 1
modal volume get dvs128-data eval/seed1/test_outputs.npz results/data/seed1/test_outputs.npz
modal volume get dvs128-data eval/seed1/corruption_outputs.npz results/data/seed1/corruption_outputs.npz
modal volume get dvs128-data eval/seed1/noise_sweep_outputs.npz results/data/seed1/noise_sweep_outputs.npz
modal volume get dvs128-data eval/seed1/energy.json results/data/seed1/energy.json
modal volume get dvs128-data checkpoints/seed1/metrics.csv results/data/seed1/step3_metrics.csv
modal volume get dvs128-data checkpoints_ann/seed1/metrics.csv results/data/seed1/step4_metrics.csv

# CIFAR10-DVS, one app
modal run --detach scripts/step11_cifar10dvs.py --mode prepare    # once
modal run --detach scripts/step11_cifar10dvs.py --mode train --model snn --seed 0
modal run --detach scripts/step11_cifar10dvs.py --mode train --model ann --seed 0
modal run scripts/step11_cifar10dvs.py --mode dump --seed 0       # then corrupt, sweep, energy
modal volume get dvs128-data eval_c10/seed0/test_outputs.npz results/data/c10/seed0/test_outputs.npz

# time-constant ablation, SNN only, against the existing ANN twins
modal run --detach scripts/step12_tau_ablation.py --mode train --tau 4.0 --seed 0
modal run --detach scripts/step12_tau_ablation.py --mode eval --tau 4.0 --seed 0

# local analysis and aggregation
python scripts/step5_analysis.py results/data/seed1/test_outputs.npz
python scripts/step6_analysis.py results/data/seed1/corruption_outputs.npz
python scripts/step5_temperature.py results/data/seed1/test_outputs.npz results/data/seed1/corruption_outputs.npz
python scripts/step6b_analysis.py results/data/seed1/noise_sweep_outputs.npz
python scripts/step10_multiseed.py
python scripts/step10_multiseed.py --root results/data/c10 --tag _c10 --name CIFAR10-DVS
python scripts/step12_tau_analysis.py
```

On Windows, if the `modal` shim is blocked, call `python -m modal ...` instead.

## Pins

torch==2.4.0, torchvision==0.19.0, spikingjelly==0.0.0.0.14, numpy<2.
These are load-bearing. See requirements.txt.

## Method notes

- The matched pair is the point. SNN and ANN share everything except the
  neuron. Any change applies to both or to neither.
- Report last.pt, the fixed-budget epoch-64 checkpoint. best.pt is selected
  on the test set and is reference-only.
- Loss is MSE against one-hot targets for both models. Outputs are
  rate-like scores, not logits. Confidence is the normalized score
  max(out)/sum(out). Softmax at temperature one on these outputs
  manufactures fake underconfidence. Temperature scaling with a fitted
  temperature is the post-hoc baseline.
- Every metric carries a bootstrap 95% CI. Across seeds: mean, SD, range
  and a hierarchical bootstrap over seeds and samples.
- Corruptions are seeded per condition and do not depend on the training
  seed, so both models of every seed see byte-identical corrupted inputs.
  The persistence sweep's k=1 condition has the same distribution as the
  background-noise corruption but is a separate draw, so the two differ
  by up to 0.009.
- The temporal shuffle is a designed control. The ANN averages its outputs
  over T, so it is invariant to frame order by construction. Any change
  under shuffle is SNN temporal processing.
- Energy is an accounting model with 4.6 pJ per multiply-accumulate and
  0.9 pJ per accumulate at 45 nm. It ignores memory traffic.
