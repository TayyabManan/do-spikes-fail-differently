# Do Spikes Fail Differently? An SNN-vs-ANN Reliability Study on DVS128Gesture

A matched-pair comparison of a spiking neural network (SNN) and its
artificial neural network (ANN) twin on DVS128Gesture. Everything is
held constant except the neuron model: architecture, parameter count,
data pipeline, seed, loss, optimizer, schedule, batch size, epochs,
and T=16 frames. Each LIF neuron in the SNN becomes a ReLU in the ANN.
The study measures what spiking costs and what it buys: accuracy,
calibration, robustness under corruption, temporal sensitivity, energy.

Semester project. Miniature of the master's thesis "Do Spikes Fail
Differently? Reliability of Parameter-Efficient Spiking Event-Language
Models for Open-Vocabulary Recognition".

## Live demo and write-up

- Live demo: [tayyabmanan.com/demo/spikes](https://tayyabmanan.com/demo/spikes).
  Pick a test recording, damage it, and watch both trained models answer
  the same damaged input. It runs the checkpoints from this repo on Modal
  through `scripts/step9_demo_api.py`, with the step 6 corruptions and the
  step 7 energy accounting unchanged, and shows the published curves under
  each live run.
- Project page: [tayyabmanan.com/projects/do-spikes-fail-differently](https://tayyabmanan.com/projects/do-spikes-fail-differently).
- Write-up: [tayyabmanan.com/blog/do-spikes-fail-differently-snn-vs-ann](https://tayyabmanan.com/blog/do-spikes-fail-differently-snn-vs-ann).
- Report: `report/step8_report.pdf`.

## Findings

All numbers on the 288-sample test set with bootstrap 95% CIs, single
seed.

| Axis | Result |
|---|---|
| Accuracy | SNN 0.9306 vs ANN 0.9653. Gap +0.035 [+0.010, +0.063], McNemar p = 0.021. Real, but 3.5 points = 10 samples. |
| Calibration (clean) | SNN ECE 0.040 vs ANN ECE 0.039. Paired difference CI includes zero. No measurable calibration cost. |
| Noise corruption | Curves cross. SNN leads by up to +0.28 accuracy. ANN grows overconfident (conf - acc up to +0.34); SNN confidence tracks its accuracy within 4 points. |
| Drop / occlusion | ANN keeps its lead. At extreme event loss both degrade to 0.368 (SNN) and 0.424 (ANN) against chance 0.091, while ~77-81% confident. |
| Temporal shuffle | ANN provably invariant (verified flat). SNN loses 0.021 [0.007, 0.038] at full shuffle. It uses timing, worth about two points. |
| Energy (accounting model) | SNN 3.30 mJ/sample vs ANN 61.90 mJ/sample, 18.7x. 98.1% of neuron-timesteps silent (weighted). |

One line: spiking here is a trade, not a downgrade. It costs 3.5
accuracy points and mild timing sensitivity. It buys a calibration tie,
large noise robustness with honest confidence, and ~19x paper energy.

Caveats, stated plainly: single seed, n=288, energy is an accounting
model (Horowitz 2014 pJ costs, not wall power; savings require
event-driven hardware), and the noise-robustness mechanism (LIF leak as
temporal low-pass filter) is a hypothesis, not a demonstrated cause.

## Repository layout

```
scripts/            numbered pipeline, step1 to step9
  step1_lif.py            hand-rolled LIF neuron, teaching script (local)
  step2_surrogate.py      surrogate gradients, 3 shapes x 5 seeds (local)
  step3_train_dvs.py      SNN training (Modal, GPU)
  step4_train_ann.py      ANN twin training (Modal, GPU)
  step5_dump_logits.py    dump clean test outputs (Modal, GPU)
  step5_analysis.py       ECE, reliability diagrams, paired tests (local)
  step6_corruption.py     corruption stress test dump (Modal, GPU)
  step6_analysis.py       corruption curves + CIs (local)
  step7_energy.py         firing rates + energy table (Modal, GPU)
  step8_training_curves.py  training curves figure from both metrics csv (local)
  viz_dataset.py          dataset grid and animated gestures figures (local)
  step9_demo_api.py       live demo backend (Modal, CPU, scales to zero) + local server
  step9_demo_export.py    published numbers -> the portfolio's demo data module (local)
results/
  data/             npz output dumps and training metrics csv
  figures/          all generated figures
report/             writeup
```

## Reproducing

Training runs on Modal (A10G). Analysis runs locally on CPU. Run
everything from the repo root.

```
# one-time Modal setup
pip install modal
modal setup
modal volume create dvs128-data
modal volume put dvs128-data <path-to>/DVS128Gesture/download /DVS128Gesture/download

# training (resumable, checkpoints every epoch)
modal run scripts/step3_train_dvs.py --mode prepare   # frame cache, once
modal run --detach scripts/step3_train_dvs.py         # SNN
modal run --detach scripts/step4_train_ann.py         # ANN twin

# evaluation dumps
modal run scripts/step5_dump_logits.py
modal run scripts/step6_corruption.py
modal run scripts/step7_energy.py

# live demo (tayyabmanan.com/demo/spikes): bank + thumbnails once, then deploy
modal secret create spikes-demo-key DEMO_KEY=<random string>
modal run scripts/step9_demo_api.py --mode prepare
modal deploy scripts/step9_demo_api.py
python scripts/step9_demo_export.py            # population stats with CIs -> portfolio
python scripts/step9_demo_api.py --local --port 8009   # local server for the page

# fetch outputs
modal volume get dvs128-data /eval/test_outputs.npz results/data/test_outputs.npz
modal volume get dvs128-data /eval/corruption_outputs.npz results/data/corruption_outputs.npz

# local analysis (writes figures to results/figures/)
python scripts/step5_analysis.py results/data/test_outputs.npz
python scripts/step6_analysis.py results/data/corruption_outputs.npz
```

## Pins

torch==2.4.0, torchvision==0.19.0, spikingjelly==0.0.0.0.14, numpy<2.
These are load-bearing. See requirements.txt.

## Method notes

- Loss is MSE against one-hot targets for both models. Outputs are
  probability-like rates, not logits. Confidence is the normalized
  score max(out)/sum(out). Softmax on rate outputs manufactures fake
  underconfidence.
- Every reported metric carries a bootstrap 95% CI. n=288 makes point
  estimates untrustworthy alone.
- Corruptions are seeded with fixed offsets. Both models see
  byte-identical corrupted inputs, and reruns are reproducible. The
  archived corruption results predate this fix: their SNN/ANN pairing
  is valid, their exact numbers are not re-derivable.
- The temporal shuffle corruption is a designed control: the ANN
  averages logits over T, so it is invariant to frame order by
  construction. Any degradation is pure SNN temporal processing.
