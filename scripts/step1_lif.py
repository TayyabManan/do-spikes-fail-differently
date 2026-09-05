"""Step 1: one LIF neuron, by hand, no libraries.

The model is a leaky bucket.
  - Input current pours water in.
  - A hole leaks water out. Always.
  - Water level = membrane voltage v.
  - When v crosses the threshold, the neuron fires one spike
    and the bucket instantly empties (reset).

That is the whole neuron. Everything in your thesis stack
(SpikingJelly's LIFNode, Meta-SpikeFormer, SpikeCLIP) is
millions of copies of the loop below.

Run (from the repo root):
    python scripts/step1_lif.py
Output: results/figures/lif_step1.png and a spike count printed.
"""

import torch
import matplotlib.pyplot as plt

# ---- constants ----
dt = 1e-3          # simulation step: 1 ms
T = 1000           # 1000 steps = 1 second of simulated time
tau = 100e-3        # membrane time constant: 20 ms.
# Big tau = small hole = long memory of past input.
v_th = 1.0         # threshold: fire when v reaches this
v_reset = 0.0      # after a spike, v snaps back to this

# ---- input: a 5 Hz sine current ----
t = torch.arange(T) * dt
current = 1.1 * torch.ones(T)

# ---- simulate, one millisecond at a time ----
v = torch.zeros(T)        # voltage trace
spikes = torch.zeros(T)   # 1.0 at every timestep with a spike

for i in range(1, T):
    # The one equation. Two forces on the water level:
    #   leak:  -v[i-1]      pulls v toward 0, always
    #   drive: +current[i]  pushes v toward the input value
    # dt/tau sets how fast both act.
    v[i] = v[i - 1] + (dt / tau) * (-v[i - 1] + current[i])

    if v[i] >= v_th:      # threshold crossed:
        spikes[i] = 1.0   # emit a spike (this is the network's output)
        v[i] = v_reset    # and empty the bucket

print(f"spikes fired: {int(spikes.sum())}")

# ---- plot: input, voltage, spikes, stacked ----
fig, ax = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
ax[0].plot(t, current, color="tab:orange")
ax[0].set_ylabel("input current")
ax[1].plot(t, v, color="tab:blue")
ax[1].axhline(v_th, ls="--", color="gray", label="threshold")
ax[1].set_ylabel("membrane v")
ax[1].legend(loc="lower right")
ax[2].eventplot(t[spikes.bool()], colors="black", linelengths=0.8)
ax[2].set_ylabel("spikes")
ax[2].set_xlabel("time (s)")
ax[2].set_yticks([])
fig.suptitle("One LIF neuron driven by a sine current")
fig.tight_layout()
fig.savefig("results/figures/lif_step1.png", dpi=150)
print("wrote results/figures/lif_step1.png")

# ---- what to look for in the plot ----
# 1. Spikes only near the crests. The neuron converts input
#    strength into spike count. That is rate coding.
# 2. Between the two spikes on each crest, v climbs in a curve,
#    not a line. The leak fights the drive the whole way up.
# 3. At the troughs v goes negative. Negative input pushes the
#    level below empty. That is inhibition.
# 4. After each spike, v restarts from zero. The neuron forgets
#    its charge. Memory lives only below threshold.

# ---- TRY THIS: predict first, then change one line, rerun ----
# A. tau = 100e-3
#    Prediction: leak weakens, v tracks the input slowly and
#    smoothly. Fewer spikes? More? Write your guess down first.
# B. v_th = 1.4
#    Prediction: threshold nearly equals the input peak (1.5).
#    The neuron barely fires, and only at the exact crest.
# C. current = 0.9 * torch.ones(T)
#    Constant input below threshold. v charges toward 0.9 and
#    flattens. Zero spikes, forever. The asymptote equals the
#    input value; if input < v_th the neuron cannot fire. The
#    minimum input that fires is called the rheobase.
# D. current = 1.1 * torch.ones(T)
#    Just above threshold. Regular, evenly spaced spikes.
#    Slow, because v crawls over the line each time.

# ---- the seed for step 2 ----
# The line `if v[i] >= v_th` is a step function: output jumps
# 0 to 1. Its slope is zero everywhere and undefined at the
# jump. Gradient descent needs slopes, so backprop dies here.
# Every trick in your thesis' training stack exists to fake a
# slope for this one line. That fake slope is the surrogate
# gradient. That is step 2.
