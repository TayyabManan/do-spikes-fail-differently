"""Step 2: make the spike differentiable.

Step 1 ended on a wall. This line:

    if v >= v_th: spike = 1

is a step function. Its slope is 0 everywhere and undefined at
the jump. Backprop multiplies slopes together, so a 0 kills the
whole chain. No slope, no learning.

The fix is a lie we tell on purpose.
  - Forward pass: keep the hard step. Real spikes, 0 or 1.
  - Backward pass: pretend the step was a smooth S-curve and
    use that slope instead.

That fake slope is the SURROGATE GRADIENT. It is the single
trick that makes every deep SNN in your thesis trainable:
Meta-SpikeFormer, SpikeCLIP, SpikeLoRA, all of them.

This script does two things:
  1. Plots three surrogate shapes so you see what you are choosing.
  2. Trains the same tiny SNN three times, one per surrogate,
     on a task where timing carries the answer.

Run (from the repo root):
    python scripts/step2_surrogate.py
Outputs: results/figures/surrogates.png, results/figures/training_curves.png
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)

V_TH = 1.0
TAU = 5.0     # in timesteps now, not seconds. decay = 1 - 1/TAU
T = 25        # timesteps per sample


# ---------------------------------------------------------------
# 1. The surrogate: a custom autograd Function
# ---------------------------------------------------------------
# forward() and backward() are deliberately inconsistent.
# That inconsistency IS the method. It is not a bug.

class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, kind):
        # x = v - v_th. Positive means "fire".
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        ctx.kind = kind
        return (x >= 0).float()          # the hard step. Real spikes.

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        a = ctx.alpha
        if ctx.kind == "atan":
            # arctan surrogate. SpikingJelly's default.
            # Long tails: neurons far from threshold still learn.
            sg = a / 2 / (1 + (torch.pi / 2 * a * x) ** 2)
        elif ctx.kind == "sigmoid":
            s = torch.sigmoid(a * x)
            sg = a * s * (1 - s)
        elif ctx.kind == "rect":
            # Rectangle / boxcar. Zero outside a narrow window.
            # Neurons far from threshold get NO gradient at all.
            sg = (x.abs() < 0.5 / a).float() * a
        else:
            raise ValueError(ctx.kind)
        return grad_out * sg, None, None


def spike(x, alpha=2.0, kind="atan"):
    return SpikeFn.apply(x, alpha, kind)


# ---------------------------------------------------------------
# 2. A LIF layer. Same equation as step 1, now on tensors.
# ---------------------------------------------------------------

class LIF(nn.Module):
    def __init__(self, kind="atan", alpha=2.0):
        super().__init__()
        self.kind, self.alpha = kind, alpha
        self.v = None

    def reset(self):
        self.v = None            # MUST be called between samples.

    def forward(self, x):
        if self.v is None:
            self.v = torch.zeros_like(x)
        # step 1's line, rewritten with decay = 1 - dt/tau
        self.v = self.v + (1.0 / TAU) * (-self.v + x)
        s = spike(self.v - V_TH, self.alpha, self.kind)
        self.v = self.v * (1 - s)        # soft reset-to-zero, differentiable
        return s


class Net(nn.Module):
    """2 inputs -> 16 spiking hidden -> 2 outputs. Tiny on purpose."""

    def __init__(self, kind, alpha=2.0):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.lif = LIF(kind, alpha)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):                 # x: [batch, T, 2]
        self.lif.reset()                  # BUG SOURCE #1 if you forget
        out = 0
        for t in range(x.shape[1]):       # loop over time. No way around it.
            s = self.lif(self.fc1(x[:, t]))
            out = out + self.fc2(s)
        return out / x.shape[1]           # average over T = rate readout


# ---------------------------------------------------------------
# 3. A task where TIMING is the answer, not intensity
# ---------------------------------------------------------------
# Class 0: channel A pulses, then channel B.
# Class 1: channel B pulses, then channel A.
# Total input is identical. Only the order differs.
# An ANN on summed input cannot solve this. An SNN can,
# because the membrane remembers.

def make_data(n):
    x = torch.zeros(n, T, 2)
    y = torch.randint(0, 2, (n,))
    for i in range(n):
        first, second = (0, 1) if y[i] == 0 else (1, 0)
        t1 = torch.randint(2, 8, (1,)).item()
        t2 = torch.randint(14, 20, (1,)).item()
        x[i, t1: t1 + 3, first] = 3.0
        x[i, t2: t2 + 3, second] = 3.0
    x += 0.1 * torch.randn_like(x)
    return x, y


def train(kind, alpha=2.0, epochs=60, seed=0):
    torch.manual_seed(seed)
    net = Net(kind, alpha)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    lossf = nn.CrossEntropyLoss()
    xtr, ytr = make_data(512)
    xte, yte = make_data(256)
    accs = []
    for _ in range(epochs):
        for i in range(0, len(xtr), 64):
            opt.zero_grad()
            loss = lossf(net(xtr[i: i + 64]), ytr[i: i + 64])
            loss.backward()
            opt.step()
        with torch.no_grad():
            accs.append((net(xte).argmax(1) == yte).float().mean().item())
    return accs


# ---------------------------------------------------------------
# 4. Run it
# ---------------------------------------------------------------

if __name__ == "__main__":
    # plot the three surrogate shapes
    xs = torch.linspace(-3, 3, 400, requires_grad=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(xs.detach(), (xs.detach() >= 0).float(), "k", lw=2)
    ax[0].set_title("forward: the hard step (slope 0 everywhere)")
    ax[0].set_xlabel("v - threshold")
    for kind in ["atan", "sigmoid", "rect"]:
        g = torch.autograd.grad(
            spike(xs, 2.0, kind).sum(), xs, retain_graph=True)[0]
        ax[1].plot(xs.detach(), g, label=kind, lw=2)
    ax[1].set_title("backward: the fake slope we use instead")
    ax[1].set_xlabel("v - threshold")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig("results/figures/surrogates.png", dpi=150)
    print("wrote results/figures/surrogates.png")

    # train 5 seeds per surrogate
    results = {}
    for kind in ["atan", "sigmoid", "rect"]:
        results[kind] = [train(kind, seed=s) for s in range(5)]
        finals = [r[-1] for r in results[kind]]
        print(f"{kind:<8} final acc: mean={sum(finals) / 5:.3f} "
              f"min={min(finals):.3f} max={max(finals):.3f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    for kind, accs in results.items():
        accs = np.array(accs)
        ax.plot(accs.mean(axis=0), label=kind, lw=2)
        ax.fill_between(range(accs.shape[1]), accs.min(axis=0),
                        accs.max(axis=0), alpha=0.2)
    ax.axhline(0.5, ls="--", c="gray", label="chance")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy")
    ax.set_title(
        "Mean of 5 seeds, band = min to max. Only the fake slope differs.")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/training_curves.png", dpi=150)
    print("wrote results/figures/training_curves.png")

# ---- what to look for ----
# surrogates.png: the left panel is flat. That flatness is why
#   training fails without the trick. The right panel shows what
#   we substitute. Note how far from zero each curve still has
#   height. That reach decides which neurons can learn.
# training_curves.png: rect usually learns slowest or stalls. Its
#   gradient is exactly 0 outside a narrow band, so a neuron
#   sitting far below threshold gets no signal and never recovers.
#   atan and sigmoid keep long tails, so they pull dead neurons back.

# ---- TRY THIS ----
# A. alpha in train(): try 0.5 and 8.0 with "atan".
#    Small alpha = wide flat gradient, blurry but forgiving.
#    Large alpha = tall narrow spike, precise but easy to miss.
#    This is the bias-variance knob of SNN training.
# B. Delete `self.lif.reset()` in Net.forward. Watch accuracy die.
#    Sample N inherits sample N-1's voltage. This is the #1
#    beginner bug and it fails quietly, not loudly.
# C. Set TAU = 1.0 (no memory at all). The task needs memory, so
#    accuracy should fall toward chance. Confirms the task is
#    really testing timing.
