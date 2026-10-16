"""Standalone lay-reader scatter for the companion site hero: who fakes, and how much.

Same mechanism model as 09/19 (pooled calibration, 40% prevalence, sigma=.50),
rendered in the site palette with plain-language axis labels.

Output: output/companion_scatter.png (and copied to the site by hand/shell)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

BG, INK, GRAY, RED, BLUE = "#faf7f2", "#1f2a36", "#5a6572", "#c0392b", "#2e6f9e"

RHO0, TARGET_R = 0.032, -0.074   # pooled
P_FAKE, SIGMA, D_SHIFT = 0.40, 0.50, 0.45


def calibrate():
    def moments(params):
        mu_f, b = params
        var_infl = P_FAKE * (mu_f**2 + b**2 + SIGMA**2) - (P_FAKE * mu_f) ** 2
        var_cobs = 1 - 2 * P_FAKE * b * RHO0 + var_infl
        r_gc = (RHO0 - P_FAKE * b) / np.sqrt(var_cobs)
        d = (P_FAKE * mu_f) / np.sqrt((1 + var_cobs) / 2)
        return [r_gc - TARGET_R, d - D_SHIFT]
    mu_f, b = fsolve(moments, x0=[D_SHIFT / P_FAKE, 0.1])
    return float(mu_f), float(b)


mu_f, b = calibrate()
rng = np.random.default_rng(7)
n = 2600
z = rng.multivariate_normal([0, 0], [[1, RHO0], [RHO0, 1]], size=n)
g = z[:, 0]
fakes = rng.random(n) < P_FAKE
infl = np.where(fakes, mu_f - b * g + rng.normal(0, SIGMA, n), 0.0)

fig = plt.figure(figsize=(8.8, 5.2), dpi=200)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0.11, 0.14, 0.86, 0.78])
ax.set_facecolor(BG)

ax.scatter(g[~fakes], infl[~fakes], s=7, color=GRAY, alpha=0.25, linewidths=0,
           label="Honest responders (no inflation)")
ax.scatter(g[fakes], infl[fakes], s=9, color=RED, alpha=0.5, linewidths=0,
           label="Fakers")
xs = np.linspace(-3, 3, 10)
ax.plot(xs, mu_f - b * xs, color=INK, lw=2.2)

ax.axhline(0, color=GRAY, lw=0.8, alpha=0.6)
ax.set_xlim(-3.2, 3.2)
ax.set_xlabel("Real cognitive ability  \u2192  higher", fontsize=12, color=INK)
ax.set_ylabel("How much they inflate their\npersonality score (SD units)", fontsize=12, color=INK)
ax.tick_params(colors=GRAY, labelsize=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRAY)

ax.annotate("Lower-ability applicants\ninflate the most",
            xy=(-2.0, mu_f - b * -2.0), xytext=(-1.45, 3.6),
            fontsize=12, fontweight="bold", color=RED,
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
ax.annotate("Higher-ability applicants\ninflate far less",
            xy=(2.0, mu_f - b * 2.0), xytext=(0.9, 1.9),
            fontsize=12, fontweight="bold", color=BLUE,
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
ax.legend(loc="lower left", fontsize=10, frameon=False, labelcolor=INK)

fig.savefig(OUT / "companion_scatter.png", facecolor=BG, bbox_inches="tight")
print("saved", OUT / "companion_scatter.png", f"(mu_f={mu_f:.2f}, b={b:.2f})")
