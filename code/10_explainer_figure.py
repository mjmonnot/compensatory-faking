"""Explainer figure: how a stakes-shifted correlation maps onto rank-order churn
and hire quality. Single condition (Non-US calibration, SR = .10, sigma = .50),
same data-generating mechanism and calibration as 09_selection_simulation.py.

Panel A: the mechanism -- conscientiousness inflation as a function of g among
         fakers (compensatory slope), with the pool correlation it produces.
Panel B: the churn -- honest vs. as-scored composite with top-10% cutoffs;
         off-diagonal quadrants are decisions changed by faking.
Panel C: the consequence -- true-performance distributions of the two hire sets.

Output: output/figure4_mechanism_explainer.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import fsolve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
rng = np.random.default_rng(2026)

V_G, V_C = 0.31, 0.19
P_FAKE, D_SHIFT, W = 0.40, 0.45, 0.5
RHO0, TARGET_R, SIGMA, SR = 0.054, -0.272, 0.50, 0.10   # Non-US condition
N_POOL = 200_000


def calibrate(rho0, target_r, sigma):
    def moments(params):
        mu_f, b = params
        var_infl = P_FAKE * (mu_f**2 + b**2 + sigma**2) - (P_FAKE * mu_f) ** 2
        var_cobs = 1 - 2 * P_FAKE * b * rho0 + var_infl
        r_gc = (rho0 - P_FAKE * b) / np.sqrt(var_cobs)
        d = (P_FAKE * mu_f) / np.sqrt((1 + var_cobs) / 2)
        return [r_gc - target_r, d - D_SHIFT]
    mu_f, b = fsolve(moments, x0=[D_SHIFT / P_FAKE, max(0.0, rho0 - target_r)])
    return float(mu_f), float(b)


mu_f, b = calibrate(RHO0, TARGET_R, SIGMA)

z = rng.multivariate_normal([0, 0], [[1, RHO0], [RHO0, 1]], size=N_POOL)
g, c_true = z[:, 0], z[:, 1]
bcoef = np.linalg.solve(np.array([[1, RHO0], [RHO0, 1]]), np.array([V_G, V_C]))
resid_sd = np.sqrt(1 - np.array([V_G, V_C]) @ bcoef)
y = bcoef[0] * g + bcoef[1] * c_true + rng.normal(0, resid_sd, N_POOL)

fakes = rng.random(N_POOL) < P_FAKE
inflation = np.where(fakes, mu_f - b * g + rng.normal(0, SIGMA, N_POOL), 0.0)
c_obs = c_true + inflation

comp_honest = W * g + (1 - W) * c_true
comp_scored = W * g + (1 - W) * (c_obs - c_obs.mean()) / c_obs.std()

n_sel = int(SR * N_POOL)
cut_h = np.sort(comp_honest)[-n_sel]
cut_s = np.sort(comp_scored)[-n_sel]
sel_h = comp_honest >= cut_h
sel_s = comp_scored >= cut_s
pct_changed = 100 * (1 - np.mean(sel_h & sel_s) / SR)
perf_h, perf_s = y[sel_h].mean(), y[sel_s].mean()
loss = 100 * (perf_h - perf_s) / perf_h
r_pool = np.corrcoef(g, c_obs)[0, 1]

# APA figure style: sans serif 8-14 pt; figure number/title/note live in the
# manuscript, not the image; panels carry plain bold letter labels.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
})
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))

# ---- Panel A: mechanism
ax = axes[0]
idx_f = rng.choice(np.flatnonzero(fakes), 2500, replace=False)
ax.scatter(g[idx_f], inflation[idx_f], s=4, alpha=0.25, color="#c0392b",
           rasterized=True)
ax.axhline(0, color="0.4", lw=1, ls="-")
ax.annotate("honest responders (60%): zero inflation", (0.02, 0.03),
            xycoords="axes fraction", fontsize=8, color="0.35")
ax.annotate("fakers (40%)", (0.06, 0.55), xycoords="axes fraction",
            fontsize=8, color="#c0392b")
xs = np.linspace(-3, 3, 10)
ax.plot(xs, mu_f - b * xs, color="black", lw=2)
ax.annotate(f"inflation = {mu_f:.2f} \u2212 {b:.2f} \u00d7 g",
            (0.03, 0.90), xycoords="axes fraction", fontsize=9, fontweight="bold")
ax.annotate(f"pool r(g, C as scored) = {r_pool:+.2f}\n(honest baseline: {RHO0:+.2f})",
            (0.52, 0.76), xycoords="axes fraction", fontsize=8)
ax.set_xlabel("General mental ability (z)")
ax.set_ylabel("Conscientiousness inflation (SD units)")
ax.set_title("A", loc="left", fontsize=11, fontweight="bold", pad=8)

# ---- Panel B: churn
ax = axes[1]
idx = rng.choice(N_POOL, 6000, replace=False)
grp_color = np.where(fakes[idx], "#c0392b", "0.65")
ax.scatter(comp_honest[idx], comp_scored[idx], s=3, c=grp_color, alpha=0.35, rasterized=True)
ax.axvline(cut_h, color="black", lw=1, ls="--")
ax.axhline(cut_s, color="black", lw=1, ls="--")
ax.annotate("hired\neither way", (0.83, 0.90), xycoords="axes fraction",
            fontsize=8, ha="center")
ax.annotate("faked in", (0.30, 0.90), xycoords="axes fraction", fontsize=8,
            color="#c0392b", fontweight="bold", ha="center")
ax.annotate("displaced", (0.86, 0.45), xycoords="axes fraction", fontsize=8,
            color="0.25", fontweight="bold", ha="center")
ax.annotate(f"{pct_changed:.0f}% of hires change", (0.03, 0.03),
            xycoords="axes fraction", fontsize=9, fontweight="bold")
ax.set_xlabel("Composite, honest responding")
ax.set_ylabel("Composite, as scored (with faking)")
ax.set_title("B", loc="left", fontsize=11, fontweight="bold", pad=8)

# ---- Panel C: consequence (SD units with dollar-value equivalents)
SALARY = 70_000            # illustrative mean salary
SDY = 0.40 * SALARY        # SD_y convention: 40% of salary (Schmidt & Hunter, 1983)
ax = axes[2]
faked_in = sel_s & ~sel_h
displaced = sel_h & ~sel_s
groups = [
    ("Hires,\nhonest scoring", y[sel_h].mean(), "0.55"),
    ("Hires,\nas scored", y[sel_s].mean(), "#c0392b"),
    ("Displaced\n(lost seat)", y[displaced].mean(), "0.35"),
    ("Faked in\n(took seat)", y[faked_in].mean(), "#7b241c"),
]
xpos = np.arange(len(groups))
for i, (label, m, color) in enumerate(groups):
    ax.bar(i, m, width=0.62, color=color)
    ax.annotate(f"{m:.2f} SD\n(${m * SDY / 1000:.0f}K/yr)", (i, m + 0.015),
                ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(xpos, [gph[0] for gph in groups], fontsize=8)
ax.set_ylabel("Mean true job performance (SD units)")
ax.set_ylim(0, 0.92)
ax2 = ax.twinx()
ax2.set_ylim(0, 0.92 * SDY / 1000)
ax2.set_ylabel("Annual output value per hire (thousands of dollars)",
               fontsize=8)
ax2.spines[["top"]].set_visible(False)
gap_sd = y[displaced].mean() - y[faked_in].mean()
overall_sd = y[sel_h].mean() - y[sel_s].mean()
ax.annotate(
    f"overall: \u2212{overall_sd:.2f} SD \u2248 \u2212${overall_sd * SDY / 1000:.1f}K "
    f"per hire per year\nseat swap: \u2212{gap_sd:.2f} SD \u2248 "
    f"\u2212${gap_sd * SDY / 1000:.1f}K per swapped hire per year",
    (0.5, 0.90), xycoords="axes fraction", fontsize=8.5, fontweight="bold", ha="center")
con = dict(arrowstyle="->", color="0.3", lw=1.2)
ax.annotate("", xy=(3, y[faked_in].mean() + 0.09), xytext=(2, y[displaced].mean() + 0.09),
            arrowprops=con)
ax.set_title("C", loc="left", fontsize=11, fontweight="bold", pad=8)

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "figure4_mechanism_explainer.png", dpi=300, bbox_inches="tight")
print(f"mu_f={mu_f:.3f} b={b:.3f} r_pool={r_pool:.3f} "
      f"pct_changed={pct_changed:.1f} loss={loss:.1f}")
print("Saved", OUT / "figure4_mechanism_explainer.png")
