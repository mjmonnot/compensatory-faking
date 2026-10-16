"""Monte Carlo standard errors for the Study 1 selection simulation (Table 4).

Replicates the six primary Table 4 cells (3 contexts x 2 selection ratios,
sigma = .50, prevalence = .40) across R = 25 independent pools (seeds
2026..2050), mirroring the Study 2 validation battery (Morris, White, &
Crowther, 2019: mc_se = SD / sqrt(R)). The DGM and calibration are copied
verbatim from 09_selection_simulation.py, except that each replication uses
its own seeded generator; 09 itself is left untouched so the published
seed-2026 values remain bit-identical.

Output: output/study1_mcse.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import fsolve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

V_G, V_C = 0.31, 0.19
D_G, D_C = 0.79, -0.07
P_MIN = 1 / 6
P_FAKE = 0.40
D_SHIFT = 0.45
W = 0.5

CONTEXTS_MC = {
    "US":       (0.013, 0.007),
    "Pooled":   (0.032, -0.074),
    "Non-US":   (0.054, -0.272),
}

N_REPS = 25
SEEDS = range(2026, 2026 + N_REPS)


def calibrate(rho0: float, target_r: float, sigma: float,
              p_fake: float = P_FAKE) -> tuple[float, float]:
    def moments(params):
        mu_f, b = params
        var_infl = p_fake * (mu_f**2 + b**2 + sigma**2) - (p_fake * mu_f) ** 2
        var_cobs = 1 - 2 * p_fake * b * rho0 + var_infl
        r_gc = (rho0 - p_fake * b) / np.sqrt(var_cobs)
        d = (p_fake * mu_f) / np.sqrt((1 + var_cobs) / 2)
        return [r_gc - target_r, d - D_SHIFT]

    mu_f, b = fsolve(moments, x0=[D_SHIFT / p_fake, max(0.0, (rho0 - target_r))])
    return float(mu_f), float(b)


def simulate(rng, context: str, sr: float, sigma: float = 0.50,
             n_pool: int = 200_000, p_fake: float = P_FAKE) -> dict:
    rho0, target_r = CONTEXTS_MC[context]
    mu_f, b = calibrate(rho0, target_r, sigma, p_fake)

    minority = rng.random(n_pool) < P_MIN
    mu_g = np.where(minority, -D_G, 0.0)
    mu_c = np.where(minority, -D_C, 0.0)
    z = rng.multivariate_normal([0, 0], [[1, rho0], [rho0, 1]], size=n_pool)
    g, c_true = z[:, 0] + mu_g, z[:, 1] + mu_c

    bcoef = np.linalg.solve(np.array([[1, rho0], [rho0, 1]]), np.array([V_G, V_C]))
    resid_sd = np.sqrt(1 - np.array([V_G, V_C]) @ bcoef)
    y = bcoef[0] * g + bcoef[1] * c_true + rng.normal(0, resid_sd, n_pool)

    fakes = rng.random(n_pool) < p_fake
    inflation = np.where(fakes, mu_f - b * g + rng.normal(0, sigma, n_pool), 0.0)
    c_obs = c_true + inflation

    comp_honest = W * g + (1 - W) * c_true
    comp_faked = W * g + (1 - W) * (c_obs - c_obs.mean()) / c_obs.std()

    n_sel = int(sr * n_pool)
    sel_honest = np.argsort(-comp_honest)[:n_sel]
    sel_faked = np.argsort(-comp_faked)[:n_sel]
    overlap = len(np.intersect1d(sel_honest, sel_faked)) / n_sel

    def air(sel):
        m = minority[sel].mean()
        return (m / P_MIN) / ((1 - m) / (1 - P_MIN))

    perf_h = y[sel_honest].mean()
    perf_f = y[sel_faked].mean()
    return {
        "context": context,
        "selection_ratio": sr,
        "pct_hires_changed": 100 * (1 - overlap),
        "perf_loss_pct": 100 * (perf_h - perf_f) / perf_h,
        "AI_ratio_faked": air(sel_faked),
    }


rows = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    for ctx in CONTEXTS_MC:
        for sr in (0.10, 0.30):
            rec = simulate(rng, ctx, sr)
            rec["seed"] = seed
            rows.append(rec)

raw = pd.DataFrame(rows)
summ = (
    raw.groupby(["context", "selection_ratio"], sort=False)
    .agg(n_reps=("seed", "size"),
         pct_hires_changed_mc_mean=("pct_hires_changed", "mean"),
         pct_hires_changed_mc_se=("pct_hires_changed", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
         perf_loss_pct_mc_mean=("perf_loss_pct", "mean"),
         perf_loss_pct_mc_se=("perf_loss_pct", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
         AIR_faked_mc_mean=("AI_ratio_faked", "mean"),
         AIR_faked_mc_se=("AI_ratio_faked", lambda s: s.std(ddof=1) / np.sqrt(len(s))))
    .round(4)
    .reset_index()
)

with open(OUT / "study1_mcse.csv", "w", newline="") as f:
    f.write("# Study 1 Table 4 Monte Carlo SEs: R=25 replications, seeds 2026..2050, "
            "fresh 200k pool per replication, (mu_f, b) recalibrated per cell; "
            "mc_se = SD/sqrt(R) (Morris, White, & Crowther, 2019)\n")
    summ.to_csv(f, index=False)
    f.write("\n# Per-replication raw values\n")
    raw.round(4).to_csv(f, index=False)

print(summ.to_string(index=False))
print("\nMax MCSEs: hires changed = {:.3f} pts; perf loss = {:.3f} pts; AIR = {:.4f}".format(
    summ.pct_hires_changed_mc_se.max(), summ.perf_loss_pct_mc_se.max(), summ.AIR_faked_mc_se.max()))
print("Saved output/study1_mcse.csv")
