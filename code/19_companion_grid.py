"""Precompute the result grid for the companion-site faking-cost calculator.

Reuses the Part B Monte Carlo mechanism model from 09_selection_simulation.py
(calibrate + simulate logic copied verbatim except: per-cell deterministic rng,
larger pool for stable single-run cells, and JSON output). All inputs are the
published/meta-analytic values documented in 09.

Grid: context {US, Pooled, Non-US} x prevalence {.20...50 step .05}
      x selection ratio {.05...50 step .05}
Outputs per cell: pct_hires_changed, perf_loss_pct, AI_ratio_honest,
AI_ratio_faked, rank_correlation_pool.

Output: output/companion_grid.json
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import fsolve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

V_G, V_C = 0.31, 0.19
D_G, D_C = 0.79, -0.07
P_MIN = 1 / 6
D_SHIFT = 0.45
W = 0.5
N_POOL = 400_000

CONTEXTS_MC = {
    "US":       (0.013, 0.007),
    "Pooled":   (0.032, -0.074),
    "Non-US":   (0.054, -0.272),
}


def calibrate(rho0, target_r, sigma, p_fake):
    def moments(params):
        mu_f, b = params
        var_infl = p_fake * (mu_f**2 + b**2 + sigma**2) - (p_fake * mu_f) ** 2
        var_cobs = 1 - 2 * p_fake * b * rho0 + var_infl
        r_gc = (rho0 - p_fake * b) / np.sqrt(var_cobs)
        d = (p_fake * mu_f) / np.sqrt((1 + var_cobs) / 2)
        return [r_gc - target_r, d - D_SHIFT]

    mu_f, b = fsolve(moments, x0=[D_SHIFT / p_fake, max(0.0, (rho0 - target_r))])
    return float(mu_f), float(b)


def simulate(context, sr, p_fake, sigma=0.50, n_pool=N_POOL, seed=0):
    rng = np.random.default_rng(seed)
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

    def ai(sel):
        m = minority[sel].mean()
        return (m / P_MIN) / ((1 - m) / (1 - P_MIN))

    perf_h = y[sel_honest].mean()
    perf_f = y[sel_faked].mean()
    # swapped hires: those selected under faking but not under honest
    # responding, vs. the honest picks they displaced
    swapped_in = np.setdiff1d(sel_faked, sel_honest, assume_unique=True)
    displaced = np.setdiff1d(sel_honest, sel_faked, assume_unique=True)
    swap_gap = float(y[displaced].mean() - y[swapped_in].mean()) if len(swapped_in) else 0.0
    return {
        "hires_changed": round(100 * (1 - overlap), 1),
        "perf_loss": round(100 * (perf_h - perf_f) / perf_h, 1),
        "perf_gap_sd": round(float(perf_h - perf_f), 3),   # aggregate, per hire
        "swap_gap_sd": round(swap_gap, 3),                 # per swapped hire
        "ai_honest": round(ai(sel_honest), 2),
        "ai_faked": round(ai(sel_faked), 2),
    }


def main():
    prevalences = [round(p, 2) for p in np.arange(0.20, 0.501, 0.05)]
    srs = [round(s, 2) for s in np.arange(0.05, 0.501, 0.05)]
    grid = {}
    for ctx in CONTEXTS_MC:
        grid[ctx] = {}
        for pf in prevalences:
            grid[ctx][f"{pf:.2f}"] = {}
            for sr in srs:
                seed = hash((ctx, pf, sr)) % (2**31)
                cell = simulate(ctx, sr, pf, seed=seed)
                grid[ctx][f"{pf:.2f}"][f"{sr:.2f}"] = cell
                print(ctx, pf, sr, cell)
    payload = {
        "meta": {
            "contexts": list(CONTEXTS_MC),
            "prevalences": [f"{p:.2f}" for p in prevalences],
            "selection_ratios": [f"{s:.2f}" for s in srs],
            "n_pool": N_POOL,
            "sigma": 0.50,
            "source": "09_selection_simulation.py Part B mechanism model",
        },
        "grid": grid,
    }
    out = OUT / "companion_grid.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
