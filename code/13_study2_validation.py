"""Study 2 validation batch (v2 model): five analyses on the anchored
trait-driven faking model of 11_study2_trait_model.py.

Executes the open validation items of literature/study2_model_notes.md
section 6 (results appended there as section 8). Core machinery (pool
construction, propensity solve, WLS calibration, hire stats) is copied from
11_study2_trait_model.py with minimal generalization (V_H as an argument,
optional inflation censoring, optional per-scale skill slopes, per-replication
rng) so that model 11 remains untouched and standalone. The module-level rng
here is seeded identically (P.SEED = 2026) and consumes draws in the same
order, so the three context pools are bit-identical to model 11's pools.

Primary specification throughout: field_ns0 propensity anchors
(r_gF = -.19, r_CF = 0, r_HF = 0), prevalence .40, sigma .50,
n = 200,000, contexts US / Pooled / Non-US, SR .10/.30.

Analyses:
(1) CENSORED-INFLATION REFIT — faker inflation floored at 0 on both the C
    and SE scales (infl+ = max(infl, 0)); WLS refit for field_ns0 in all
    three contexts plus field/lab for Non-US. Key question: is the Non-US
    applicant r(g, C_obs) = -.272 target still reachable when self-sabotage
    (negative inflation) is disallowed? A blown-up distance is the finding.
(2) V_H SENSITIVITY — integrity operational validity swept over
    {.15, .18, .41} (Van Iddekinge et al. 2012 corrected vs Ones et al.
    1993); only the performance regression is re-solved, calibration is
    unchanged (it does not involve y).
(3) MONTE CARLO SEs — ADEMP-style replications (Morris, White, & Crowther
    2019): R = 25 seeds 2026..2050, fresh 200k pool per replication, full
    recalibration (propensity solve + WLS) each time; per-cell MC means and
    SEs for the headline outcomes and the calibrated inflation parameters.
    (Auto-drops to R = 10 if the projected total exceeds ~30 min.)
(4) r_CF PROFILE GRID — r_gF = -.19 and r_HF = -.04 fixed at the field
    anchors, r_CF swept over [-.30, +.20] in .05 steps; weighted-distance
    profile per context with heuristic acceptance regions at 2x the minimum
    and at minimum + 3.84 (1-df chi-square analogy — heuristic, not formal
    inference, since the WLS criterion is not a calibrated deviance).
(5) FREE THE SHARED m_skill — separate m_skill_C and m_skill_SE (4 free
    inflation parameters, 4 moments: just-identified); do the two skill
    slopes agree with the shared-mechanism restriction?

Fit failures are findings, not errors; nothing is forced.

Outputs: output/study2_validation_results.csv
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import fsolve, least_squares
from scipy.special import expit

import study2_params as P

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
rng = np.random.default_rng(P.SEED)

MOMENT_KEYS = ("d_shift", "r_gC", "delta_r_SE", "sd_ratio")
ACTIVE_MOMENTS = MOMENT_KEYS if P.INCLUDE_SE else tuple(
    k for k in MOMENT_KEYS if k != "delta_r_SE")

PRIMARY = "field_ns0"
CONTEXTS = ("US", "Pooled", "Non-US")

V_H_GRID = (0.15, 0.18, 0.41)   # Van Iddekinge 2012 corrected / default / Ones 1993
MC_SEEDS = tuple(range(2026, 2051))          # 25 replications
MC_FALLBACK_N = 10                           # if projected runtime > ~30 min
R_CF_GRID = (-0.30, -0.25, -0.20, -0.15, -0.10, -0.05,
             0.0, 0.05, 0.10, 0.15, 0.20)
R_GF_FIXED, R_HF_FIXED = -0.19, -0.04        # field anchors (Holtrop et al. 2021)


# ------------------------------------------------------------ model 11 machinery
# Copied from 11_study2_trait_model.py; generalizations noted inline.
def trait_corr(context: str) -> np.ndarray:
    """Latent correlation matrix, order (g, C, H, SE)."""
    r = P.RHO_GC_BASE[context]
    R = np.array([
        [1.0,       r,         P.RHO_GH,  P.RHO_GSE],
        [r,         1.0,       P.RHO_CH,  P.RHO_CSE],
        [P.RHO_GH,  P.RHO_CH,  1.0,       P.RHO_HSE],
        [P.RHO_GSE, P.RHO_CSE, P.RHO_HSE, 1.0],
    ])
    if np.linalg.eigvalsh(R).min() <= 0:
        raise ValueError(f"trait correlation matrix not positive definite ({context})")
    return R


def perf_coefs(R: np.ndarray, include_h: bool,
               v_h: float = P.V_H) -> tuple[np.ndarray, float]:
    """Regression weights of y on (g, C[, H]) reproducing operational
    validities. Generalized: V_H is an argument (analysis 2)."""
    idx = [0, 1, 2] if include_h else [0, 1]
    v = np.array([P.V_G, P.V_C, v_h])[: len(idx)]
    b = np.linalg.solve(R[np.ix_(idx, idx)], v)
    return b, float(np.sqrt(1 - v @ b))


def build_pool(context: str, gen: np.random.Generator,
               n_pool: int = P.N_POOL) -> dict:
    """Draw an applicant pool. Draw order matches model 11 exactly, so with
    gen = module rng seeded P.SEED and contexts built in model 11's order the
    pools are identical to model 11's. Additionally stores R and eps_y so the
    performance regression can be re-solved for other V_H values."""
    R = trait_corr(context)
    minority = gen.random(n_pool) < P.P_MIN
    z = gen.multivariate_normal(np.zeros(4), R, size=n_pool)
    z[minority] -= np.array([P.D_G, P.D_C, P.D_H, P.D_SE])
    g, c, h, se = z.T

    eps_y = gen.normal(0, 1, n_pool)  # common shock so H-path toggle is comparable
    b_h, sd_h = perf_coefs(R, include_h=True)
    b_n, sd_n = perf_coefs(R, include_h=False)
    y_h = b_h[0] * g + b_h[1] * c + b_h[2] * h + sd_h * eps_y
    y_n = b_n[0] * g + b_n[1] * c + sd_n * eps_y

    return {
        "context": context, "minority": minority, "R": R,
        "g": g, "c": c, "h": h, "se": se, "eps_y": eps_y,
        "need": -(P.W_G_NEED * g + P.W_C_NEED * c),
        "y_hpath": y_h, "y_nohpath": y_n,
        # common random numbers, fixed across conditions and optimizer steps
        "u_fake": gen.random(n_pool), "u_rand": gen.random(n_pool),
        "eps_c": gen.normal(0, 1, n_pool), "eps_se": gen.normal(0, 1, n_pool),
        "eps_rand": gen.normal(0, 1, n_pool),
    }


def perf_with_vh(pool: dict, v_h: float) -> np.ndarray:
    """Re-solve the H-path performance regression for a different V_H on the
    same pool (same eps_y shock). y without the H path does not involve V_H."""
    b_h, sd_h = perf_coefs(pool["R"], include_h=True, v_h=v_h)
    return (b_h[0] * pool["g"] + b_h[1] * pool["c"] + b_h[2] * pool["h"]
            + sd_h * pool["eps_y"])


def solve_propensity(pool: dict, targets: dict,
                     p_fake: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Solve (gamma_0, gamma_g, gamma_C, gamma_H) so the pool reproduces the
    anchored point-biserial trait-faking correlations and the prevalence
    (identical to model 11)."""
    X = np.column_stack([pool["g"], pool["c"], pool["h"]])
    tvec = np.array([targets["r_gF"], targets["r_CF"], targets["r_HF"]])

    def moments(par):
        p = expit(par[0] + X @ par[1:])
        pbar = p.mean()
        sd_f = np.sqrt(pbar * (1 - pbar))
        r = np.array([np.cov(X[:, j], p)[0, 1] / (X[:, j].std() * sd_f)
                      for j in range(3)])
        return np.concatenate([[pbar - p_fake], r - tvec])

    x0 = np.concatenate([[np.log(p_fake / (1 - p_fake))], 2.5 * tvec])
    sol = least_squares(moments, x0=x0, xtol=1e-13)
    p = expit(sol.x[0] + X @ sol.x[1:])
    F = pool["u_fake"] < p
    resid = moments(sol.x)
    achieved = {"prevalence": p_fake + resid[0],
                "r_gF": tvec[0] + resid[1], "r_CF": tvec[1] + resid[2],
                "r_HF": tvec[2] + resid[3]}
    if np.abs(resid).max() > 0.005:
        print(f"  [warn] propensity targets not fully reproduced: {achieved}")
    return F, sol.x, achieved


def apply_faking(pool: dict, F: np.ndarray, theta: np.ndarray, sigma: float,
                 censor: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Observed C and SE. Generalized from model 11:
    - theta of length 3 = (m0, m_need, m_skill), skill shared (model 11 form);
      length 4 = (m0, m_need, m_skill_C, m_skill_SE), skill freed (analysis 5).
    - censor=True floors the inflation at 0 on both scales (analysis 1)."""
    if len(theta) == 4:
        m0, m_need, ms_c, ms_se = theta
    else:
        m0, m_need, ms_c = theta
        ms_se = ms_c
    infl_c = m0 + m_need * pool["need"] + ms_c * pool["g"] + sigma * pool["eps_c"]
    infl_se = P.S0_SE + ms_se * pool["g"] + sigma * pool["eps_se"]
    if censor:
        infl_c = np.maximum(infl_c, 0.0)
        infl_se = np.maximum(infl_se, 0.0)
    return pool["c"] + F * infl_c, pool["se"] + F * infl_se


def pool_moments(pool: dict, c_obs: np.ndarray, se_obs: np.ndarray) -> dict:
    g, c, se = pool["g"], pool["c"], pool["se"]
    return {
        "d_shift": float((c_obs.mean() - c.mean())
                         / np.sqrt((c_obs.var() + c.var()) / 2)),
        "r_gC": float(np.corrcoef(g, c_obs)[0, 1]),
        "delta_r_SE": float(np.corrcoef(g, se_obs)[0, 1]
                            - np.corrcoef(g, se)[0, 1]),
        "sd_ratio": float(c_obs.std() / c.std()),
    }


def calibrate(pool: dict, F: np.ndarray, sigma: float, target_r: float,
              censor: bool = False, free_skill: bool = False,
              sd_target: float = P.SD_RATIO_TARGET,
              ) -> tuple[np.ndarray, dict, dict, float]:
    """Minimum-distance WLS fit of the inflation parameters to the target
    moments (model 11's calibrate_trait_model, generalized). The censored
    objective is only piecewise smooth, so it gets multiple starts and the
    best minimum is kept; the uncensored shared-skill path uses model 11's
    single start and reproduces its fits exactly."""
    targets = {"d_shift": P.D_SHIFT, "r_gC": target_r,
               "delta_r_SE": P.DELTA_R_SE_G, "sd_ratio": sd_target}
    sqw = np.sqrt(np.array([P.MOMENT_WEIGHTS[k] for k in ACTIVE_MOMENTS]))

    def resid(theta):
        m = pool_moments(pool, *apply_faking(pool, F, theta, sigma, censor))
        return sqw * np.array([m[k] - targets[k] for k in ACTIVE_MOMENTS])

    base = [P.D_SHIFT / F.mean(), 1.0, 0.3]
    starts = [base]
    if censor:
        starts += [[1.0, 0.0, 0.0], [0.8, 3.0, 0.8], [1.5, 1.5, 0.5]]
    if free_skill:
        starts = [s + [s[-1]] for s in starts] + [[base[0], 1.0, 0.5, 0.0]]

    best = None
    for x0 in starts:
        fit = least_squares(resid, x0=x0, xtol=1e-12)
        if best is None or fit.cost < best.cost:
            best = fit
    achieved = pool_moments(pool, *apply_faking(pool, F, best.x, sigma, censor))
    return best.x, targets, achieved, float(2 * best.cost)


def calibrate_random(rho0: float, target_r: float, sigma: float,
                     p_fake: float) -> tuple[float, float]:
    """09's random-faking calibration (closed-form moments), as in model 11."""

    def moments(params):
        mu_f, b = params
        var_infl = p_fake * (mu_f**2 + b**2 + sigma**2) - (p_fake * mu_f) ** 2
        var_cobs = 1 - 2 * p_fake * b * rho0 + var_infl
        r_gc = (rho0 - p_fake * b) / np.sqrt(var_cobs)
        d = (p_fake * mu_f) / np.sqrt((1 + var_cobs) / 2)
        return [r_gc - target_r, d - P.D_SHIFT]

    mu_f, b = fsolve(moments, x0=[P.D_SHIFT / p_fake, max(0.0, rho0 - target_r)])
    return float(mu_f), float(b)


def hire_stats(pool: dict, F: np.ndarray, c_obs: np.ndarray, sr: float,
               y_h: np.ndarray | None = None) -> dict:
    """Top-down selection outcomes (model 11). Generalized: the H-path
    performance vector can be overridden (analysis 2's re-solved V_H)."""
    g, c, h, minority = pool["g"], pool["c"], pool["h"], pool["minority"]
    y_h = pool["y_hpath"] if y_h is None else y_h
    comp_honest = P.W_COMPOSITE * g + (1 - P.W_COMPOSITE) * c
    comp_faked = (P.W_COMPOSITE * g
                  + (1 - P.W_COMPOSITE) * (c_obs - c_obs.mean()) / c_obs.std())

    n_sel = int(sr * len(g))
    sel_h = np.argsort(-comp_honest)[:n_sel]
    sel_f = np.argsort(-comp_faked)[:n_sel]
    faked_in = np.setdiff1d(sel_f, sel_h, assume_unique=True)
    displaced = np.setdiff1d(sel_h, sel_f, assume_unique=True)

    def air(sel):
        pm = minority[sel].mean()
        return (pm / P.P_MIN) / ((1 - pm) / (1 - P.P_MIN))

    perf_hh, perf_hn = y_h[sel_h].mean(), pool["y_nohpath"][sel_h].mean()
    perf_fh, perf_fn = y_h[sel_f].mean(), pool["y_nohpath"][sel_f].mean()
    return {
        "pct_hires_changed": 100 * len(faked_in) / n_sel,
        "perf_honest_Hpath": perf_hh, "perf_faked_Hpath": perf_fh,
        "perf_loss_pct_Hpath": 100 * (perf_hh - perf_fh) / perf_hh,
        "perf_honest_noH": perf_hn, "perf_faked_noH": perf_fn,
        "perf_loss_pct_noH": 100 * (perf_hn - perf_fn) / perf_hn,
        "AIR_honest": air(sel_h), "AIR_faked": air(sel_f),
        "mean_H_hires_honest": h[sel_h].mean(), "mean_H_hires_faked": h[sel_f].mean(),
        "mean_C_true_hires_faked": c[sel_f].mean(),
        "mean_H_faked_in": h[faked_in].mean(), "mean_H_displaced": h[displaced].mean(),
        "pct_fakers_among_faked_in": 100 * F[faked_in].mean(),
    }


def random_benchmark(pool: dict, sigma: float, p_fake: float,
                     ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """09-style random-faking benchmark on the same pool (as in model 11)."""
    ctx = pool["context"]
    mu_f, b = calibrate_random(P.RHO_GC_BASE[ctx], P.TARGET_R_APPLICANT[ctx],
                               sigma, p_fake)
    F_r = pool["u_rand"] < p_fake
    c_obs_r = pool["c"] + F_r * (mu_f - b * pool["g"] + sigma * pool["eps_rand"])
    return F_r, c_obs_r, mu_f, b


# ------------------------------------------------------------ shared setup
print("Building pools (identical to model 11's: same seed, same draw order)...")
POOLS = {ctx: build_pool(ctx, rng) for ctx in CONTEXTS}

# faking indicator per (context, param set), cached — CRN across analyses
F_CACHE, GAMMA_CACHE = {}, {}
for ctx in CONTEXTS:
    for ps in ("field_ns0", "field", "lab"):
        F, gam, _ = solve_propensity(POOLS[ctx], P.PARAM_SETS[ps], P.P_FAKE)
        F_CACHE[(ctx, ps)], GAMMA_CACHE[(ctx, ps)] = F, gam

# baseline (uncensored, shared-skill, field_ns0) calibration per context —
# reference point for analyses 1, 2, and 5
BASE = {}
for ctx in CONTEXTS:
    F = F_CACHE[(ctx, PRIMARY)]
    theta, targets, achieved, dist = calibrate(
        POOLS[ctx], F, P.SIGMA_INFL, P.TARGET_R_APPLICANT[ctx])
    c_obs, _ = apply_faking(POOLS[ctx], F, theta, P.SIGMA_INFL)
    BASE[ctx] = {"theta": theta, "achieved": achieved, "dist": dist,
                 "c_obs": c_obs,
                 "stats": {sr: hire_stats(POOLS[ctx], F, c_obs, sr)
                           for sr in P.SRS}}


def calib_row(ctx, ps, variant, theta, targets, achieved, dist):
    row = {"context": ctx, "param_set": ps, "variant": variant,
           "m0": round(float(theta[0]), 3), "m_need": round(float(theta[1]), 3)}
    if len(theta) == 4:
        row["m_skill_C"] = round(float(theta[2]), 3)
        row["m_skill_SE"] = round(float(theta[3]), 3)
    else:
        row["m_skill"] = round(float(theta[2]), 3)
    row["weighted_distance"] = round(dist, 2)
    for k in MOMENT_KEYS:
        row[f"target_{k}"] = (round(targets[k], 3) if k in ACTIVE_MOMENTS
                              else np.nan)
        row[f"achieved_{k}"] = round(achieved[k], 3)
    return row


def outcome_rows(ctx, ps, variant, pool, F, c_obs, with_random=True):
    rows = []
    if with_random:
        F_r, c_obs_r, _, _ = random_benchmark(pool, P.SIGMA_INFL, P.P_FAKE)
    for sr in P.SRS:
        t = hire_stats(pool, F, c_obs, sr)
        row = {"context": ctx, "param_set": ps, "variant": variant,
               "selection_ratio": sr}
        row.update({k: round(v, 3) for k, v in t.items()})
        if with_random:
            r = hire_stats(pool, F_r, c_obs_r, sr)
            row.update({
                "rand_pct_hires_changed": round(r["pct_hires_changed"], 3),
                "rand_perf_loss_pct_Hpath": round(r["perf_loss_pct_Hpath"], 3),
                "rand_AIR_faked": round(r["AIR_faked"], 3),
                "rand_mean_H_faked_in": round(r["mean_H_faked_in"], 3),
                "extra_perf_loss_vs_random_Hpath":
                    round(t["perf_loss_pct_Hpath"] - r["perf_loss_pct_Hpath"], 3),
                "H_gap_faked_in_vs_random":
                    round(t["mean_H_faked_in"] - r["mean_H_faked_in"], 3),
            })
        rows.append(row)
    return rows


# ============================================================ (1) censored refit
print("\n[1] Censored-inflation refit (infl+ = max(infl, 0), C and SE scales)")
cens_calib_rows, cens_out_rows = [], []
CENS_CELLS = ([(ctx, PRIMARY) for ctx in CONTEXTS]
              + [("Non-US", "field"), ("Non-US", "lab")])
for ctx, ps in CENS_CELLS:
    pool, F = POOLS[ctx], F_CACHE[(ctx, ps)]
    # uncensored counterpart on the same pool/F, for direct comparison
    th_u, tg, ach_u, d_u = calibrate(pool, F, P.SIGMA_INFL,
                                     P.TARGET_R_APPLICANT[ctx])
    th_c, _, ach_c, d_c = calibrate(pool, F, P.SIGMA_INFL,
                                    P.TARGET_R_APPLICANT[ctx], censor=True)
    # share of fakers actually hitting the zero floor under the censored fit
    raw_c = (th_c[0] + th_c[1] * pool["need"] + th_c[2] * pool["g"]
             + P.SIGMA_INFL * pool["eps_c"])
    raw_se = P.S0_SE + th_c[2] * pool["g"] + P.SIGMA_INFL * pool["eps_se"]
    row = calib_row(ctx, ps, "censored", th_c, tg, ach_c, d_c)
    row["pct_fakers_floored_C"] = round(100 * float((raw_c[F] < 0).mean()), 1)
    row["pct_fakers_floored_SE"] = round(100 * float((raw_se[F] < 0).mean()), 1)
    row["uncensored_distance"] = round(d_u, 2)
    row["uncensored_achieved_r_gC"] = round(ach_u["r_gC"], 3)
    cens_calib_rows.append(row)
    c_obs_c, _ = apply_faking(pool, F, th_c, P.SIGMA_INFL, censor=True)
    cens_out_rows.extend(outcome_rows(ctx, ps, "censored", pool, F, c_obs_c))
    print(f"  {ctx:7s} {ps:9s} dist {d_u:6.2f} -> {d_c:6.2f} censored | "
          f"achieved r_gC {ach_u['r_gC']:+.3f} -> {ach_c['r_gC']:+.3f} "
          f"(target {P.TARGET_R_APPLICANT[ctx]:+.3f})")

# ============================================================ (2) V_H sensitivity
print("\n[2] V_H sensitivity {.15, .18, .41} (field_ns0, uncensored)")
vh_rows = []
for ctx in ("US", "Non-US"):
    pool, F = POOLS[ctx], F_CACHE[(ctx, PRIMARY)]
    c_obs = BASE[ctx]["c_obs"]  # calibration does not involve y: theta unchanged
    F_r, c_obs_r, _, _ = random_benchmark(pool, P.SIGMA_INFL, P.P_FAKE)
    for v_h in V_H_GRID:
        y_h = perf_with_vh(pool, v_h)
        for sr in P.SRS:
            t = hire_stats(pool, F, c_obs, sr, y_h=y_h)
            r = hire_stats(pool, F_r, c_obs_r, sr, y_h=y_h)
            vh_rows.append({
                "context": ctx, "param_set": PRIMARY, "V_H": v_h,
                "selection_ratio": sr,
                "perf_loss_pct_Hpath": round(t["perf_loss_pct_Hpath"], 3),
                "rand_perf_loss_pct_Hpath": round(r["perf_loss_pct_Hpath"], 3),
                "extra_perf_loss_vs_random_Hpath":
                    round(t["perf_loss_pct_Hpath"] - r["perf_loss_pct_Hpath"], 3),
                "perf_loss_pct_noH": round(t["perf_loss_pct_noH"], 3),
                "pct_hires_changed": round(t["pct_hires_changed"], 3),
                "mean_H_faked_in": round(t["mean_H_faked_in"], 3),
            })
print(pd.DataFrame(vh_rows).to_string(index=False))

# ============================================================ (3) Monte Carlo SEs
MC_STATS = ("pct_hires_changed", "perf_loss_pct_Hpath", "AIR_faked",
            "mean_H_faked_in", "mean_H_displaced")


def one_replication(seed: int) -> tuple[list[dict], list[dict]]:
    """Fresh pool per context, full recalibration (propensity + WLS)."""
    gen = np.random.default_rng(seed)
    o_rows, p_rows = [], []
    for ctx in CONTEXTS:
        pool = build_pool(ctx, gen)
        F, _, _ = solve_propensity(pool, P.PARAM_SETS[PRIMARY], P.P_FAKE)
        theta, _, _, dist = calibrate(pool, F, P.SIGMA_INFL,
                                      P.TARGET_R_APPLICANT[ctx])
        c_obs, _ = apply_faking(pool, F, theta, P.SIGMA_INFL)
        p_rows.append({"seed": seed, "context": ctx,
                       "m0": float(theta[0]), "m_need": float(theta[1]),
                       "m_skill": float(theta[2]),
                       "weighted_distance": dist})
        for sr in P.SRS:
            t = hire_stats(pool, F, c_obs, sr)
            o_rows.append({"seed": seed, "context": ctx, "selection_ratio": sr,
                           **{k: float(t[k]) for k in MC_STATS}})
    return o_rows, p_rows


print(f"\n[3] Monte Carlo replications (field_ns0, seeds "
      f"{MC_SEEDS[0]}..{MC_SEEDS[-1]}, fresh pools, full recalibration)")
t0 = time.time()
mc_o, mc_p = one_replication(MC_SEEDS[0])
per_rep = time.time() - t0
seeds = MC_SEEDS
if per_rep * len(MC_SEEDS) > 30 * 60:
    seeds = MC_SEEDS[:MC_FALLBACK_N]
    print(f"  projected {per_rep * len(MC_SEEDS) / 60:.0f} min for "
          f"{len(MC_SEEDS)} reps > 30 min -> dropping to {MC_FALLBACK_N}")
print(f"  rep 1 took {per_rep:.0f}s; running {len(seeds)} total")
mc_out_rows, mc_par_rows = list(mc_o), list(mc_p)
for s in seeds[1:]:
    o, p = one_replication(s)
    mc_out_rows.extend(o)
    mc_par_rows.extend(p)
print(f"  done in {(time.time() - t0) / 60:.1f} min")

mc_out_df = pd.DataFrame(mc_out_rows)
mc_par_df = pd.DataFrame(mc_par_rows)
R_ACTUAL = len(seeds)


def mc_summary(df: pd.DataFrame, group: list[str], stats: tuple) -> pd.DataFrame:
    rows = []
    for keys, gdf in df.groupby(group, sort=False):
        row = dict(zip(group, keys if isinstance(keys, tuple) else (keys,)))
        row["n_reps"] = len(gdf)
        for s in stats:
            row[f"{s}_mc_mean"] = round(gdf[s].mean(), 4)
            row[f"{s}_mc_se"] = round(gdf[s].std(ddof=1) / np.sqrt(len(gdf)), 4)
        rows.append(row)
    return pd.DataFrame(rows)


mc_out_sum = mc_summary(mc_out_df, ["context", "selection_ratio"], MC_STATS)
mc_par_sum = mc_summary(mc_par_df, ["context"],
                        ("m0", "m_need", "m_skill", "weighted_distance"))
print(mc_out_sum.to_string(index=False))
print(mc_par_sum.to_string(index=False))

# ============================================================ (4) r_CF profile grid
print("\n[4] r_CF profile grid (r_gF = -.19, r_HF = -.04 fixed)")
grid_rows = []
for ctx in CONTEXTS:
    pool = POOLS[ctx]
    for r_cf in R_CF_GRID:
        targets = {"r_gF": R_GF_FIXED, "r_CF": r_cf, "r_HF": R_HF_FIXED}
        F, _, ach = solve_propensity(pool, targets, P.P_FAKE)
        theta, tg, achm, dist = calibrate(pool, F, P.SIGMA_INFL,
                                          P.TARGET_R_APPLICANT[ctx])
        grid_rows.append({"context": ctx, "r_CF": r_cf,
                          "achieved_r_CF": round(ach["r_CF"], 3),
                          "weighted_distance": round(dist, 3),
                          "m0": round(float(theta[0]), 3),
                          "m_need": round(float(theta[1]), 3),
                          "m_skill": round(float(theta[2]), 3),
                          "achieved_d_shift": round(achm["d_shift"], 3),
                          "achieved_r_gC": round(achm["r_gC"], 3),
                          "achieved_sd_ratio": round(achm["sd_ratio"], 3)})
grid_df = pd.DataFrame(grid_rows)


def acceptance_region(xs: np.ndarray, ds: np.ndarray,
                      thr: float) -> tuple[str, str]:
    """Interpolated crossing points of the distance profile with a threshold;
    open-ended at the grid edge if the profile never crosses."""
    i0 = int(np.argmin(ds))

    def edge(direction):
        j = i0
        while 0 <= j + direction < len(xs) and ds[j + direction] <= thr:
            j += direction
        if j + direction < 0 or j + direction >= len(xs):
            return f"<{xs[0]:+.2f}" if direction < 0 else f">{xs[-1]:+.2f}"
        x1, x2 = xs[j], xs[j + direction]
        d1, d2 = ds[j], ds[j + direction]
        return f"{x1 + (thr - d1) * (x2 - x1) / (d2 - d1):+.3f}"

    return edge(-1), edge(+1)


region_rows = []
for ctx in CONTEXTS:
    sub = grid_df[grid_df["context"] == ctx]
    xs = sub["r_CF"].to_numpy()
    ds = sub["weighted_distance"].to_numpy()
    dmin, xmin = float(ds.min()), float(xs[np.argmin(ds)])
    lo2, hi2 = acceptance_region(xs, ds, 2 * dmin)
    loc, hic = acceptance_region(xs, ds, dmin + 3.84)
    region_rows.append({
        "context": ctx, "r_CF_at_min": xmin, "min_distance": round(dmin, 3),
        "region_2x_min_lo": lo2, "region_2x_min_hi": hi2,
        "region_chi2_3.84_lo": loc, "region_chi2_3.84_hi": hic,
        "note": "heuristic 1-df chi-square analogy, not formal inference"})
    print(f"  {ctx:7s} min dist {dmin:6.3f} at r_CF {xmin:+.2f} | "
          f"2x-min region [{lo2}, {hi2}] | +3.84 region [{loc}, {hic}]")
region_df = pd.DataFrame(region_rows)

# ============================================================ (5) free m_skill
print("\n[5] Freed skill slopes: separate m_skill_C / m_skill_SE "
      "(just-identified: 4 params, 4 moments)")
free_calib_rows, free_out_rows = [], []
for ctx in CONTEXTS:
    pool, F = POOLS[ctx], F_CACHE[(ctx, PRIMARY)]
    th_f, tg, ach_f, d_f = calibrate(pool, F, P.SIGMA_INFL,
                                     P.TARGET_R_APPLICANT[ctx], free_skill=True)
    row = calib_row(ctx, PRIMARY, "free_skill", th_f, tg, ach_f, d_f)
    row["shared_m_skill"] = round(float(BASE[ctx]["theta"][2]), 3)
    row["shared_distance"] = round(BASE[ctx]["dist"], 2)
    free_calib_rows.append(row)
    c_obs_f, _ = apply_faking(pool, F, th_f, P.SIGMA_INFL)
    for r in outcome_rows(ctx, PRIMARY, "free_skill", pool, F, c_obs_f,
                          with_random=False):
        sr = r["selection_ratio"]
        base_t = BASE[ctx]["stats"][sr]
        r["shared_pct_hires_changed"] = round(base_t["pct_hires_changed"], 3)
        r["shared_perf_loss_pct_Hpath"] = round(base_t["perf_loss_pct_Hpath"], 3)
        r["delta_perf_loss_vs_shared"] = round(
            r["perf_loss_pct_Hpath"] - base_t["perf_loss_pct_Hpath"], 3)
        free_out_rows.append(r)
    print(f"  {ctx:7s} m_skill_C {th_f[2]:+.3f} vs m_skill_SE {th_f[3]:+.3f} "
          f"(shared fit: {BASE[ctx]['theta'][2]:+.3f}) | "
          f"dist {BASE[ctx]['dist']:.2f} -> {d_f:.4f}")

# ============================================================ write output
cens_calib_df = pd.DataFrame(cens_calib_rows)
cens_out_df = pd.DataFrame(cens_out_rows)
vh_df = pd.DataFrame(vh_rows)
free_calib_df = pd.DataFrame(free_calib_rows)
free_out_df = pd.DataFrame(free_out_rows)

with open(OUT / f"study2_validation_results{P.HH_SUFFIX}.csv", "w", newline="") as f:
    f.write("# Study 2 validation batch (13_study2_validation.py) on the "
            "anchored v2 trait model (11_study2_trait_model.py);\n"
            "# primary spec field_ns0 (r_gF=-.19, r_CF=0, r_HF=0), "
            "prevalence .40, sigma .50, n=200,000, seed 2026 "
            "(pools identical to model 11's)\n")
    f.write("# (1a) Censored-inflation refit: infl+ = max(infl,0) on C and "
            "SE scales; uncensored_* columns = same pool/F without the "
            "floor; pct_fakers_floored_* = share of fakers at the zero "
            "floor under the censored fit\n")
    cens_calib_df.to_csv(f, index=False)
    f.write("\n# (1b) Selection outcomes under the censored fits, SR "
            ".10/.30; rand_* = 09-style random-faking benchmark "
            "(uncensored, as in model 11) on the same pool\n")
    cens_out_df.to_csv(f, index=False)
    f.write("\n# (2) V_H sensitivity {.15, .18, .41} (Van Iddekinge et al. "
            "2012 corrected vs Ones et al. 1993), field_ns0 uncensored; "
            "only the performance regression is re-solved - calibration "
            "does not involve y, so theta is unchanged\n")
    vh_df.to_csv(f, index=False)
    f.write(f"\n# (3a) Monte Carlo outcome summaries: R={R_ACTUAL} "
            f"replications, seeds {seeds[0]}..{seeds[-1]}, fresh 200k pool "
            "and full recalibration (propensity solve + WLS) per "
            "replication (Morris, White, & Crowther 2019); "
            "mc_se = SD/sqrt(R)\n")
    mc_out_sum.to_csv(f, index=False)
    f.write("\n# (3b) Monte Carlo calibrated-parameter summaries "
            "(same replications)\n")
    mc_par_sum.to_csv(f, index=False)
    f.write("\n# (3c) Per-replication raw values (outcomes)\n")
    mc_out_df.round(4).to_csv(f, index=False)
    f.write("\n# (3d) Per-replication raw values (calibrated parameters)\n")
    mc_par_df.round(4).to_csv(f, index=False)
    f.write("\n# (4a) r_CF profile grid: r_gF=-.19, r_HF=-.04 fixed (field "
            "anchors), r_CF swept; weighted_distance = minimized WLS "
            "criterion after full propensity solve + refit per point\n")
    grid_df.to_csv(f, index=False)
    f.write("\n# (4b) Acceptance regions for r_CF: distance <= 2x minimum "
            "and <= minimum+3.84 (HEURISTIC 1-df chi-square analogy, not "
            "formal inference); < or > marks an open end at the grid edge\n")
    region_df.to_csv(f, index=False)
    f.write("\n# (5a) Freed skill slopes: separate m_skill_C / m_skill_SE "
            "(4 free params, 4 moments, just-identified); shared_* columns "
            "= model 11's shared-skill fit on the same pool/F\n")
    free_calib_df.to_csv(f, index=False)
    f.write("\n# (5b) Selection outcomes under the freed-skill fit vs the "
            "shared-skill fit (field_ns0)\n")
    free_out_df.to_csv(f, index=False)

print(f"\nSaved study2_validation_results{P.HH_SUFFIX}.csv")
