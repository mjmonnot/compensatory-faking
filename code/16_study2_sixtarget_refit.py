"""Six-target recalibration campaign (triage rows 5b, 6b-6e) on the expanded
Study 2 model of 12_study2_expanded_model.py.

Background: model 12 fits the primary linear targeted-inflation form against
SIX empirical targets (C shift d = .45, ES shift d = .44, A shift d = .16,
applicant r(g, C_obs) by context, Self-Esteem-GMA shift +.17, applicant/honest
C SD ratio 0.86). The two functional-form relaxations (inflation censored at
zero; freed skill slopes) previously lived in 13_study2_validation.py, which
fits the older four-trait model against only FOUR of those targets — so the
"relaxed forms fit better" comparison was not apples-to-apples (reviewer item
6b). This script refits both relaxations inside model 12's six-target
framework and adds the campaign's other analyses.

Machinery is copied from 12_study2_expanded_model.py (the same convention 13
used with 11) with these generalizations, so model 12 remains untouched and
its outputs reproducible:
  - apply_faking gains censor (inflation floored at 0 on all inflated scales)
    and free_skill (theta length 5: m0_C, m0_ES, m_need, m_skill_CES,
    m_skill_SE — the C and ES scales share one skill slope, the SE scale gets
    its own, generalizing 13's C-vs-SE split; 5 free parameters against the
    5 fittable targets, since variant T leaves the A target at zero by
    construction).
  - calibrate gains multiple starts for the piecewise-smooth censored
    objective, and p_fake flows through the propensity solve for the
    prevalence sweep.

Pools are built with the module rng seeded P.SEED in model 12's draw order,
so the three base context pools are bit-identical to model 12's.

Analyses (numbers = triage rows):
(6b) SIX-TARGET REFIT of linear / censored / free-skill targeted inflation,
     field_ns0, all contexts: weighted distances (all include the constant
     ~3.5 Agreeableness penalty), achieved targets, selection outcomes.
(6c) ANCHOR-SET OUTCOME GRID: selection outcomes for field_ns0, field, and
     lab under the six-target linear fit in every context (model 12 reported
     outcomes for field_ns0 only). Uniform reporting rule, stated ex ante:
     outcomes are reported for every anchor set in every context; a
     fit-supported flag marks sets within 2x the context-minimum distance
     (the same heuristic threshold used in the r_CF profile analysis).
     Implausibility of implied losses is then an argument, not a reporting
     filter.
(6d) RHO GRID: RHO_CSE and RHO_HSE (both PROVISIONAL single-source values)
     varied +/-.10 around their defaults, 3x3 grid, field_ns0 linear refit
     per context: do distances and outcome directions hold?
(6e) ES-GMA FALSIFICATION: the model-implied applicant shift in the
     Emotional Stability-GMA correlation, r(g, ES_obs) - r(g, ES_honest),
     computed from the fitted pools. This quantity was never a fitting
     target; Study 1 measured it independently (Neuroticism contrast,
     Table 1: delta = -.013, ns => ES-side benchmark +.013 with its 95% CI
     rebuilt from the contrast z). Held-out comparison, R02's test.
(5b) PREVALENCE SWEEP: p_fake in P_FAKE_SENS = {.30, .50, .20} (within-person
     range endpoints plus the Griffith & Converse, 2011 synthesis lower
     bound, added 2026-08-25), field_ns0 linear, full propensity re-solve +
     refit per prevalence, selection outcomes vs the 40% primary.

Fit failures are findings, not errors; nothing is forced.

Outputs: output/study2_sixtarget_refit.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit

import study2_params as P

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
rng = np.random.default_rng(P.SEED)

MOMENT_KEYS = ("d_shift", "d_shift_ES", "d_shift_A",
               "r_gC", "delta_r_SE", "sd_ratio")
MOMENT_WEIGHTS = {**P.MOMENT_WEIGHTS, **P.MOMENT_WEIGHTS_12}

CONTEXTS = ("US", "Pooled", "Non-US")
PRIMARY = P.PRIMARY_PARAM_SET_12          # field_ns0
ANCHOR_SETS = P.PARAM_SETS_12             # field_ns0, field, lab

RHO_CSE_GRID = (P.RHO_CSE - 0.10, P.RHO_CSE, P.RHO_CSE + 0.10)
RHO_HSE_GRID = (P.RHO_HSE - 0.10, P.RHO_HSE, P.RHO_HSE + 0.10)


# ------------------------------------------------------------ model 12 machinery
def trait_corr(context: str, rho_cse: float = P.RHO_CSE,
               rho_hse: float = P.RHO_HSE) -> np.ndarray:
    """Latent correlation matrix, order (g, C, ES, A, H, SE). Generalized:
    RHO_CSE / RHO_HSE overridable for the 6d grid; fails loudly if not PD."""
    r = P.RHO_GC_BASE[context]
    R = np.array([
        # g         C          ES          A          H          SE
        [1.0,       r,         P.RHO_GES,  P.RHO_GA,  P.RHO_GH,  P.RHO_GSE],
        [r,         1.0,       P.RHO_CES,  P.RHO_CA,  P.RHO_CH,  rho_cse],
        [P.RHO_GES, P.RHO_CES, 1.0,        P.RHO_ESA, P.RHO_ESH, P.RHO_ESSE],
        [P.RHO_GA,  P.RHO_CA,  P.RHO_ESA,  1.0,       P.RHO_AH,  P.RHO_ASE],
        [P.RHO_GH,  P.RHO_CH,  P.RHO_ESH,  P.RHO_AH,  1.0,       rho_hse],
        [P.RHO_GSE, rho_cse,   P.RHO_ESSE, P.RHO_ASE, rho_hse,   1.0],
    ])
    if np.linalg.eigvalsh(R).min() <= 0:
        raise ValueError(
            f"correlation matrix not PD ({context}, rho_CSE={rho_cse}, "
            f"rho_HSE={rho_hse}; min eig {np.linalg.eigvalsh(R).min():.4f})")
    return R


def perf_coefs(R: np.ndarray, include_h: bool) -> tuple[np.ndarray, float]:
    idx = [0, 1, 4] if include_h else [0, 1]
    v = np.array([P.V_G, P.V_C, P.V_H])[: len(idx)]
    b = np.linalg.solve(R[np.ix_(idx, idx)], v)
    return b, float(np.sqrt(1 - v @ b))


def build_pool(context: str, rho_cse: float = P.RHO_CSE,
               rho_hse: float = P.RHO_HSE, n_pool: int = P.N_POOL) -> dict:
    """Model 12's pool builder (base pools bit-identical: same rng order)."""
    R = trait_corr(context, rho_cse, rho_hse)
    minority = rng.random(n_pool) < P.P_MIN
    z = rng.multivariate_normal(np.zeros(6), R, size=n_pool)
    z[minority] -= np.array([P.D_G, P.D_C, P.D_ES, P.D_A, P.D_H, P.D_SE])
    g, c, es, a, h, se = z.T

    eps_y = rng.normal(0, 1, n_pool)
    b_h, sd_h = perf_coefs(R, include_h=True)
    b_n, sd_n = perf_coefs(R, include_h=False)
    y_h = b_h[0] * g + b_h[1] * c + b_h[2] * h + sd_h * eps_y
    y_n = b_n[0] * g + b_n[1] * c + sd_n * eps_y

    return {
        "context": context, "minority": minority,
        "g": g, "c": c, "es": es, "a": a, "h": h, "se": se,
        "need": -(P.W_G_NEED * g + P.W_C_NEED * c),
        "y_hpath": y_h, "y_nohpath": y_n,
        "u_fake": rng.random(n_pool),
        "eps_c": rng.normal(0, 1, n_pool), "eps_es": rng.normal(0, 1, n_pool),
        "eps_a": rng.normal(0, 1, n_pool), "eps_se": rng.normal(0, 1, n_pool),
    }


def solve_propensity(pool: dict, targets: dict,
                     p_fake: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Identical to model 12."""
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
                 censor: bool = False) -> dict:
    """Variant T (targeted inflation) generalized from model 12:
    - theta length 4 = (m0_C, m0_ES, m_need, m_skill), skill shared (12's form)
    - theta length 5 = (m0_C, m0_ES, m_need, m_skill_CES, m_skill_SE), the
      C and ES scales sharing one skill slope, SE freed
    - censor=True floors the inflation at 0 on all inflated scales."""
    need, g = pool["need"], pool["g"]
    if len(theta) == 5:
        m0_c, m0_es, m_need, ms_ces, ms_se = theta
    else:
        m0_c, m0_es, m_need, ms_ces = theta
        ms_se = ms_ces
    infl_c = m0_c + m_need * need + ms_ces * g + sigma * pool["eps_c"]
    infl_es = m0_es + m_need * need + ms_ces * g + sigma * pool["eps_es"]
    infl_se = P.S0_SE + ms_se * g + sigma * pool["eps_se"]
    if censor:
        infl_c = np.maximum(infl_c, 0.0)
        infl_es = np.maximum(infl_es, 0.0)
        infl_se = np.maximum(infl_se, 0.0)
    return {
        "c_obs": pool["c"] + F * infl_c,
        "es_obs": pool["es"] + F * infl_es,
        "a_obs": pool["a"],
        "se_obs": pool["se"] + F * infl_se,
    }


def d_shift(obs: np.ndarray, lat: np.ndarray) -> float:
    return float((obs.mean() - lat.mean()) / np.sqrt((obs.var() + lat.var()) / 2))


def pool_moments(pool: dict, scales: dict) -> dict:
    g = pool["g"]
    return {
        "d_shift": d_shift(scales["c_obs"], pool["c"]),
        "d_shift_ES": d_shift(scales["es_obs"], pool["es"]),
        "d_shift_A": d_shift(scales["a_obs"], pool["a"]),
        "r_gC": float(np.corrcoef(g, scales["c_obs"])[0, 1]),
        "delta_r_SE": float(np.corrcoef(g, scales["se_obs"])[0, 1]
                            - np.corrcoef(g, pool["se"])[0, 1]),
        "sd_ratio": float(scales["c_obs"].std() / pool["c"].std()),
    }


def es_gap(pool: dict, scales: dict) -> float:
    """Model-implied applicant shift in the ES-GMA correlation (6e). Never a
    fitting target; Study 1's Neuroticism contrast is the held-out benchmark."""
    return float(np.corrcoef(pool["g"], scales["es_obs"])[0, 1]
                 - np.corrcoef(pool["g"], pool["es"])[0, 1])


def calibrate(pool: dict, F: np.ndarray, sigma: float, target_r: float,
              censor: bool = False, free_skill: bool = False,
              ) -> tuple[np.ndarray, dict, dict, float]:
    """WLS minimum-distance fit against all six targets (model 12's
    calibrate, variant T, generalized). Censored objective is piecewise
    smooth -> multiple starts, best minimum kept; the plain path uses model
    12's single start and reproduces its fits exactly."""
    targets = {"d_shift": P.D_SHIFT, "d_shift_ES": P.D_SHIFT_ES,
               "d_shift_A": P.D_SHIFT_A, "r_gC": target_r,
               "delta_r_SE": P.DELTA_R_SE_G, "sd_ratio": P.SD_RATIO_TARGET}
    sqw = np.sqrt(np.array([MOMENT_WEIGHTS[k] for k in MOMENT_KEYS]))

    def resid(theta):
        m = pool_moments(pool, apply_faking(pool, F, theta, sigma, censor))
        return sqw * np.array([m[k] - targets[k] for k in MOMENT_KEYS])

    base = [P.D_SHIFT / F.mean(), P.D_SHIFT_ES / F.mean(), 1.0, 0.3]
    starts = [base]
    if censor:
        starts += [[1.0, 1.0, 0.0, 0.0], [0.8, 0.8, 3.0, 0.8],
                   [1.5, 1.5, 1.5, 0.5]]
    if free_skill:
        starts = [s + [s[-1]] for s in starts] + [base[:3] + [0.5, 0.0]]

    best = None
    for x0 in starts:
        fit = least_squares(resid, x0=x0, xtol=1e-12)
        if best is None or fit.cost < best.cost:
            best = fit
    achieved = pool_moments(pool, apply_faking(pool, F, best.x, sigma, censor))
    return best.x, targets, achieved, float(2 * best.cost)


def hire_stats(pool: dict, F: np.ndarray, c_obs: np.ndarray, sr: float) -> dict:
    """Identical to model 12."""
    g, c, h, minority = pool["g"], pool["c"], pool["h"], pool["minority"]
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

    perf_hh = pool["y_hpath"][sel_h].mean()
    perf_fh = pool["y_hpath"][sel_f].mean()
    return {
        "pct_hires_changed": 100 * len(faked_in) / n_sel,
        "perf_honest_Hpath": perf_hh, "perf_faked_Hpath": perf_fh,
        "perf_loss_pct_Hpath": 100 * (perf_hh - perf_fh) / perf_hh,
        "AIR_honest": air(sel_h), "AIR_faked": air(sel_f),
        "mean_H_faked_in": h[faked_in].mean(),
        "mean_H_displaced": h[displaced].mean(),
        "H_gap_faked_in_vs_displaced": h[faked_in].mean() - h[displaced].mean(),
    }


def calib_row(ctx, ps, form, theta, targets, achieved, dist, extra=None):
    names = {4: ("m0_C", "m0_ES", "m_need", "m_skill"),
             5: ("m0_C", "m0_ES", "m_need", "m_skill_CES", "m_skill_SE")}
    row = {"context": ctx, "param_set": ps, "form": form,
           "weighted_distance": round(dist, 2),
           "distance_excl_A_target": round(
               dist - MOMENT_WEIGHTS["d_shift_A"]
               * (achieved["d_shift_A"] - targets["d_shift_A"]) ** 2, 2)}
    for n_, v in zip(names[len(theta)], theta):
        row[n_] = round(float(v), 3)
    for k in MOMENT_KEYS:
        row[f"target_{k}"] = round(targets[k], 3)
        row[f"achieved_{k}"] = round(achieved[k], 3)
    if extra:
        row.update(extra)
    return row


def outcome_row(ctx, ps, form, pool, F, scales, sr):
    s = hire_stats(pool, F, scales["c_obs"], sr)
    return {"context": ctx, "param_set": ps, "form": form,
            "selection_ratio": sr,
            **{k: round(v, 3) for k, v in s.items()}}


# ------------------------------------------------------------ base pools + F
print("Building base pools (bit-identical to model 12's)...")
POOLS = {ctx: build_pool(ctx) for ctx in CONTEXTS}
F_CACHE = {}
for ctx in CONTEXTS:
    for ps in ANCHOR_SETS:
        F, _, _ = solve_propensity(POOLS[ctx], P.PARAM_SETS[ps], P.P_FAKE)
        F_CACHE[(ctx, ps)] = F

# ============================================================ (6b) six-target refit
print("\n[6b] Six-target refit: linear / censored / free-skill "
      "(field_ns0, variant T)")
refit_calib, refit_out = [], []
FITS = {}   # (ctx, form) -> (theta, scales) for 6e
for ctx in CONTEXTS:
    pool, F = POOLS[ctx], F_CACHE[(ctx, PRIMARY)]
    tr = P.TARGET_R_APPLICANT[ctx]
    for form, kw in (("linear", {}), ("censored", {"censor": True}),
                     ("free_skill", {"free_skill": True})):
        theta, tg, ach, dist = calibrate(pool, F, P.SIGMA_INFL, tr, **kw)
        scales = apply_faking(pool, F, theta, P.SIGMA_INFL,
                              censor=kw.get("censor", False))
        FITS[(ctx, form)] = (theta, scales)
        extra = {}
        if form == "censored":
            raw_c = (theta[0] + theta[2] * pool["need"] + theta[3] * pool["g"]
                     + P.SIGMA_INFL * pool["eps_c"])
            extra["pct_fakers_floored_C"] = round(
                100 * float((raw_c[F] < 0).mean()), 1)
        refit_calib.append(calib_row(ctx, PRIMARY, form, theta, tg, ach,
                                     dist, extra))
        for sr in P.SRS:
            refit_out.append(outcome_row(ctx, PRIMARY, form, pool, F,
                                         scales, sr))
        print(f"  {ctx:7s} {form:10s} dist {dist:6.2f} "
              f"(excl A {refit_calib[-1]['distance_excl_A_target']:6.2f}) | "
              f"loss SR.10 {refit_out[-2]['perf_loss_pct_Hpath']:5.1f}% "
              f"SR.30 {refit_out[-1]['perf_loss_pct_Hpath']:5.1f}%")

# ============================================================ (6c) anchor-set outcomes
print("\n[6c] Anchor-set outcome grid (six-target linear fit, all sets, "
      "all contexts)")
set_calib, set_out = [], []
DIST = {}
for ctx in CONTEXTS:
    pool = POOLS[ctx]
    tr = P.TARGET_R_APPLICANT[ctx]
    for ps in ANCHOR_SETS:
        if ps == PRIMARY:   # reuse the 6b linear fit
            theta, scales = FITS[(ctx, "linear")]
            row = next(r for r in refit_calib
                       if r["context"] == ctx and r["form"] == "linear")
            DIST[(ctx, ps)] = row["weighted_distance"]
            thetas_sc = (theta, scales)
        else:
            F = F_CACHE[(ctx, ps)]
            theta, tg, ach, dist = calibrate(pool, F, P.SIGMA_INFL, tr)
            scales = apply_faking(pool, F, theta, P.SIGMA_INFL)
            DIST[(ctx, ps)] = round(dist, 2)
            set_calib.append(calib_row(ctx, ps, "linear", theta, tg, ach, dist))
            thetas_sc = (theta, scales)
        F = F_CACHE[(ctx, ps)]
        for sr in P.SRS:
            set_out.append(outcome_row(ctx, ps, "linear", pool, F,
                                       thetas_sc[1], sr))
for ctx in CONTEXTS:
    dmin = min(DIST[(ctx, ps)] for ps in ANCHOR_SETS)
    for r in set_out:
        if r["context"] == ctx:
            d = DIST[(ctx, r["param_set"])]
            r["weighted_distance"] = d
            r["fit_supported_2x_min"] = bool(d <= 2 * dmin)
for r in set_out:
    print(f"  {r['context']:7s} {r['param_set']:10s} SR {r['selection_ratio']:.2f} "
          f"loss {r['perf_loss_pct_Hpath']:5.1f}% | hires chg "
          f"{r['pct_hires_changed']:5.1f}% | dist {r['weighted_distance']:6.2f} "
          f"supported={r['fit_supported_2x_min']}")

# ============================================================ (6d) rho grid
print("\n[6d] RHO_CSE x RHO_HSE grid (+/-.10 around defaults; field_ns0, "
      "linear)")
rho_rows = []
for ctx in CONTEXTS:
    tr = P.TARGET_R_APPLICANT[ctx]
    for rc in RHO_CSE_GRID:
        for rh in RHO_HSE_GRID:
            if (rc, rh) == (P.RHO_CSE, P.RHO_HSE):
                pool, F = POOLS[ctx], F_CACHE[(ctx, PRIMARY)]
                theta, scales = FITS[(ctx, "linear")]
                row0 = next(r for r in refit_calib
                            if r["context"] == ctx and r["form"] == "linear")
                dist = row0["weighted_distance"]
            else:
                pool = build_pool(ctx, rho_cse=rc, rho_hse=rh)
                F, _, _ = solve_propensity(pool, P.PARAM_SETS[PRIMARY],
                                           P.P_FAKE)
                theta, tg, ach, dist = calibrate(pool, F, P.SIGMA_INFL, tr)
                scales = apply_faking(pool, F, theta, P.SIGMA_INFL)
            s10 = hire_stats(pool, F, scales["c_obs"], 0.10)
            rho_rows.append({
                "context": ctx, "rho_CSE": rc, "rho_HSE": rh,
                "is_default": (rc, rh) == (P.RHO_CSE, P.RHO_HSE),
                "weighted_distance": round(float(dist), 2),
                "perf_loss_pct_SR10": round(s10["perf_loss_pct_Hpath"], 2),
                "pct_hires_changed_SR10": round(s10["pct_hires_changed"], 2),
                "H_gap_SR10": round(s10["H_gap_faked_in_vs_displaced"], 3),
                "AIR_honest_SR10": round(s10["AIR_honest"], 3),
                "AIR_faked_SR10": round(s10["AIR_faked"], 3),
            })
rho_df = pd.DataFrame(rho_rows)
print(rho_df.to_string(index=False))

# ============================================================ (6e) ES falsification
print("\n[6e] Model-implied ES-GMA applicant shift vs Study 1's held-out "
      "Neuroticism contrast")
# Study 1 benchmark: Neuroticism applicant-vs-incumbent contrast from
# table2_contrasts.csv, sign-reversed to the ES scale; 95% CI from diff/z.
t2 = pd.read_csv(OUT / "table2_contrasts.csv")
nrow = t2[(t2["trait"] == "Neuroticism")
          & (t2["comparison"] == "applicant_vs_incumbent")].iloc[0]
es_obs_shift = -float(nrow["diff"])
es_se = abs(float(nrow["diff"]) / float(nrow["z"]))
es_lo, es_hi = es_obs_shift - 1.96 * es_se, es_obs_shift + 1.96 * es_se
es_rows = []
for ctx in CONTEXTS:
    for form in ("linear", "censored", "free_skill"):
        gap = es_gap(POOLS[ctx], FITS[(ctx, form)][1])
        es_rows.append({
            "context": ctx, "form": form,
            "model_implied_delta_r_gES": round(gap, 3),
            "study1_ES_benchmark": round(es_obs_shift, 3),
            "benchmark_95CI_lo": round(es_lo, 3),
            "benchmark_95CI_hi": round(es_hi, 3),
            "inside_95CI": bool(es_lo <= gap <= es_hi),
        })
        print(f"  {ctx:7s} {form:10s} implied {gap:+.3f} vs observed "
              f"{es_obs_shift:+.3f} [{es_lo:+.3f}, {es_hi:+.3f}] "
              f"{'INSIDE' if es_lo <= gap <= es_hi else 'OUTSIDE'}")
es_df = pd.DataFrame(es_rows)

# ============================================================ (5b) prevalence sweep
print(f"\n[5b] Prevalence sweep p_fake in {P.P_FAKE_SENS} (field_ns0, linear, "
      "full propensity re-solve + refit)")
prev_calib, prev_out = [], []
for ctx in CONTEXTS:
    pool = POOLS[ctx]
    tr = P.TARGET_R_APPLICANT[ctx]
    for pf in P.P_FAKE_SENS:
        F, _, _ = solve_propensity(pool, P.PARAM_SETS[PRIMARY], pf)
        theta, tg, ach, dist = calibrate(pool, F, P.SIGMA_INFL, tr)
        scales = apply_faking(pool, F, theta, P.SIGMA_INFL)
        row = calib_row(ctx, PRIMARY, "linear", theta, tg, ach, dist)
        row["p_fake"] = pf
        prev_calib.append(row)
        for sr in P.SRS:
            o = outcome_row(ctx, PRIMARY, "linear", pool, F, scales, sr)
            o["p_fake"] = pf
            base = next(r for r in refit_out
                        if r["context"] == ctx and r["form"] == "linear"
                        and r["selection_ratio"] == sr)
            o["primary40_pct_hires_changed"] = base["pct_hires_changed"]
            o["primary40_perf_loss_pct"] = base["perf_loss_pct_Hpath"]
            prev_out.append(o)
        print(f"  {ctx:7s} p={pf:.2f} dist {dist:6.2f} | loss SR.10 "
              f"{prev_out[-2]['perf_loss_pct_Hpath']:5.1f}% SR.30 "
              f"{prev_out[-1]['perf_loss_pct_Hpath']:5.1f}% | hires chg "
              f"{prev_out[-2]['pct_hires_changed']:5.1f}/"
              f"{prev_out[-1]['pct_hires_changed']:5.1f}%")

# ============================================================ write output
with open(OUT / f"study2_sixtarget_refit{P.HH_SUFFIX}.csv", "w", newline="") as f:
    f.write("# Six-target recalibration campaign "
            "(16_study2_sixtarget_refit.py) on the expanded Study 2 model "
            "(12_study2_expanded_model.py machinery, pools bit-identical);\n"
            f"# primary spec field_ns0, prevalence {P.P_FAKE}, sigma "
            f"{P.SIGMA_INFL}, n={P.N_POOL}, seed {P.SEED}. All six targets: "
            f"C shift {P.D_SHIFT}, ES shift {P.D_SHIFT_ES}, A shift "
            f"{P.D_SHIFT_A}, r_gC by context, delta r_SE {P.DELTA_R_SE_G}, "
            f"SD ratio {P.SD_RATIO_TARGET}.\n"
            "# All targeted-inflation distances include the constant "
            "Agreeableness penalty (~3.5); distance_excl_A_target removes "
            "it for cross-form comparison.\n")
    f.write("# (6b-a) Six-target refit of the functional forms: linear "
            "(= model 12), censored (inflation floored at 0), free_skill "
            "(m_skill_CES vs m_skill_SE; 5 params, 5 fittable targets)\n")
    pd.DataFrame(refit_calib).to_csv(f, index=False)
    f.write("\n# (6b-b) Selection outcomes under each form (field_ns0)\n")
    pd.DataFrame(refit_out).to_csv(f, index=False)
    f.write("\n# (6c-a) Anchor-set calibrations, six-target linear "
            "(field/lab; field_ns0 rows are in 6b-a)\n")
    pd.DataFrame(set_calib).to_csv(f, index=False)
    f.write("\n# (6c-b) Anchor-set selection outcomes, all sets x contexts; "
            "uniform reporting rule: all cells reported, "
            "fit_supported_2x_min flags sets within 2x the context-minimum "
            "distance (same heuristic as the r_CF profile)\n")
    pd.DataFrame(set_out).to_csv(f, index=False)
    f.write("\n# (6d) RHO_CSE x RHO_HSE grid, +/-.10 around defaults "
            f"(defaults {P.RHO_CSE}/{P.RHO_HSE}, both PROVISIONAL), "
            "field_ns0 linear, SR .10 outcomes\n")
    rho_df.to_csv(f, index=False)
    f.write("\n# (6e) Model-implied ES-GMA applicant shift (never a fitting "
            "target) vs Study 1's Neuroticism contrast sign-reversed to the "
            "ES scale (held-out benchmark; CI rebuilt from the contrast z)\n")
    es_df.to_csv(f, index=False)
    f.write("\n# (5b-a) Prevalence-sweep calibrations, p_fake in "
            f"{P.P_FAKE_SENS} (field_ns0, linear)\n")
    pd.DataFrame(prev_calib).to_csv(f, index=False)
    f.write("\n# (5b-b) Prevalence-sweep selection outcomes; primary40_* "
            "columns = the p=.40 primary fit on the same pool\n")
    pd.DataFrame(prev_out).to_csv(f, index=False)

print(f"\nSaved study2_sixtarget_refit{P.HH_SUFFIX}.csv")
