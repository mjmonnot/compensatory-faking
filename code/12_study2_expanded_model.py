"""Study 2 expanded model (v3): six-trait latent vector, targeted vs
desirability-proportional inflation.

Resolves the structural-null limitation of 11_study2_trait_model.py (model
notes section 4): Emotional Stability (ES) and Agreeableness (A) join the
latent vector (g, C, ES, A, H, SE), so the Birkeland et al. (2006)
applicant-incumbent shift pattern — C d = .45, ES d = .44, A ~ 0 — becomes a
genuine over-identifying test rather than being satisfied by construction.

WHO fakes is unchanged from model 11: P(fake) = logistic in (g, C, H), all
gammas solved so the pool reproduces anchored point-biserial trait-faking
correlations plus prevalence. Primary anchor set: field_ns0 (r_gF = -.19,
r_CF = 0, r_HF = 0), the project's decided primary specification; field and
lab as comparison sets.

WHAT gets inflated — two competing variants, calibrated by the same WLS
minimum-distance machinery and compared on the minimized weighted distance:

  Variant T (targeted): fakers inflate only the C, ES, and SE scales
    (job-relevant targeting). C and ES each get a free mean-inflation
    intercept; need and skill slopes are shared across scales as in model 11;
    the SE scale keeps model 11's form (intercept s0 fixed at 0, skill term
    only) because no SE mean-shift anchor exists. Free: (m0_C, m0_ES,
    m_need, m_skill). The A scale is untouched, so the A null is satisfied
    structurally — but it now sits in the moment vector, so variant T's fit
    is directly comparable to variant D's.

  Variant D (desirability-proportional / trait-general): fakers inflate ALL
    modeled Big Five scales (C, ES, A) plus SE proportionally to a
    social-desirability loading vector lambda (Ones, Viswesvaran, & Reiss,
    1996: SD-ES = .37, SD-C = .20 verified; lambda_A, lambda_SE provisional,
    see study2_params.py). Inflation on scale j:
        infl_j = (lambda_j / lambda_C) * (kappa + m_need*need + m_skill*g
                                          + N(0, sigma))
    One free scale parameter kappa multiplies the whole vector. Free:
    (kappa, m_need, m_skill). Variant D predicts nonzero A shifts whenever
    lambda_A > 0 and d_ES/d_C ~ lambda_ES/lambda_C = 1.85 — both are
    testable against the Birkeland pattern (ratio ~ .98, A null), so the
    observed moments can now REJECT the trait-general account.

Calibration: WLS over SIX applicant-pool moments — C shift d = .45, ES shift
d = .44, A shift d = .00 (Birkeland), applicant r(g, C_obs) by context (this
study), applicant delta r(g, SE_obs) = +.17 (this study), applicant/incumbent
observed-C SD ratio 1.00 (provisional). Fit failures are findings, not
errors; nothing is forced.

Trait correlation matrix extended per study2_params.py (sources and
PROVISIONAL flags there); positive definiteness is verified and the script
fails loudly if any anchored matrix is not positive definite.

Outcomes reuse the model-11 machinery (composite = .5*g + .5*std C_obs,
top-down at SR .10/.30, percent hires changed, performance loss with/without
the H path, AIR, mean H of faked-in vs displaced) for BOTH variants under the
primary spec (field_ns0), n = 200,000, seed 2026, common random numbers
across variants. Sensitivities: rho_gES = .09 (Anglim), lambda_A in
{.10, .30}.

Outputs: output/study2_expanded_results.csv
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

# trait order everywhere: (g, C, ES, A, H, SE)
TRAITS = ("g", "c", "es", "a", "h", "se")
MOMENT_KEYS = ("d_shift", "d_shift_ES", "d_shift_A",
               "r_gC", "delta_r_SE", "sd_ratio")
MOMENT_WEIGHTS = {**P.MOMENT_WEIGHTS, **P.MOMENT_WEIGHTS_12}
LAM = {k: v / P.LAMBDA_SD["C"] for k, v in P.LAMBDA_SD.items()}  # lambda_C = 1


# ---------------------------------------------------------------- pool construction
def trait_corr(context: str, rho_ges: float = P.RHO_GES) -> tuple[np.ndarray, float]:
    """Latent correlation matrix, order (g, C, ES, A, H, SE).

    Returns (R, pd_adjustment): pd_adjustment is 0.0 if the anchored matrix
    is already positive definite, else the max element change applied by the
    nearest-PD fallback.
    """
    r = P.RHO_GC_BASE[context]
    R = np.array([
        # g       C          ES         A          H          SE
        [1.0,     r,         rho_ges,   P.RHO_GA,  P.RHO_GH,  P.RHO_GSE],
        [r,       1.0,       P.RHO_CES, P.RHO_CA,  P.RHO_CH,  P.RHO_CSE],
        [rho_ges, P.RHO_CES, 1.0,       P.RHO_ESA, P.RHO_ESH, P.RHO_ESSE],
        [P.RHO_GA, P.RHO_CA, P.RHO_ESA, 1.0,       P.RHO_AH,  P.RHO_ASE],
        [P.RHO_GH, P.RHO_CH, P.RHO_ESH, P.RHO_AH,  1.0,       P.RHO_HSE],
        [P.RHO_GSE, P.RHO_CSE, P.RHO_ESSE, P.RHO_ASE, P.RHO_HSE, 1.0],
    ])
    if np.linalg.eigvalsh(R).min() > 0:
        return R, 0.0
    # Production matrices are all PD (audited 2026-08-20: min eigenvalue
    # >= .45 in every condition); reaching this branch means a parameter
    # change broke that, so fail loudly rather than silently reshaping R.
    raise ValueError(
        f"Anchored correlation matrix for {context!r} is not positive "
        f"definite (min eigenvalue {np.linalg.eigvalsh(R).min():.4f}). "
        "Review the changed parameters before using a nearest-PD projection."
    )


def perf_coefs(R: np.ndarray, include_h: bool) -> tuple[np.ndarray, float]:
    """Regression weights of y on (g, C[, H]) reproducing operational
    validities — same performance model as 11 (ES/A carry no direct path;
    they influence y only through their correlations with g, C, H)."""
    idx = [0, 1, 4] if include_h else [0, 1]
    v = np.array([P.V_G, P.V_C, P.V_H])[: len(idx)]
    b = np.linalg.solve(R[np.ix_(idx, idx)], v)
    return b, float(np.sqrt(1 - v @ b))


def build_pool(context: str, rho_ges: float = P.RHO_GES,
               n_pool: int = P.N_POOL) -> dict:
    """Draw the applicant pool once per (context, rho_gES); all conditions
    and both variants reuse it (common random numbers)."""
    R, pd_adj = trait_corr(context, rho_ges)
    minority = rng.random(n_pool) < P.P_MIN
    z = rng.multivariate_normal(np.zeros(6), R, size=n_pool)
    z[minority] -= np.array([P.D_G, P.D_C, P.D_ES, P.D_A, P.D_H, P.D_SE])
    g, c, es, a, h, se = z.T

    eps_y = rng.normal(0, 1, n_pool)  # common shock so H-path toggle is comparable
    b_h, sd_h = perf_coefs(R, include_h=True)
    b_n, sd_n = perf_coefs(R, include_h=False)
    y_h = b_h[0] * g + b_h[1] * c + b_h[2] * h + sd_h * eps_y
    y_n = b_n[0] * g + b_n[1] * c + sd_n * eps_y

    return {
        "context": context, "rho_ges": rho_ges, "pd_adjustment": pd_adj,
        "minority": minority,
        "g": g, "c": c, "es": es, "a": a, "h": h, "se": se,
        "need": -(P.W_G_NEED * g + P.W_C_NEED * c),
        "y_hpath": y_h, "y_nohpath": y_n,
        # common random numbers, fixed across variants and optimizer steps
        "u_fake": rng.random(n_pool),
        "eps_c": rng.normal(0, 1, n_pool), "eps_es": rng.normal(0, 1, n_pool),
        "eps_a": rng.normal(0, 1, n_pool), "eps_se": rng.normal(0, 1, n_pool),
    }


# ---------------------------------------------------------------- faking mechanisms
def solve_propensity(pool: dict, targets: dict,
                     p_fake: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """WHO fakes — identical mechanism to model 11: gammas solved so the pool
    reproduces the anchored point-biserial trait-faking correlations and the
    prevalence (smooth moments via E[F|traits])."""
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
                 variant: str, lam_a: float = LAM["A"]) -> dict:
    """Observed scales under the given inflation variant.

    T: theta = (m0_C, m0_ES, m_need, m_skill); A untouched; SE as model 11.
    D: theta = (kappa, m_need, m_skill); all of C, ES, A, SE inflated
       proportionally to the normalized lambda vector.
    """
    need, g = pool["need"], pool["g"]
    if variant == "T":
        m0_c, m0_es, m_need, m_skill = theta
        slope = m_need * need + m_skill * g
        return {
            "c_obs": pool["c"] + F * (m0_c + slope + sigma * pool["eps_c"]),
            "es_obs": pool["es"] + F * (m0_es + slope + sigma * pool["eps_es"]),
            "a_obs": pool["a"],
            "se_obs": pool["se"] + F * (P.S0_SE + m_skill * g
                                        + sigma * pool["eps_se"]),
        }
    kappa, m_need, m_skill = theta
    base = kappa + m_need * need + m_skill * g
    return {
        "c_obs": pool["c"] + F * (base + sigma * pool["eps_c"]),  # lambda_C = 1
        "es_obs": pool["es"] + F * LAM["ES"] * (base + sigma * pool["eps_es"]),
        "a_obs": pool["a"] + F * lam_a * (base + sigma * pool["eps_a"]),
        "se_obs": pool["se"] + F * LAM["SE"] * (base + sigma * pool["eps_se"]),
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


def calibrate(pool: dict, F: np.ndarray, sigma: float, variant: str,
              target_r: float, lam_a: float = LAM["A"],
              ) -> tuple[np.ndarray, dict, dict, float]:
    """Minimum-distance (WLS) fit of the variant's free inflation parameters
    to the six target moments. A poor fit is reported, never forced."""
    targets = {"d_shift": P.D_SHIFT, "d_shift_ES": P.D_SHIFT_ES,
               "d_shift_A": P.D_SHIFT_A, "r_gC": target_r,
               "delta_r_SE": P.DELTA_R_SE_G, "sd_ratio": P.SD_RATIO_TARGET}
    sqw = np.sqrt(np.array([MOMENT_WEIGHTS[k] for k in MOMENT_KEYS]))

    def resid(theta):
        m = pool_moments(pool, apply_faking(pool, F, theta, sigma, variant, lam_a))
        return sqw * np.array([m[k] - targets[k] for k in MOMENT_KEYS])

    x0 = ([P.D_SHIFT / F.mean(), P.D_SHIFT_ES / F.mean(), 1.0, 0.3]
          if variant == "T" else [P.D_SHIFT / F.mean(), 1.0, 0.3])
    fit = least_squares(resid, x0=x0, xtol=1e-12)
    achieved = pool_moments(pool, apply_faking(pool, F, fit.x, sigma,
                                               variant, lam_a))
    return fit.x, targets, achieved, float(2 * fit.cost)  # cost = .5*sum(resid^2)


# ---------------------------------------------------------------- selection outcomes
def hire_stats(pool: dict, F: np.ndarray, c_obs: np.ndarray, sr: float) -> dict:
    """Identical to model 11: only g and standardized C_obs are scored."""
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

    perf_hh, perf_hn = pool["y_hpath"][sel_h].mean(), pool["y_nohpath"][sel_h].mean()
    perf_fh, perf_fn = pool["y_hpath"][sel_f].mean(), pool["y_nohpath"][sel_f].mean()
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


# ---------------------------------------------------------------- condition runner
def run_calibration(pool: dict, F: np.ndarray, param_set: str, variant: str,
                    lam_a: float = LAM["A"], tag: str = "") -> tuple[dict, np.ndarray]:
    context = pool["context"]
    theta, targets, achieved, dist = calibrate(
        pool, F, P.SIGMA_INFL, variant, P.TARGET_R_APPLICANT[context], lam_a)
    row = {"context": context, "param_set": param_set, "variant": variant,
           "sensitivity": tag, "rho_gES": pool["rho_ges"],
           "lambda_A_norm": round(lam_a, 3) if variant == "D" else np.nan,
           "p_fake": P.P_FAKE, "sigma_noise": P.SIGMA_INFL,
           "weighted_distance": round(dist, 2)}
    names = (("m0_C", "m0_ES", "m_need", "m_skill") if variant == "T"
             else ("kappa", "m_need", "m_skill"))
    for n_, v in zip(names, theta):
        row[n_] = round(float(v), 3)
    for k in MOMENT_KEYS:
        row[f"target_{k}"] = round(targets[k], 3)
        row[f"achieved_{k}"] = round(achieved[k], 3)
    return row, theta


# ---------------------------------------------------------------- run
CONTEXTS = ("US", "Pooled", "Non-US")
POOLS = {ctx: build_pool(ctx) for ctx in CONTEXTS}
for ctx in CONTEXTS:
    if POOLS[ctx]["pd_adjustment"] > 0:
        print(f"[PD] {ctx}: nearest-PD applied, max element change "
              f"= {POOLS[ctx]['pd_adjustment']:.4f}")

# cache the faking indicator per (context, param set): identical across variants
F_CACHE, PROP_ROWS = {}, []
for ctx in CONTEXTS:
    for ps in P.PARAM_SETS_12:
        F, gammas, ach = solve_propensity(POOLS[ctx], P.PARAM_SETS[ps], P.P_FAKE)
        F_CACHE[(ctx, ps)] = F
        PROP_ROWS.append({"context": ctx, "param_set": ps,
                          **{f"gamma_{n}": round(float(v), 3)
                             for n, v in zip(("0", "g", "C", "H"), gammas)},
                          **{f"achieved_{k}": round(v, 3) for k, v in ach.items()}})

# headline grid: variant T vs variant D, all contexts x anchor sets
calib_rows, thetas = [], {}
for ctx in CONTEXTS:
    for ps in P.PARAM_SETS_12:
        for var in ("T", "D"):
            row, theta = run_calibration(POOLS[ctx], F_CACHE[(ctx, ps)], ps, var)
            calib_rows.append(row)
            thetas[(ctx, ps, var)] = theta

# outcomes: primary spec (field_ns0), BOTH variants, SR .10/.30, common pool
out_rows = []
for ctx in CONTEXTS:
    ps = P.PRIMARY_PARAM_SET_12
    F = F_CACHE[(ctx, ps)]
    for var in ("T", "D"):
        scales = apply_faking(POOLS[ctx], F, thetas[(ctx, ps, var)],
                              P.SIGMA_INFL, var)
        for sr in P.SRS:
            s = hire_stats(POOLS[ctx], F, scales["c_obs"], sr)
            out_rows.append({"context": ctx, "param_set": ps, "variant": var,
                             "selection_ratio": sr,
                             **{k: round(v, 3) for k, v in s.items()}})

# sensitivities (calibration only): rho_gES = .09; lambda_A in {.10, .30}
sens_rows = []
for ctx in ("US", "Non-US"):
    pool_s = build_pool(ctx, rho_ges=P.RHO_GES_SENS)
    F_s, _, _ = solve_propensity(pool_s, P.PARAM_SETS[P.PRIMARY_PARAM_SET_12],
                                 P.P_FAKE)
    for var in ("T", "D"):
        row, _ = run_calibration(pool_s, F_s, P.PRIMARY_PARAM_SET_12, var,
                                 tag="rho_gES=.09")
        sens_rows.append(row)
    for la in P.LAMBDA_A_SENS:
        row, _ = run_calibration(POOLS[ctx], F_CACHE[(ctx, P.PRIMARY_PARAM_SET_12)],
                                 P.PRIMARY_PARAM_SET_12, "D",
                                 lam_a=la / P.LAMBDA_SD["C"],
                                 tag=f"lambda_A={la}")
        sens_rows.append(row)

prop_df = pd.DataFrame(PROP_ROWS)
calib_df = pd.DataFrame(calib_rows)
out_df = pd.DataFrame(out_rows)
sens_df = pd.DataFrame(sens_rows)

with open(OUT / f"study2_expanded_results{P.HH_SUFFIX}.csv", "w", newline="") as f:
    f.write("# Study 2 expanded model (v3): six-trait latent vector "
            "(g, C, ES, A, H, SE), targeted (T) vs desirability-proportional "
            "(D) inflation\n"
            "# (12_study2_expanded_model.py; parameters in study2_params.py "
            "section '12_study2_expanded_model.py additions',\n"
            "# provisional values flagged there). "
            f"Moments: C shift {P.D_SHIFT} / ES shift {P.D_SHIFT_ES} / "
            f"A shift {P.D_SHIFT_A} (Birkeland 2006), applicant r_gC by "
            f"context, delta r_SE-g +{P.DELTA_R_SE_G}, "
            f"SD ratio {P.SD_RATIO_TARGET}\n")
    f.write("# Propensity solves (WHO fakes; identical across variants)\n")
    prop_df.to_csv(f, index=False)
    f.write("\n# Headline calibration grid: weighted_distance = minimized "
            "WLS criterion; variant D lambda (normalized by lambda_C): "
            f"ES {LAM['ES']:.2f}, A {LAM['A']:.2f}, SE {LAM['SE']:.2f}\n")
    calib_df.to_csv(f, index=False)
    f.write("\n# Selection outcomes, primary spec (field_ns0), both "
            "variants, SR .10/.30, common random numbers across variants\n")
    out_df.to_csv(f, index=False)
    f.write("\n# Sensitivity calibrations (field_ns0): rho_gES = .09 "
            "(Anglim 2022 anchor), lambda_A in {.10, .30} (crux provisional "
            "loading, variant D)\n")
    sens_df.to_csv(f, index=False)

show = ["context", "param_set", "variant", "weighted_distance",
        "achieved_d_shift", "achieved_d_shift_ES", "achieved_d_shift_A",
        "achieved_r_gC", "achieved_delta_r_SE", "achieved_sd_ratio"]
print(f"\nHeadline calibration grid (targets: d_C {P.D_SHIFT}, "
      f"d_ES {P.D_SHIFT_ES}, d_A {P.D_SHIFT_A}, r_gC by context, "
      f"delta_r_SE {P.DELTA_R_SE_G}, sd_ratio {P.SD_RATIO_TARGET}):")
print(calib_df[show + ["m0_C", "m0_ES", "kappa", "m_need", "m_skill"]]
      .to_string(index=False))
print("\nOutcomes, primary spec (field_ns0):")
print(out_df[["context", "variant", "selection_ratio", "pct_hires_changed",
              "perf_loss_pct_Hpath", "perf_loss_pct_noH", "AIR_honest",
              "AIR_faked", "mean_H_faked_in", "mean_H_displaced"]]
      .to_string(index=False))
print("\nSensitivities:")
print(sens_df[["sensitivity"] + show[:1] + show[2:]].to_string(index=False))
print(f"\nSaved study2_expanded_results{P.HH_SUFFIX}.csv")
