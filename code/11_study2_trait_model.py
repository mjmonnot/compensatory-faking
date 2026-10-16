"""Study 2 prototype: trait-driven (rather than random) faking in selection.

Extends the mechanism model of 09_selection_simulation.py, in which WHO fakes
is a random 40% of applicants, to a model in which faking propensity is a
function of the applicant's own traits. All numeric inputs live in
study2_params.py; provisional values there are flagged pending
literature/study2_anchor_table.md. Sourced anchors repeat 09: operational
validities V_g = .31, V_C = .19 (Sackett, Zhang, Berry, & Lievens, 2022,
Table 4); faking prevalence 40% (Griffith et al., 2007; Donovan et al.,
2014; union rate across dimensions); applicant-
incumbent C shift d = .45 (Birkeland et al., 2006); baseline and applicant
g-C correlations from this study (output/region_contrasts.csv).

Latent traits per applicant, multivariate normal: g (GMA), C (true
conscientiousness), H (honesty-humility/integrity), SE (confidence/
self-esteem). Model:

    y       = b_g*g + b_C*C [+ b_H*H] + e      (b solved so r(y, .) hits the
                                                operational validities; the H
                                                path is toggled on/off)
    need_i  = -(w_g*g_i + w_C*C_i)             (expectancy: applicants who
                                                cannot compete honestly
                                                perceive more need to fake;
                                                enters inflation magnitude)
    P(fake) = logistic(gamma_0 + gamma_g*g_i + gamma_C*C_i + gamma_H*H_i),
              with ALL gammas solved so the pool reproduces the anchored
              point-biserial trait-faking correlations (field: Holtrop et
              al. 2021; lab: Dunlop et al. 2022) and prevalence = target
    infl_i  = m0 + m_need*need_i + m_skill*g_i + N(0, sigma)   (fakers only)
    C_obs   = C + F*infl
    SE_obs  = SE + F*(s0 + m_skill*g_i + N(0, sigma))          (shared skill
                                                                coefficient)

Calibration: minimum-distance (weighted least squares) over applicant-pool
moments, because the system is over-identified — 4 moments constrain the 3
free inflation parameters (m0, m_need, m_skill), with (gamma_H, gamma_need,
w_g, w_C, s0) pinned externally by the anchor table. Target moments:
    (1) C mean shift d = .45 (Birkeland);
    (2) applicant g-C correlation (fit separately per context: US +.007,
        non-US -.272, pooled -.074);
    (3) applicant SE-g correlation shift +.17 (this study, compound traits);
    (4) applicant/incumbent observed-C SD ratio (target 0.86, Hu & Connelly
        2021, Table 4; sensitivity spans the 80% CrI lower bound to 1.00).
Achieved vs target moments and the minimized weighted distance are reported;
a poor fit is itself informative and is not forced. The N/E/A/O null
constraints (no applicant-incumbent shift) cannot enter the moment vector
because those traits are not in the latent vector; the model satisfies them
structurally (only the C and SE scales are faked) — a limitation documented
in literature/study2_model_notes.md.

Outcomes mirror 09 on the same simulated pool: % of top-down hires changed
vs the honest counterfactual, mean true performance of hires (with and
without the H->performance path), AIR, at SR .10 and .30, n = 200,000,
seeded rng. Additionally: mean H of faked-in hires (hired only under faked
scoring) vs displaced hires (hired only under honest scoring), benchmarked
against the random-faking mechanism of 09 recalibrated on the same pool.
Sensitivity: field- vs lab-anchored propensity sets, prevalence P_FAKE_SENS
plus the .40 primary, sigma {.25, .50, 1.00}.

Outputs: output/study2_prototype_results.csv
"""

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


# ---------------------------------------------------------------- pool construction
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


def perf_coefs(R: np.ndarray, include_h: bool) -> tuple[np.ndarray, float]:
    """Regression weights of y on (g, C[, H]) reproducing operational validities."""
    idx = [0, 1, 2] if include_h else [0, 1]
    v = np.array([P.V_G, P.V_C, P.V_H])[: len(idx)]
    b = np.linalg.solve(R[np.ix_(idx, idx)], v)
    return b, float(np.sqrt(1 - v @ b))


def build_pool(context: str, n_pool: int = P.N_POOL) -> dict:
    """Draw the applicant pool once per context; all conditions reuse it."""
    R = trait_corr(context)
    minority = rng.random(n_pool) < P.P_MIN
    z = rng.multivariate_normal(np.zeros(4), R, size=n_pool)
    z[minority] -= np.array([P.D_G, P.D_C, P.D_H, P.D_SE])
    g, c, h, se = z.T

    eps_y = rng.normal(0, 1, n_pool)  # common shock so H-path toggle is comparable
    b_h, sd_h = perf_coefs(R, include_h=True)
    b_n, sd_n = perf_coefs(R, include_h=False)
    y_h = b_h[0] * g + b_h[1] * c + b_h[2] * h + sd_h * eps_y
    y_n = b_n[0] * g + b_n[1] * c + sd_n * eps_y

    return {
        "context": context, "minority": minority,
        "g": g, "c": c, "h": h, "se": se,
        "need": -(P.W_G_NEED * g + P.W_C_NEED * c),
        "y_hpath": y_h, "y_nohpath": y_n,
        # common random numbers, fixed across conditions and optimizer steps
        "u_fake": rng.random(n_pool), "u_rand": rng.random(n_pool),
        "eps_c": rng.normal(0, 1, n_pool), "eps_se": rng.normal(0, 1, n_pool),
        "eps_rand": rng.normal(0, 1, n_pool),
    }


# ---------------------------------------------------------------- faking mechanisms
def solve_propensity(pool: dict, targets: dict,
                     p_fake: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Solve (gamma_0, gamma_g, gamma_C, gamma_H) so the pool reproduces the
    anchored point-biserial trait-faking correlations and the prevalence.

    Uses E[F | traits] = p_i for smooth moments: Cov(x, F) = Cov(x, p) by the
    tower property, and Var(F) = pbar(1 - pbar) exactly for a Bernoulli mixture,
    so the point-biserial correlations are deterministic functions of the
    gammas. F itself is then drawn once from the common random numbers.
    """
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


def apply_faking(pool: dict, F: np.ndarray, theta: np.ndarray,
                 sigma: float) -> tuple[np.ndarray, np.ndarray]:
    m0, m_need, m_skill = theta
    c_obs = pool["c"] + F * (m0 + m_need * pool["need"] + m_skill * pool["g"]
                             + sigma * pool["eps_c"])
    se_obs = pool["se"] + F * (P.S0_SE + m_skill * pool["g"]
                               + sigma * pool["eps_se"])
    return c_obs, se_obs


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


def calibrate_trait_model(pool: dict, F: np.ndarray, sigma: float,
                          target_r: float, sd_target: float = P.SD_RATIO_TARGET,
                          ) -> tuple[np.ndarray, dict, dict, float]:
    """Minimum-distance fit of (m0, m_need, m_skill) to the target moments."""
    targets = {"d_shift": P.D_SHIFT, "r_gC": target_r,
               "delta_r_SE": P.DELTA_R_SE_G, "sd_ratio": sd_target}
    sqw = np.sqrt(np.array([P.MOMENT_WEIGHTS[k] for k in ACTIVE_MOMENTS]))

    def resid(theta):
        m = pool_moments(pool, *apply_faking(pool, F, theta, sigma))
        return sqw * np.array([m[k] - targets[k] for k in ACTIVE_MOMENTS])

    fit = least_squares(resid, x0=[P.D_SHIFT / F.mean(), 1.0, 0.3], xtol=1e-12)
    achieved = pool_moments(pool, *apply_faking(pool, F, fit.x, sigma))
    return fit.x, targets, achieved, float(2 * fit.cost)  # cost = .5*sum(resid^2)


def calibrate_random(rho0: float, target_r: float, sigma: float,
                     p_fake: float) -> tuple[float, float]:
    """09's random-faking calibration (closed-form moments), prevalence generalized."""

    def moments(params):
        mu_f, b = params
        var_infl = p_fake * (mu_f**2 + b**2 + sigma**2) - (p_fake * mu_f) ** 2
        var_cobs = 1 - 2 * p_fake * b * rho0 + var_infl
        r_gc = (rho0 - p_fake * b) / np.sqrt(var_cobs)
        d = (p_fake * mu_f) / np.sqrt((1 + var_cobs) / 2)
        return [r_gc - target_r, d - P.D_SHIFT]

    mu_f, b = fsolve(moments, x0=[P.D_SHIFT / p_fake, max(0.0, rho0 - target_r)])
    return float(mu_f), float(b)


# ---------------------------------------------------------------- selection outcomes
def hire_stats(pool: dict, F: np.ndarray, c_obs: np.ndarray, sr: float) -> dict:
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


def run_condition(pool: dict, param_set: str, p_fake: float, sigma: float,
                  sd_target: float = P.SD_RATIO_TARGET) -> tuple[dict, list[dict]]:
    context = pool["context"]
    gs = P.PARAM_SETS[param_set]
    F, gammas, prop_achieved = solve_propensity(pool, gs, p_fake)
    theta, targets, achieved, dist = calibrate_trait_model(
        pool, F, sigma, P.TARGET_R_APPLICANT[context], sd_target)
    c_obs, _ = apply_faking(pool, F, theta, sigma)

    # diagnostic: share of fakers receiving negative inflation (self-sabotage
    # artifact of the linear inflation form; see model notes section 5)
    infl = (theta[0] + theta[1] * pool["need"] + theta[2] * pool["g"]
            + sigma * pool["eps_c"])
    pct_neg_infl = 100 * float((infl[F] < 0).mean())

    # 09-style random-faking benchmark on the same pool
    mu_f, b = calibrate_random(P.RHO_GC_BASE[context],
                               P.TARGET_R_APPLICANT[context], sigma, p_fake)
    F_r = pool["u_rand"] < p_fake
    c_obs_r = pool["c"] + F_r * (mu_f - b * pool["g"] + sigma * pool["eps_rand"])

    label = {"context": context, "param_set": param_set,
             "p_fake": p_fake, "sigma_noise": sigma,
             "sd_ratio_target": sd_target}
    calib = dict(label)
    calib.update({"target_r_gF": gs["r_gF"], "target_r_CF": gs["r_CF"],
                  "target_r_HF": gs["r_HF"],
                  "gamma_0": round(float(gammas[0]), 3),
                  "gamma_g": round(float(gammas[1]), 3),
                  "gamma_C": round(float(gammas[2]), 3),
                  "gamma_H": round(float(gammas[3]), 3),
                  "achieved_r_gF": round(prop_achieved["r_gF"], 3),
                  "achieved_r_CF": round(prop_achieved["r_CF"], 3),
                  "achieved_r_HF": round(prop_achieved["r_HF"], 3),
                  "m0": round(float(theta[0]), 3),
                  "m_need": round(float(theta[1]), 3),
                  "m_skill": round(float(theta[2]), 3),
                  "pct_fakers_neg_inflation": round(pct_neg_infl, 1),
                  "weighted_distance": round(dist, 2),
                  "rand_mu_f": round(mu_f, 3), "rand_g_slope_b": round(b, 3)})
    for k in MOMENT_KEYS:
        calib[f"target_{k}"] = round(targets[k], 3) if k in ACTIVE_MOMENTS else np.nan
        calib[f"achieved_{k}"] = round(achieved[k], 3)

    rows = []
    for sr in P.SRS:
        t = hire_stats(pool, F, c_obs, sr)
        r = hire_stats(pool, F_r, c_obs_r, sr)
        row = dict(label)
        row["selection_ratio"] = sr
        row.update({k: round(v, 3) for k, v in t.items()})
        row.update({
            "rand_pct_hires_changed": round(r["pct_hires_changed"], 3),
            "rand_perf_loss_pct_Hpath": round(r["perf_loss_pct_Hpath"], 3),
            "rand_perf_loss_pct_noH": round(r["perf_loss_pct_noH"], 3),
            "rand_AIR_faked": round(r["AIR_faked"], 3),
            "rand_mean_H_hires_faked": round(r["mean_H_hires_faked"], 3),
            "rand_mean_H_faked_in": round(r["mean_H_faked_in"], 3),
            "extra_perf_loss_vs_random_Hpath":
                round(t["perf_loss_pct_Hpath"] - r["perf_loss_pct_Hpath"], 3),
            "H_gap_faked_in_vs_random":
                round(t["mean_H_faked_in"] - r["mean_H_faked_in"], 3),
        })
        rows.append(row)
    return calib, rows


# ---------------------------------------------------------------- run
CONTEXTS = ("US", "Pooled", "Non-US")
POOLS = {ctx: build_pool(ctx) for ctx in CONTEXTS}

calib_rows, main_rows, sens_rows = [], [], []

# main grid: the field-vs-lab adjudication, all contexts
for ctx in CONTEXTS:
    for ps in ("field", "lab"):
        calib, rows = run_condition(POOLS[ctx], ps, P.P_FAKE, P.SIGMA_INFL)
        calib_rows.append(calib)
        main_rows.extend(rows)

# one-at-a-time sensitivity (US and Non-US, reported at SR = .10)
sens_conditions = ([(ps, P.P_FAKE, P.SIGMA_INFL)
                    for ps in ("field_ns0", "lab_negH")]
                   + [(P.DEFAULT_PARAM_SET, pf, P.SIGMA_INFL) for pf in P.P_FAKE_SENS]
                   + [(P.DEFAULT_PARAM_SET, P.P_FAKE, sg) for sg in P.SIGMA_SENS]
                   # SD-ratio target sweep: no empirical anchor retrievable
                   # (Hu & Connelly 2021 paywalled); direction-only evidence
                   # says applicant SD slightly < honest SD
                   + [(P.DEFAULT_PARAM_SET, P.P_FAKE, P.SIGMA_INFL, sdt)
                      for sdt in P.SD_RATIO_SENS])
for ctx in ("US", "Non-US"):
    for cond in sens_conditions:
        calib, rows = run_condition(POOLS[ctx], *cond)
        calib_rows.append(calib)
        sens_rows.extend(r for r in rows if r["selection_ratio"] == P.SRS[0])

calib_df = pd.DataFrame(calib_rows)
main_df = pd.DataFrame(main_rows)
sens_df = pd.DataFrame(sens_rows)

with open(OUT / f"study2_prototype_results{P.HH_SUFFIX}.csv", "w", newline="") as f:
    f.write("# Study 2 prototype: trait-driven faking model "
            "(11_study2_trait_model.py; parameters in study2_params.py,\n"
            "# provisional values flagged there pending "
            "literature/study2_anchor_table.md)\n")
    f.write("# Calibration: minimum-distance fit, achieved vs target moments "
            "(weighted_distance = minimized WLS criterion)\n")
    calib_df.to_csv(f, index=False)
    f.write("\n# Main grid: field- vs lab-anchored propensity sets "
            "(prevalence .40, sigma .50), SR .10/.30; rand_* = 09-style "
            "random-faking benchmark on the same pool\n")
    main_df.to_csv(f, index=False)
    f.write("\n# Sensitivity (SR = .10, one-at-a-time): field ns-to-zero "
            f"variant, lab negative-H variant, prevalence {P.P_FAKE_SENS}, "
            f"sigma .25/1.00, SD-ratio target {P.SD_RATIO_SENS[0]}/"
            f"{P.SD_RATIO_SENS[1]} (anchor {P.SD_RATIO_TARGET}, "
            "Hu & Connelly 2021)\n")
    sens_df.to_csv(f, index=False)

show_calib = ["context", "param_set", "p_fake", "sigma_noise",
              "gamma_g", "gamma_C", "gamma_H",
              "m0", "m_need", "m_skill", "pct_fakers_neg_inflation",
              "weighted_distance",
              "target_r_gC", "achieved_r_gC", "achieved_d_shift",
              "achieved_delta_r_SE", "target_sd_ratio", "achieved_sd_ratio"]
show_main = ["context", "param_set", "selection_ratio", "pct_hires_changed",
             "perf_loss_pct_Hpath", "perf_loss_pct_noH",
             "rand_perf_loss_pct_Hpath", "extra_perf_loss_vs_random_Hpath",
             "mean_H_faked_in", "mean_H_displaced", "rand_mean_H_faked_in",
             "AIR_honest", "AIR_faked"]
print("Calibration (achieved vs target; d target = "
      f"{P.D_SHIFT}, delta_r_SE target = {P.DELTA_R_SE_G}, "
      f"sd_ratio target = {P.SD_RATIO_TARGET}):")
print(calib_df[show_calib].to_string(index=False))
print("\nMain grid (rand_* = random-faking benchmark):")
print(main_df[show_main].to_string(index=False))
print("\nSensitivity (SR = .10):")
print(sens_df[["context", "param_set", "p_fake", "sigma_noise",
               "sd_ratio_target"] + show_main[2:]].to_string(index=False))
print(f"\nSaved study2_prototype_results{P.HH_SUFFIX}.csv")
