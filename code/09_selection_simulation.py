"""Model contextual (stakes/region) effects on rank-order selection decisions.

All inputs are sourced from published meta-analyses or from this study's database
estimates (no illustrative values):

Operational validities (criterion-reliability corrected, no range-restriction
correction), Sackett, Zhang, Berry, & Lievens (2022, JAP, Table 4):
    v_g = .31 (cognitive ability tests), v_c = .19 (conscientiousness, overall).
Black-White standardized differences, Sackett et al. (2022, Table 4; d_g from
Roth et al., 2011 applicant data; d_c from Dahlke & Sackett, 2017):
    d_g = .79, d_c = -.07 (positive = White higher).
Baseline (honest) C-GMA correlations, this study (output/region_contrasts.csv,
incumbent research administrations): US +.013, non-US +.054. Field-standard
matrix input for comparison: .03 (Berry, Lievens, Zhang, & Sackett, 2024, Table 1).
Applicant (high-stakes) target correlations, this study: US +.007,
pooled -.074 (observed, table1), non-US -.272.
Faking prevalence among applicants: 40% (midpoint of the 30-50% within-person
range; Griffith, Chmielowski, & Yoshita, 2007; Donovan, Dwight, & Schneider,
2014; the Griffith & Converse, 2011 synthesis concludes 30% +/-10%).
Prevalence is the union ("faked on >=1 dimension") rate, matching the model's
single faker indicator; per-dimension classification rates run lower (14-25%
in Donovan et al., 2014).
Applicant-incumbent mean shift on C: d = .45 (Birkeland et al., 2006).

Part A (analytic): expected standardized job performance of hires and adverse-
impact (AI) ratio for an equal-weight GMA + C composite under top-down selection.

Part B (Monte Carlo mechanism model): honest scores (g, C_true) at the incumbent
baseline correlation; a random 40% of applicants fake:
    C_obs = C_true + F * (mu_f - b*g + e),  F ~ Bernoulli(.40),  e ~ N(0, sigma)
(mu_f, b) are jointly calibrated so the pool reproduces (i) the Birkeland et al.
mean shift (d = .45 against the honest distribution) and (ii) the applicant
C-GMA correlation observed in this study. sigma (idiosyncratic faking noise) is
varied {.25, .50, 1.00} as a sensitivity parameter. Consequences are measured
against the no-faking counterfactual on the same simulated pool.

Outputs: output/table7_selection_consequences.csv, output/figure3_selection_effects.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq, fsolve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
rng = np.random.default_rng(2026)

V_G, V_C = 0.31, 0.19          # Sackett et al. (2022, Table 4)
D_G, D_C = 0.79, -0.07         # Sackett et al. (2022); Roth et al. (2011); Dahlke & Sackett (2017)
P_MIN = 1 / 6                  # minority applicant-pool proportion (common convention)
P_FAKE = 0.40                  # Griffith et al. (2007) / Donovan et al. (2014): 30-50%
D_SHIFT = 0.45                 # Birkeland et al. (2006): applicant-incumbent C shift
W = 0.5                        # equal weights on standardized g and C

# context: (honest/incumbent rho0 [this study], applicant target r [this study])
# Pooled rho0 = Table 1 incumbent mean (.032, table1_meta_by_stakes.csv);
# corrected 2026-08-20 from .010, which was the total-database observed r.
CONTEXTS_MC = {
    "US":       (0.013, 0.007),
    "Pooled":   (0.032, -0.074),
    "Non-US":   (0.054, -0.272),
}
# Part A plug-in intercorrelations (label -> r12)
CONTEXTS_A = {
    "Berry et al. (2024) matrix (+.03)": 0.03,
    "US applicants (+.01)": 0.01,
    "Pooled applicants (-.07)": -0.074,
    "Non-US applicants (-.27)": -0.272,
}


# ---------------------------------------------------------------- Part A: analytic
def composite_params(r12: float, w: float = W) -> tuple[float, float]:
    sd = np.sqrt(w**2 + (1 - w) ** 2 + 2 * w * (1 - w) * r12)
    validity = (w * V_G + (1 - w) * V_C) / sd
    d_comp = (w * D_G + (1 - w) * D_C) / sd
    return validity, d_comp


def ai_ratio(d_comp: float, sr_total: float) -> float:
    """AI ratio for a mean-shifted normal composite, common cutoff, overall SR fixed."""
    f = lambda c: (1 - P_MIN) * stats.norm.sf(c) + P_MIN * stats.norm.sf(c + d_comp) - sr_total
    c = brentq(f, -5, 5)
    return stats.norm.sf(c + d_comp) / stats.norm.sf(c)


srs = np.linspace(0.05, 0.5, 46)
analytic_rows = []
for label, r12 in CONTEXTS_A.items():
    validity, d_comp = composite_params(r12)
    for sr in srs:
        cut = stats.norm.isf(sr)
        exp_perf = validity * stats.norm.pdf(cut) / sr  # E[performance | selected]
        analytic_rows.append(
            {"context": label, "selection_ratio": round(float(sr), 3),
             "composite_validity": round(validity, 3),
             "expected_performance_of_hires": round(float(exp_perf), 3),
             "AI_ratio": round(ai_ratio(d_comp, float(sr)), 3)}
        )
analytic = pd.DataFrame(analytic_rows)


# ---------------------------------------------------------------- Part B: Monte Carlo
def calibrate(rho0: float, target_r: float, sigma: float,
              p_fake: float = P_FAKE) -> tuple[float, float]:
    """Solve (mu_f, b) so the faked pool reproduces the Birkeland mean shift and
    the observed applicant C-GMA correlation (closed-form moments)."""

    def moments(params):
        mu_f, b = params
        var_infl = p_fake * (mu_f**2 + b**2 + sigma**2) - (p_fake * mu_f) ** 2
        var_cobs = 1 - 2 * p_fake * b * rho0 + var_infl
        r_gc = (rho0 - p_fake * b) / np.sqrt(var_cobs)
        # Cohen's d of C_obs vs honest C_true using pooled SD
        d = (p_fake * mu_f) / np.sqrt((1 + var_cobs) / 2)
        return [r_gc - target_r, d - D_SHIFT]

    mu_f, b = fsolve(moments, x0=[D_SHIFT / p_fake, max(0.0, (rho0 - target_r))])
    return float(mu_f), float(b)


def simulate(context: str, sr: float, sigma: float = 0.50, n_pool: int = 200_000,
             p_fake: float = P_FAKE) -> dict:
    rho0, target_r = CONTEXTS_MC[context]
    mu_f, b = calibrate(rho0, target_r, sigma, p_fake)

    minority = rng.random(n_pool) < P_MIN
    mu_g = np.where(minority, -D_G, 0.0)
    mu_c = np.where(minority, -D_C, 0.0)
    z = rng.multivariate_normal([0, 0], [[1, rho0], [rho0, 1]], size=n_pool)
    g, c_true = z[:, 0] + mu_g, z[:, 1] + mu_c

    # true performance: regression of y on (g, C_true) reproducing operational validities
    bcoef = np.linalg.solve(np.array([[1, rho0], [rho0, 1]]), np.array([V_G, V_C]))
    resid_sd = np.sqrt(1 - np.array([V_G, V_C]) @ bcoef)
    y = bcoef[0] * g + bcoef[1] * c_true + rng.normal(0, resid_sd, n_pool)

    fakes = rng.random(n_pool) < p_fake
    inflation = np.where(fakes, mu_f - b * g + rng.normal(0, sigma, n_pool), 0.0)
    c_obs = c_true + inflation

    comp_honest = W * g + (1 - W) * c_true
    comp_faked = W * g + (1 - W) * (c_obs - c_obs.mean()) / c_obs.std()  # standardized, as scored

    n_sel = int(sr * n_pool)
    sel_honest = np.argsort(-comp_honest)[:n_sel]
    sel_faked = np.argsort(-comp_faked)[:n_sel]
    overlap = len(np.intersect1d(sel_honest, sel_faked)) / n_sel

    def out(sel):
        return {
            "mean_true_performance": y[sel].mean(),
            "mean_C_true": c_true[sel].mean(),
            "AI_ratio": (minority[sel].mean() / P_MIN) / ((1 - minority[sel].mean()) / (1 - P_MIN)),
        }

    honest, faked = out(sel_honest), out(sel_faked)
    return {
        "context": context,
        "sigma_noise": sigma,
        "p_fake": p_fake,
        "selection_ratio": sr,
        "achieved_r_gC": round(float(np.corrcoef(g, c_obs)[0, 1]), 3),
        "achieved_d_shift": round(float(c_obs.mean() - c_true.mean())
                                  / float(np.sqrt((c_obs.var() + 1) / 2)), 3),
        "mu_f": round(mu_f, 3),
        "g_slope_b": round(b, 3),
        "pct_hires_changed": round(100 * (1 - overlap), 1),
        "perf_honest": round(honest["mean_true_performance"], 3),
        "perf_faked": round(faked["mean_true_performance"], 3),
        "perf_loss_pct": round(100 * (honest["mean_true_performance"] - faked["mean_true_performance"])
                               / honest["mean_true_performance"], 1),
        "mean_C_true_of_hires_honest": round(honest["mean_C_true"], 3),
        "mean_C_true_of_hires_faked": round(faked["mean_C_true"], 3),
        "AI_ratio_honest": round(honest["AI_ratio"], 3),
        "AI_ratio_faked": round(faked["AI_ratio"], 3),
        "rank_correlation_pool": round(float(stats.spearmanr(comp_honest, comp_faked).statistic), 3),
    }


mc_rows = [simulate(ctx, sr) for ctx in CONTEXTS_MC for sr in (0.10, 0.30)]
sens_rows = [simulate("Non-US", 0.10, sigma) for sigma in (0.25, 1.00)]
# New sensitivity runs appended AFTER the originals so the sequential module
# rng leaves all previously reported values bit-identical.
sens_us_pooled = [simulate(ctx, 0.10, sigma)
                  for ctx in ("US", "Pooled") for sigma in (0.25, 1.00)]
# 0.20 (the Griffith & Converse, 2011 synthesis lower bound) appended last
# (2026-08-25) so the sequential rng leaves the .30/.50 rows bit-identical.
prev_rows = [simulate(ctx, sr, p_fake=pf)
             for pf in (0.30, 0.50, 0.20)
             for ctx in CONTEXTS_MC for sr in (0.10, 0.30)]
mc = pd.DataFrame(mc_rows)
sens = pd.DataFrame(sens_rows)
sens2 = pd.DataFrame(sens_us_pooled)
prev = pd.DataFrame(prev_rows)

with open(OUT / "table7_selection_consequences.csv", "w", newline="") as f:
    f.write("# Part A: analytic selection curves (context-specific r12)\n")
    analytic.to_csv(f, index=False)
    f.write("\n# Part B: Monte Carlo compensatory-faking mechanism model "
            "(fit to Birkeland d=.45 and the applicant correlations; "
            "Griffith prevalence 40%, sigma=.50)\n")
    mc.to_csv(f, index=False)
    f.write("\n# Part B sensitivity: faking-noise sigma (Non-US calibration, SR=.10)\n")
    sens.to_csv(f, index=False)
    f.write("\n# Part B sensitivity: faking-noise sigma, US and Pooled "
            "(SR=.10; added 2026-08-21, triage 5b)\n")
    sens2.to_csv(f, index=False)
    f.write("\n# Part B sensitivity: faking prevalence at the within-person "
            "range endpoints {.30, .50} plus the Griffith & Converse (2011) "
            "synthesis lower bound .20 (appended 2026-08-25), (mu_f, b) "
            "refit per prevalence (added 2026-08-21, triage 5b)\n")
    prev.to_csv(f, index=False)

# ---------------------------------------------------------------- Figure (APA style)
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
greys = dict(zip(CONTEXTS_A, ["0.75", "0.55", "0.35", "0.0"]))
styles = dict(zip(CONTEXTS_A, [":", "-.", "--", "-"]))
for label in CONTEXTS_A:
    d = analytic[analytic.context == label]
    axes[0].plot(d.selection_ratio, d.expected_performance_of_hires,
                 color=greys[label], ls=styles[label], lw=1.8, label=label)
    axes[1].plot(d.selection_ratio, d.AI_ratio,
                 color=greys[label], ls=styles[label], lw=1.8, label=label)
axes[0].set_ylabel("Expected performance of hires (SD units)")
axes[1].set_ylabel("Adverse-impact ratio")
axes[1].axhline(0.8, color="gray", ls=":", lw=1)
axes[1].annotate("four-fifths rule", (0.33, 0.805), fontsize=8, color="gray")
for ax in axes[:2]:
    ax.set_xlabel("Selection ratio")
    ax.legend(fontsize=7, frameon=False)

x = np.arange(len(mc))
bar_labels = [f"{r.context}\nSR={r.selection_ratio:.0%}" for r in mc.itertuples()]
axes[2].bar(x - 0.2, mc.pct_hires_changed, width=0.4, color="0.3", label="% of hires changed")
axes[2].bar(x + 0.2, mc.perf_loss_pct, width=0.4, color="0.7", label="% performance loss")
axes[2].set_xticks(x, bar_labels, fontsize=7)
axes[2].set_ylabel("Percent")
axes[2].legend(fontsize=8, frameon=False)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "figure3_selection_effects.png", dpi=300, bbox_inches="tight")

print(mc.to_string(index=False))
print()
print(sens.to_string(index=False))
print()
print(analytic[analytic.selection_ratio.isin([0.1, 0.3])].to_string(index=False))
print("\nSaved table7_selection_consequences.csv and figure3_selection_effects.png")
