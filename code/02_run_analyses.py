"""Run the three linked analyses for Manuscript A and write results tables.

Analysis 1: g x Big Five meta-analytic correlations by assessment stakes.
Analysis 2: compensation-hypothesis adjudication (applicant vs incumbent contrasts,
            plus meta-regression controls and DL random-effects sensitivity).
Analysis 3: updated GMA-Big Five intercorrelation inputs for high-stakes selection
            contexts (observed and measurement-error-corrected).

Outputs (output/):
  table1_meta_by_stakes.csv
  table2_contrasts.csv
  table3_intercorrelation_inputs.csv
  sensitivity_dl_random_effects.csv
  metareg_stakes_controls.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from meta_lib import bare_bones, dl_random_effects, subgroup_contrast

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

TRAITS = ["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"]
GROUPS = [
    "High stakes (applicants)",
    "Low stakes (incumbents)",
    "Low stakes (students)",
    "Low stakes (community)",
    "Military (mixed stakes)",
]

df = pd.read_csv(ROOT / "data" / "analysis_effects.csv")

# ---------------------------------------------------------------- Analysis 1
rows = []
results: dict[tuple[str, str], object] = {}
for trait in TRAITS:
    for grp in GROUPS + ["Low stakes (all)"]:
        if grp == "Low stakes (all)":
            d = df[(df.trait == trait) & (df.stakes.str.startswith("Low stakes"))]
        else:
            d = df[(df.trait == trait) & (df.stakes == grp)]
        if len(d) < 3:
            continue
        bb = bare_bones(d.r.values, d.N.values)
        results[(trait, grp)] = bb
        rho, cv_lo_c, cv_hi_c = bb.corrected()
        rows.append(
            {
                "trait": trait,
                "group": grp,
                "k": bb.k,
                "N": round(bb.total_n),
                "mean_r": round(bb.mean_r, 3),
                "SD_r": round(bb.sd_r, 3),
                "SD_res": round(bb.sd_res, 3),
                "CI95_lo": round(bb.ci_lo, 3),
                "CI95_hi": round(bb.ci_hi, 3),
                "CV80_lo": round(bb.cv_lo, 3),
                "CV80_hi": round(bb.cv_hi, 3),
                "rho_corrected": round(rho, 3),
            }
        )
pd.DataFrame(rows).to_csv(OUT / "table1_meta_by_stakes.csv", index=False)

# ---------------------------------------------------------------- Analysis 2
contrast_rows = []
comparisons = [
    ("Low stakes (incumbents)", "applicant_vs_incumbent"),
    ("Low stakes (all)", "applicant_vs_all_low_stakes"),
]
for trait in TRAITS:
    hi = results.get((trait, "High stakes (applicants)"))
    if hi is None:
        continue
    for comp_grp, label in comparisons:
        lo = results.get((trait, comp_grp))
        if lo is None:
            continue
        diff, z, p = subgroup_contrast(hi, lo)
        contrast_rows.append(
            {
                "trait": trait,
                "comparison": label,
                "r_high_stakes": round(hi.mean_r, 3),
                "r_comparison": round(lo.mean_r, 3),
                "diff": round(diff, 3),
                "z": round(z, 2),
                "p": round(p, 4),
            }
        )
pd.DataFrame(contrast_rows).to_csv(OUT / "table2_contrasts.csv", index=False)

# DL random-effects sensitivity (Fisher z), applicant and incumbent groups.
sens_rows = []
for trait in TRAITS:
    for grp in ["High stakes (applicants)", "Low stakes (incumbents)", "Low stakes (all)"]:
        if grp == "Low stakes (all)":
            d = df[(df.trait == trait) & (df.stakes.str.startswith("Low stakes"))]
        else:
            d = df[(df.trait == trait) & (df.stakes == grp)]
        if len(d) < 3:
            continue
        m, lo, hi, tau = dl_random_effects(d.r.values, d.N.values)
        sens_rows.append(
            {
                "trait": trait,
                "group": grp,
                "k": len(d),
                "mean_r_DL": round(m, 3),
                "CI95_lo": round(lo, 3),
                "CI95_hi": round(hi, 3),
                "tau_fisherz": round(tau, 3),
            }
        )
pd.DataFrame(sens_rows).to_csv(OUT / "sensitivity_dl_random_effects.csv", index=False)

# Meta-regression: does the stakes effect survive controls (year, % male, mean age)?
metareg_rows = []
sub = df[df.stakes.isin(["High stakes (applicants)", "Low stakes (incumbents)"])].copy()
sub["high_stakes"] = (sub.stakes == "High stakes (applicants)").astype(int)
for trait in TRAITS:
    d = sub[sub.trait == trait].copy()
    d = d.dropna(subset=["r", "N"])
    for spec, cols in [("stakes_only", ["high_stakes"]), ("with_controls", ["high_stakes", "year", "pct_male", "age_mean"])]:
        dd = d.dropna(subset=cols)
        if len(dd) < len(cols) + 3:
            continue
        X = sm.add_constant(dd[cols])
        model = sm.WLS(dd.r, X, weights=dd.N).fit(cov_type="HC1")
        metareg_rows.append(
            {
                "trait": trait,
                "spec": spec,
                "k": len(dd),
                "b_high_stakes": round(model.params["high_stakes"], 3),
                "se": round(model.bse["high_stakes"], 3),
                "p": round(model.pvalues["high_stakes"], 4),
            }
        )
pd.DataFrame(metareg_rows).to_csv(OUT / "metareg_stakes_controls.csv", index=False)

# ---------------------------------------------------------------- Analysis 3
# GMA-Big Five intercorrelation inputs for validity-diversity simulations:
# high-stakes (applicant) cells vs pooled low-stakes cells vs full database.
input_rows = []
for trait in TRAITS:
    row = {"trait": trait}
    for grp, label in [
        ("High stakes (applicants)", "high_stakes"),
        ("Low stakes (all)", "low_stakes"),
    ]:
        bb = results.get((trait, grp))
        if bb is None:
            continue
        rho, _, _ = bb.corrected()
        row[f"r_obs_{label}"] = round(bb.mean_r, 3)
        row[f"rho_{label}"] = round(rho, 3)
        row[f"k_{label}"] = bb.k
        row[f"N_{label}"] = round(bb.total_n)
    d_all = df[df.trait == trait]
    bb_all = bare_bones(d_all.r.values, d_all.N.values)
    rho_all, _, _ = bb_all.corrected()
    row["r_obs_total_db"] = round(bb_all.mean_r, 3)
    row["rho_total_db"] = round(rho_all, 3)
    row["k_total_db"] = bb_all.k
    input_rows.append(row)
pd.DataFrame(input_rows).to_csv(OUT / "table3_intercorrelation_inputs.csv", index=False)

print("All analyses complete. Tables written to output/.")
for f in sorted(OUT.glob("*.csv")):
    print(" -", f.name)
