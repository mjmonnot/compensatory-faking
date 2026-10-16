"""Pre-submission robustness checks for Manuscript A.

1. Leave-one-out for the applicant Conscientiousness-g pool (k = 20).
2. Publication-bias diagnostics (Egger regression, PET-PEESE) for applicant pools.
3. Reliability-correction sensitivity grid for high-stakes rho estimates.
4. Exploratory facet-level analysis (C-Achievement, C-Dependability; applicant vs incumbent).
5. Applicant-sample audit list for manual verification against primary studies.

Outputs (output/):
  check_leave_one_out.csv, check_pub_bias.csv, check_reliability_grid.csv,
  facet_conscientiousness.csv, audit_applicant_samples.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from meta_lib import bare_bones, subgroup_contrast

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
TRAITS = ["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"]

df = pd.read_csv(ROOT / "data" / "analysis_effects.csv")

# ------------------------------------------------ 1. Leave-one-out (applicant C-g)
d = df[(df.trait == "Conscientiousness") & (df.stakes == "High stakes (applicants)")].reset_index(drop=True)
loo_rows = []
full = bare_bones(d.r.values, d.N.values)
for i in range(len(d)):
    keep = d.drop(index=i)
    bb = bare_bones(keep.r.values, keep.N.values)
    loo_rows.append(
        {
            "omitted_sample": f"{d.first_author.iat[i]} ({d['Meta-Analysis Sample ID'].iat[i]})",
            "omitted_r": round(d.r.iat[i], 3),
            "omitted_N": round(d.N.iat[i]),
            "mean_r_without": round(bb.mean_r, 3),
            "CI95_lo": round(bb.ci_lo, 3),
            "CI95_hi": round(bb.ci_hi, 3),
        }
    )
loo = pd.DataFrame(loo_rows)
loo.to_csv(OUT / "check_leave_one_out.csv", index=False)
print("Leave-one-out (applicant C-g): full mean r =", round(full.mean_r, 3))
print("  range of mean r omitting one sample:", loo.mean_r_without.min(), "to", loo.mean_r_without.max())
print("  k with CI95 excluding 0:", int((loo.CI95_hi < 0).sum()), "of", len(loo))

# ------------------------------------------------ 2. Publication bias (applicant pools)
bias_rows = []
for trait in TRAITS:
    d = df[(df.trait == trait) & (df.stakes == "High stakes (applicants)")]
    if len(d) < 10:
        bias_rows.append({"trait": trait, "k": len(d), "note": "k too small for bias tests"})
        continue
    y = np.arctanh(np.clip(d.r.values, -0.999, 0.999))
    se = 1.0 / np.sqrt(d.N.values - 3)
    # Egger: precision-weighted regression of effect on SE; intercept-free asymmetry slope test
    X = sm.add_constant(se)
    egger = sm.WLS(y, X, weights=1 / se**2).fit()
    # PET and PEESE intercepts (both reported; no conditional switching)
    pet_b0 = egger.params[0]
    Xp = sm.add_constant(se**2)
    peese = sm.WLS(y, Xp, weights=1 / se**2).fit()
    bias_rows.append(
        {
            "trait": trait,
            "k": len(d),
            "egger_slope": round(egger.params[1], 3),
            "egger_slope_p": round(egger.pvalues[1], 4),
            "PET_intercept_r": round(np.tanh(pet_b0), 3),
            "PEESE_intercept_r": round(np.tanh(peese.params[0]), 3),
            "note": "",
        }
    )
pd.DataFrame(bias_rows).to_csv(OUT / "check_pub_bias.csv", index=False)

# ------------------------------------------------ 3. Reliability sensitivity grid
grid_rows = []
for trait in TRAITS:
    d = df[(df.trait == trait) & (df.stakes == "High stakes (applicants)")]
    bb = bare_bones(d.r.values, d.N.values)
    for rxx_p in (0.75, 0.80, 0.85):
        for rxx_c in (0.80, 0.85, 0.90):
            grid_rows.append(
                {
                    "trait": trait,
                    "rxx_personality": rxx_p,
                    "rxx_gma": rxx_c,
                    "rho": round(bb.mean_r / np.sqrt(rxx_p * rxx_c), 3),
                }
            )
pd.DataFrame(grid_rows).to_csv(OUT / "check_reliability_grid.csv", index=False)

# ------------------------------------------------ 4. Facet-level (exploratory)
db = pd.read_pickle(ROOT / "database" / "db.pkl")
FACETS = ["C--Achievement", "C--Dependability"]
sub = db[
    (db["Meta-Analyzed Cognitive Ability Construct"] == "g")
    & (db["Meta-Analyzed Personality Construct"].isin(FACETS))
].copy()
flip_p = sub["Personality Construct Require Reversal?"].astype(str).str.strip().str.lower() == "reverse"
flip_c = sub["Cognitive Ability Construct Require Reversal?"].astype(str).str.strip().str.lower() == "reverse"
sub["r_aligned"] = sub["r"] * np.where(flip_p, -1, 1) * np.where(flip_c, -1, 1)
sub = sub.dropna(subset=["r_aligned", "N"])
sub = sub[sub["N"] >= 10]
agg = (
    sub.groupby(["Meta-Analysis Sample ID", "Meta-Analyzed Personality Construct", "Sample Type"])
    .agg(r=("r_aligned", "mean"), N=("N", "median"))
    .reset_index()
)
facet_rows = []
for facet in FACETS:
    res = {}
    for st in ["Applicant", "Employee"]:
        d = agg[(agg["Meta-Analyzed Personality Construct"] == facet) & (agg["Sample Type"] == st)]
        if len(d) < 3:
            continue
        bb = bare_bones(d.r.values, d.N.values)
        res[st] = bb
        facet_rows.append(
            {
                "facet": facet,
                "group": st,
                "k": bb.k,
                "N": round(bb.total_n),
                "mean_r": round(bb.mean_r, 3),
                "CI95_lo": round(bb.ci_lo, 3),
                "CI95_hi": round(bb.ci_hi, 3),
            }
        )
    if "Applicant" in res and "Employee" in res:
        diff, z, p = subgroup_contrast(res["Applicant"], res["Employee"])
        facet_rows.append(
            {"facet": facet, "group": "contrast (App - Emp)", "k": np.nan, "N": np.nan,
             "mean_r": round(diff, 3), "z": round(z, 2), "p": round(p, 4)}
        )
fac = pd.DataFrame(facet_rows)
fac.to_csv(OUT / "facet_conscientiousness.csv", index=False)
print()
print("Facet analysis (contrast rows: mean_r = difference, with z and p):")
print(fac.to_string(index=False))

# ------------------------------------------------ 5. Applicant audit list
b5 = ["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"]
app = db[
    (db["Meta-Analyzed Cognitive Ability Construct"] == "g")
    & (db["Meta-Analyzed Personality Construct"].isin(b5))
    & (db["Sample Type"] == "Applicant")
][
    [
        "Author", "Year", "Type of Publication", "Source Name", "Country",
        "Meta-Analysis Sample ID", "N", "r", "Meta-Analyzed Personality Construct",
        "Personality Measure and Scale", "Cognitive Ability Measure and Scale",
    ]
].sort_values(["Meta-Analysis Sample ID", "Meta-Analyzed Personality Construct"])
app.to_csv(OUT / "audit_applicant_samples.csv", index=False)
print()
print(f"Audit list: {app['Meta-Analysis Sample ID'].nunique()} unique applicant samples, {len(app)} effect rows")
