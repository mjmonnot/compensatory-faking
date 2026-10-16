"""Exploratory scan: applicant vs incumbent g-correlations for all sub-Big-Five
personality constructs (facets, aspects, compounds, meta-traits) with at least
MIN_K independent samples in both groups.

Output: output/facet_scan.csv (contrast per construct, sorted by |difference|).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from meta_lib import bare_bones, subgroup_contrast

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
MIN_K = 5
BIG_FIVE = ["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"]

db = pd.read_pickle(ROOT / "database" / "db.pkl")
sub = db[
    (db["Meta-Analyzed Cognitive Ability Construct"] == "g")
    & (~db["Meta-Analyzed Personality Construct"].isin(BIG_FIVE))
    & (db["Sample Type"].isin(["Applicant", "Employee"]))
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
    .rename(columns={"Meta-Analyzed Personality Construct": "construct"})
)

rows = []
for construct, d in agg.groupby("construct"):
    app = d[d["Sample Type"] == "Applicant"]
    emp = d[d["Sample Type"] == "Employee"]
    if len(app) < MIN_K or len(emp) < MIN_K:
        continue
    bb_a = bare_bones(app.r.values, app.N.values)
    bb_e = bare_bones(emp.r.values, emp.N.values)
    diff, z, p = subgroup_contrast(bb_a, bb_e)
    rows.append(
        {
            "construct": construct,
            "k_app": bb_a.k,
            "N_app": round(bb_a.total_n),
            "r_applicant": round(bb_a.mean_r, 3),
            "k_emp": bb_e.k,
            "N_emp": round(bb_e.total_n),
            "r_incumbent": round(bb_e.mean_r, 3),
            "diff": round(diff, 3),
            "z": round(z, 2),
            "p": round(p, 4),
        }
    )

res = pd.DataFrame(rows).sort_values("diff", key=abs, ascending=False)
res.to_csv(OUT / "facet_scan.csv", index=False)
print(res.to_string(index=False))
