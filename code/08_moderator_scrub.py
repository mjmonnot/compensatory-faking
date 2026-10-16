"""Additional moderator analyses from remaining database fields.

Within the applicant (high-stakes) g x Big Five pool:
  1. Published (journal) vs unpublished (dissertation/thesis/raw data/report) sources.
  2. US vs non-US samples.
  3. Study era (median split on year).
  4. Measure family: general research inventories (NEO/BFI) vs occupational inventories.

Across all stakes groups (updated-inputs question):
  5. Era trends: mean r by half-century era for each trait, applicant vs low-stakes pooled.

Output: output/moderator_scrub.csv (one row per contrast), output/era_trends.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from meta_lib import bare_bones, subgroup_contrast

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
BIG_FIVE = ["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"]
MIN_K = 4

db = pd.read_pickle(ROOT / "database" / "db.pkl")
sub = db[
    (db["Meta-Analyzed Cognitive Ability Construct"] == "g")
    & (db["Meta-Analyzed Personality Construct"].isin(BIG_FIVE))
].copy()
flip_p = sub["Personality Construct Require Reversal?"].astype(str).str.strip().str.lower() == "reverse"
flip_c = sub["Cognitive Ability Construct Require Reversal?"].astype(str).str.strip().str.lower() == "reverse"
sub["r_aligned"] = sub["r"] * np.where(flip_p, -1, 1) * np.where(flip_c, -1, 1)
sub = sub.dropna(subset=["r_aligned", "N"])
sub = sub[sub["N"] >= 10]

sub["published"] = np.where(sub["Type of Publication"] == "Journal Article", "published", "unpublished")
sub["us"] = np.where(sub["Country"] == "United States of America", "US", "non-US")
sub["year_num"] = pd.to_numeric(sub["Year"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
research_inv = sub["Personality Measure and Scale"].str.contains("NEO|Big Five Inventory", case=False, na=False)
sub["measure_family"] = np.where(research_inv, "research inventory", "occupational inventory")

agg = (
    sub.groupby(["Meta-Analysis Sample ID", "Meta-Analyzed Personality Construct"])
    .agg(
        r=("r_aligned", "mean"),
        N=("N", "median"),
        sample_type=("Sample Type", lambda x: x.mode().iat[0]),
        published=("published", lambda x: x.mode().iat[0]),
        us=("us", lambda x: x.mode().iat[0]),
        year=("year_num", "median"),
        measure_family=("measure_family", lambda x: x.mode().iat[0]),
    )
    .reset_index()
    .rename(columns={"Meta-Analyzed Personality Construct": "trait"})
)

app = agg[agg.sample_type == "Applicant"].copy()
year_median = app.year.median()
app["era"] = np.where(app.year <= year_median, f"<= {int(year_median)}", f"> {int(year_median)}")

rows = []
moderators = [("published", ["published", "unpublished"]), ("us", ["US", "non-US"]),
              ("era", [f"<= {int(year_median)}", f"> {int(year_median)}"]),
              ("measure_family", ["research inventory", "occupational inventory"])]
for trait in BIG_FIVE:
    d = app[app.trait == trait]
    for mod, (lv1, lv2) in moderators:
        d1, d2 = d[d[mod] == lv1], d[d[mod] == lv2]
        if len(d1) < MIN_K or len(d2) < MIN_K:
            continue
        bb1, bb2 = bare_bones(d1.r.values, d1.N.values), bare_bones(d2.r.values, d2.N.values)
        diff, z, p = subgroup_contrast(bb1, bb2)
        rows.append(
            {
                "trait": trait, "moderator": mod,
                "level1": lv1, "k1": bb1.k, "r1": round(bb1.mean_r, 3),
                "level2": lv2, "k2": bb2.k, "r2": round(bb2.mean_r, 3),
                "diff": round(diff, 3), "z": round(z, 2), "p": round(p, 4),
            }
        )
res = pd.DataFrame(rows)
res.to_csv(OUT / "moderator_scrub.csv", index=False)
print("=== Within-applicant moderator contrasts (min k per cell =", MIN_K, ") ===")
print(res.to_string(index=False) if len(res) else "none reached minimum k")

# Era trends across stakes groups (updated-inputs question)
agg["low_stakes"] = agg.sample_type.isin(["Employee", "Student", "Community"])
era_rows = []
for trait in BIG_FIVE:
    for grp, mask in [("applicant", agg.sample_type == "Applicant"), ("low_stakes", agg.low_stakes)]:
        d = agg[(agg.trait == trait) & mask].dropna(subset=["year"])
        for lo, hi, label in [(0, 1990, "pre-1990"), (1990, 2004, "1990-2004"), (2004, 2100, "2005-2013")]:
            dd = d[(d.year > lo) & (d.year <= hi)]
            if len(dd) < MIN_K:
                continue
            bb = bare_bones(dd.r.values, dd.N.values)
            era_rows.append(
                {"trait": trait, "group": grp, "era": label, "k": bb.k,
                 "N": round(bb.total_n), "mean_r": round(bb.mean_r, 3),
                 "CI95_lo": round(bb.ci_lo, 3), "CI95_hi": round(bb.ci_hi, 3)}
            )
era = pd.DataFrame(era_rows)
era.to_csv(OUT / "era_trends.csv", index=False)
print()
print("=== Era trends ===")
print(era.to_string(index=False))

# Applicant vs incumbent contrast within region (manuscript Table 6)
region_rows = []
for trait in BIG_FIVE:
    for region in ["US", "non-US"]:
        a = agg[(agg.trait == trait) & (agg.sample_type == "Applicant") & (agg.us == region)]
        e = agg[(agg.trait == trait) & (agg.sample_type == "Employee") & (agg.us == region)]
        if len(a) < MIN_K or len(e) < MIN_K:
            continue
        bb_a, bb_e = bare_bones(a.r.values, a.N.values), bare_bones(e.r.values, e.N.values)
        diff, z, p = subgroup_contrast(bb_a, bb_e)
        region_rows.append(
            {"trait": trait, "region": region,
             "r_applicant": round(bb_a.mean_r, 3), "k_app": bb_a.k, "N_app": round(bb_a.total_n),
             "r_incumbent": round(bb_e.mean_r, 3), "k_emp": bb_e.k, "N_emp": round(bb_e.total_n),
             "diff": round(diff, 3), "z": round(z, 2), "p": round(p, 4)}
        )
reg = pd.DataFrame(region_rows)
reg.to_csv(OUT / "region_contrasts.csv", index=False)
print()
print("=== Applicant vs incumbent within region ===")
print(reg.to_string(index=False))
