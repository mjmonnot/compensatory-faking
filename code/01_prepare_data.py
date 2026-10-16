"""Prepare analysis dataset for Manuscript A: personality-GMA relations in high-stakes contexts.

Reads the Stanek & Ones (2023) personality-ability relations database (prepared pickle),
applies construct reversals per the database legend, restricts to g x Big Five effects,
codes assessment stakes from sample type, and composites statistically dependent effect
sizes (same construct pair within the same Meta-Analysis Sample ID).

Output: data/analysis_effects.csv  (one row = one independent sample x construct pair)
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "database" / "db.pkl"
OUT_DIR = ROOT / "data"
OUT_DIR.mkdir(exist_ok=True)

BIG_FIVE = [
    "Neuroticism",
    "Extraversion",
    "Openness",
    "Agreeableness",
    "Conscientiousness",
]

# Stakes coding rationale:
# - Applicant samples completed assessments as part of real selection decisions -> high stakes.
# - Employee (incumbent), student, and community samples completed assessments for
#   research/development purposes -> low stakes.
# - Military samples mix entry screening and research administrations; kept as a separate
#   category and examined in sensitivity analyses rather than forced into either group.
STAKES_MAP = {
    "Applicant": "High stakes (applicants)",
    "Employee": "Low stakes (incumbents)",
    "Student": "Low stakes (students)",
    "Community": "Low stakes (community)",
    "Military": "Military (mixed stakes)",
}


def main() -> None:
    db = pd.read_pickle(DB_PATH)

    sub = db[
        (db["Meta-Analyzed Cognitive Ability Construct"] == "g")
        & (db["Meta-Analyzed Personality Construct"].isin(BIG_FIVE))
    ].copy()

    # Apply reversals so all effects align with the named constructs.
    flip_p = sub["Personality Construct Require Reversal?"].astype(str).str.strip().str.lower() == "reverse"
    flip_c = sub["Cognitive Ability Construct Require Reversal?"].astype(str).str.strip().str.lower() == "reverse"
    sub["r_aligned"] = sub["r"] * np.where(flip_p, -1, 1) * np.where(flip_c, -1, 1)

    sub = sub.dropna(subset=["r_aligned", "N"])
    sub = sub[sub["N"] >= 10]

    sub["stakes"] = sub["Sample Type"].map(STAKES_MAP).fillna("Other/unreported")

    pm = sub["% Male"].astype(str).str.rstrip("?")
    sub["pct_male"] = pd.to_numeric(pm, errors="coerce")
    sub["age_mean"] = pd.to_numeric(sub["Age Mean"].astype(str).str.rstrip("?"), errors="coerce")
    sub["year_num"] = pd.to_numeric(sub["Year"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")

    # Composite dependent effect sizes: average r within sample x trait (unit weights across
    # measures of the same constructs within a sample), N = median reported N for that sample.
    grp_cols = ["Meta-Analysis Sample ID", "Meta-Analyzed Personality Construct"]
    agg = (
        sub.groupby(grp_cols)
        .agg(
            r=("r_aligned", "mean"),
            N=("N", "median"),
            n_effects=("r_aligned", "size"),
            stakes=("stakes", lambda x: x.mode().iat[0]),
            sample_type=("Sample Type", lambda x: x.mode().iat[0]),
            country=("Country", lambda x: x.mode().iat[0]),
            year=("year_num", "median"),
            pct_male=("pct_male", "median"),
            age_mean=("age_mean", "median"),
            first_author=("Author", "first"),
        )
        .reset_index()
        .rename(columns={"Meta-Analyzed Personality Construct": "trait"})
    )

    agg.to_csv(OUT_DIR / "analysis_effects.csv", index=False)

    print(f"Effect-size rows before compositing: {len(sub)}")
    print(f"Independent sample x trait rows: {len(agg)}")
    print()
    print(agg.groupby(["trait", "stakes"]).agg(k=("r", "size"), N=("N", "sum")).unstack("stakes"))


if __name__ == "__main__":
    main()
