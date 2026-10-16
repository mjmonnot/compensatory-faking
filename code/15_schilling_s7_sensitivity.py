"""Sensitivity: add Schilling et al. (2021) Table S7 mega-samples to our applicant cells.

Schilling et al. excluded six very large operational applicant datasets from
their meta-analysis so single samples would not dominate the averages, but
published the correlations in Supporting Information Table S7
(literature/Schilling_2021_Supinfo.docx). Five of the six do not appear in the
Stanek-Ones database (checked by first author). De Fruyt (2006) DOES appear
there (French applicant sample, ID 440, N = 20,183 vs. 19,365 in Schilling's
transcription) but is coded Fluid--Induction rather than g, so it falls
outside the manuscript's inclusion criteria; appending it here is additive
relative to the analyzed g-only slice, not to the raw database (pipeline
audit F1, 2026-08-20).

This script recomputes our bare-bones applicant estimates with those samples
appended, overall and within region. Values below are transcribed from Table
S7 (mixed-CA column where available, otherwise the reported column). All are
field applicant samples per the paper's exclusion-criteria description.

Output: output/schilling_s7_sensitivity.csv plus console report.
"""

import pandas as pd

from meta_lib import bare_bones

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

# (study, year, country_group, trait, N, r)
S7 = [
    ("Levashina, Weekley, Roulin, & Hauck", 2014, "US", "Conscientiousness", 65651, .089),
    ("Levashina, Weekley, Roulin, & Hauck", 2014, "US", "Extraversion", 69909, .102),
    ("Levashina, Weekley, Roulin, & Hauck", 2014, "US", "Openness", 26197, .169),
    ("MacKenzie, Ployhart, Weekley, & Ehlers", 2010, "US", "Agreeableness", 20880, .060),
    ("MacKenzie, Ployhart, Weekley, & Ehlers", 2010, "US", "Conscientiousness", 20880, .030),
    ("MacKenzie, Ployhart, Weekley, & Ehlers", 2010, "US", "Extraversion", 20880, .055),
    ("Delgado, Tristan, Kung, & O'Connell", 2011, "US", "Agreeableness", 28280, .150),
    ("Delgado, Tristan, Kung, & O'Connell", 2011, "US", "Conscientiousness", 28280, .178),
    ("Huang, Wells, & Nguyen", 2015, "US", "Agreeableness", 62020, .002),
    ("Huang, Wells, & Nguyen", 2015, "US", "Conscientiousness", 62020, .020),
    ("Huang, Wells, & Nguyen", 2015, "US", "Neuroticism", 62020, .035),  # ES -.035 reversed
    ("Huang, Wells, & Nguyen", 2015, "US", "Extraversion", 62020, .066),
    ("Huang, Wells, & Nguyen", 2015, "US", "Openness", 62020, .234),
    ("Arthur, Doverspike, Munoz, Taylor, & Carr (s88)", 2014, "US", "Agreeableness", 2006239, .060),
    ("Arthur, Doverspike, Munoz, Taylor, & Carr (s88)", 2014, "US", "Conscientiousness", 2006239, .050),
    ("Arthur, Doverspike, Munoz, Taylor, & Carr (s88)", 2014, "US", "Neuroticism", 2006239, -.070),  # ES .070 reversed
    ("Arthur, Doverspike, Munoz, Taylor, & Carr (s88)", 2014, "US", "Extraversion", 2006239, .040),
    ("Arthur, Doverspike, Munoz, Taylor, & Carr (s88)", 2014, "US", "Openness", 2006239, .250),
    ("Arthur, Doverspike, Munoz, Taylor, & Carr (s89)", 2014, "US", "Agreeableness", 36550, .100),
    ("Arthur, Doverspike, Munoz, Taylor, & Carr (s89)", 2014, "US", "Conscientiousness", 36550, .070),
    ("Arthur, Doverspike, Munoz, Taylor, & Carr (s89)", 2014, "US", "Neuroticism", 36550, -.120),  # ES .120 reversed
    ("Arthur, Doverspike, Munoz, Taylor, & Carr (s89)", 2014, "US", "Extraversion", 36550, .080),
    ("Arthur, Doverspike, Munoz, Taylor, & Carr (s89)", 2014, "US", "Openness", 36550, .190),
    ("De Fruyt, Aluja, Garcia, Rolland, & Jung", 2006, "non-US", "Agreeableness", 19365, .113),
    ("De Fruyt, Aluja, Garcia, Rolland, & Jung", 2006, "non-US", "Conscientiousness", 19365, .104),
    ("De Fruyt, Aluja, Garcia, Rolland, & Jung", 2006, "non-US", "Neuroticism", 19365, -.167),  # ES .167 reversed
    ("De Fruyt, Aluja, Garcia, Rolland, & Jung", 2006, "non-US", "Extraversion", 19365, .114),
    ("De Fruyt, Aluja, Garcia, Rolland, & Jung", 2006, "non-US", "Openness", 19365, .111),
]

mega = pd.DataFrame(S7, columns=["study", "year", "region", "trait", "N", "r"])

df = pd.read_csv(ROOT / "data" / "analysis_effects.csv")
app = df[df.stakes == "High stakes (applicants)"].copy()
app["region"] = (app.country == "United States of America").map(
    {True: "US", False: "non-US"})


def bb_row(label, r, n):
    b = bare_bones(r.to_numpy(), n.to_numpy())
    return {"scenario": label, "k": b.k, "N": int(b.total_n),
            "mean_r": round(b.mean_r, 3),
            "ci_lo": round(b.ci_lo, 3), "ci_hi": round(b.ci_hi, 3)}


rows = []
for trait in ["Conscientiousness", "Neuroticism", "Openness",
              "Extraversion", "Agreeableness"]:
    a = app[app.trait == trait]
    m = mega[mega.trait == trait]
    m_cap = m[m.N < 1_000_000]  # Schilling-style: drop the 2M Arthur sample
    rows.append(bb_row(f"{trait} | ours", a.r, a.N))
    rows.append(bb_row(f"{trait} | ours + megas (excl. 2M sample)",
                       pd.concat([a.r, m_cap.r]), pd.concat([a.N, m_cap.N])))
    rows.append(bb_row(f"{trait} | ours + all megas",
                       pd.concat([a.r, m.r]), pd.concat([a.N, m.N])))
    if trait == "Conscientiousness":
        for reg in ["US", "non-US"]:
            ar, mr = a[a.region == reg], m_cap[m_cap.region == reg]
            rows.append(bb_row(f"{trait} | {reg} ours", ar.r, ar.N))
            rows.append(bb_row(f"{trait} | {reg} ours + megas (excl. 2M)",
                               pd.concat([ar.r, mr.r]), pd.concat([ar.N, mr.N])))

res = pd.DataFrame(rows)
res.to_csv(OUT / "schilling_s7_sensitivity.csv", index=False)
print(res.to_string(index=False))
print("\nSaved", OUT / "schilling_s7_sensitivity.csv")
