"""Integrity-gap comparison: Ones (1993) vs Howard & Van Zandt (2020).

Same Study 2 machinery as 12_study2_expanded_model.py (primary spec
field_ns0, targeted and desirability-proportional inflation, SR .10 and
.30). Only the honesty-humility correlations change. Does not overwrite
study2_expanded_results.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import study2_params as P

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).with_name("12_study2_expanded_model.py")
text = SRC.read_text(encoding="utf-8")
text = text.split("# ---------------------------------------------------------------- run")[0]
ns = {"__file__": str(SRC), "__name__": "study2_expanded_model_funcs"}
exec(compile(text, str(SRC), "exec"), ns)

SOURCES = {
    # Blonigen et al. (2011) figures previously treated as Ones (1993).
    "cited_42_33": {"RHO_CH": 0.42, "RHO_ESH": 0.33, "RHO_AH": 0.45},
    # Sackett & Wanek (1996, Table 3) overt tests, corrected / observed,
    # as quoted by Lee, Ashton, & de Vries (2005, p. 182).
    "Sackett_overt_corrected": {"RHO_CH": 0.39, "RHO_ESH": 0.28, "RHO_AH": 0.34},
    "Sackett_overt_observed": {"RHO_CH": 0.26, "RHO_ESH": 0.18, "RHO_AH": 0.23},
    "Howard_2020": {"RHO_CH": 0.25, "RHO_ESH": 0.14, "RHO_AH": 0.50},
}

rows = []
for name, vals in SOURCES.items():
    for key, val in vals.items():
        setattr(P, key, val)
    ns["rng"] = np.random.default_rng(P.SEED)
    print(f"\n=== {name}  CH={P.RHO_CH} ESH={P.RHO_ESH} AH={P.RHO_AH} ===", flush=True)
    pools = {ctx: ns["build_pool"](ctx) for ctx in ("US", "Pooled", "Non-US")}
    ps = P.PRIMARY_PARAM_SET_12
    for ctx, pool in pools.items():
        F, _, ach = ns["solve_propensity"](pool, P.PARAM_SETS[ps], P.P_FAKE)
        for var in ("T", "D"):
            row, theta = ns["run_calibration"](pool, F, ps, var)
            scales = ns["apply_faking"](pool, F, theta, P.SIGMA_INFL, var)
            for sr in P.SRS:
                stats = ns["hire_stats"](pool, F, scales["c_obs"], sr)
                gap = stats["mean_H_displaced"] - stats["mean_H_faked_in"]
                out = {
                    "source": name,
                    "context": ctx,
                    "variant": var,
                    "selection_ratio": sr,
                    "weighted_distance": row["weighted_distance"],
                    "H_gap": round(gap, 3),
                    **{k: round(v, 3) if isinstance(v, float) else v
                       for k, v in stats.items()},
                }
                rows.append(out)
                if var == "T":
                    print(
                        f"  {ctx:7} SR {sr:.2f}  changed {stats['pct_hires_changed']:.1f}%"
                        f"  loss {stats['perf_loss_pct_Hpath']:.1f}%"
                        f"  H gap {gap:.3f}"
                        f"  ({stats['mean_H_faked_in']:.3f} vs {stats['mean_H_displaced']:.3f})"
                        f"  dist {row['weighted_distance']}",
                        flush=True,
                    )

df = pd.DataFrame(rows)
dest = ROOT / "output" / "hh_source_comparison.csv"
df.to_csv(dest, index=False)
print("\nSaved", dest)
