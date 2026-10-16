"""Illustrative validity-diversity simulation: consequences of using high-stakes vs
low-stakes Conscientiousness-GMA intercorrelation inputs in a GMA + C composite.

Standardized two-predictor composite with weight w on GMA and (1 - w) on C:
  validity  R(w)  = (w*v_gma + (1-w)*v_c) / sqrt(w^2 + (1-w)^2 + 2*w*(1-w)*r12)
  subgroup  d(w)  = (w*d_gma + (1-w)*d_c) / same denominator
Input values (verified against published sources):
  v_gma = .31, v_c = .19   (operational validities, Sackett et al., 2022, Table 4)
  d_gma = .79, d_c = -.07  (Black-White standardized differences, Sackett et al., 2022,
                            Table 4; d_gma from Roth et al., 2011 applicant samples,
                            d_c from Dahlke & Sackett, 2017; positive = White higher)
Compared inputs (both reliability-corrected, Table 3 Panel B): r12 = +.01 (total-database
rho, all contexts pooled) vs r12 = -.09 (pooled applicant rho). Labels renamed 2026-08-21
(review triage 7b): the +.01 scenario was previously mislabeled "low-stakes" -- the
corrected low-stakes value is -.01; +.01 is the total-database value.

Outputs: output/table4_simulation.csv, output/figure2_simulation.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

V_GMA, V_C = 0.31, 0.19
D_GMA, D_C = 0.79, -0.07
SCENARIOS = {"Total-database input (r12 = +.01)": 0.01, "Pooled applicant input (r12 = -.09)": -0.09}


def composite(w: np.ndarray, r12: float) -> tuple[np.ndarray, np.ndarray]:
    denom = np.sqrt(w**2 + (1 - w) ** 2 + 2 * w * (1 - w) * r12)
    return (w * V_GMA + (1 - w) * V_C) / denom, (w * D_GMA + (1 - w) * D_C) / denom


w = np.linspace(0, 1, 101)
rows = []
for label, r12 in SCENARIOS.items():
    R, d = composite(w, r12)
    r2_opt = (V_GMA**2 + V_C**2 - 2 * V_GMA * V_C * r12) / (1 - r12**2)
    for wi in (0.0, 0.25, 0.5, 0.75, 1.0):
        Ri, di = composite(np.array([wi]), r12)
        rows.append(
            {
                "scenario": label,
                "w_gma": wi,
                "composite_validity": round(float(Ri[0]), 3),
                "composite_d": round(float(di[0]), 3),
            }
        )
    rows.append(
        {
            "scenario": label,
            "w_gma": "optimal (regression) R",
            "composite_validity": round(np.sqrt(r2_opt), 3),
            "composite_d": np.nan,
        }
    )
pd.DataFrame(rows).to_csv(OUT / "table4_simulation.csv", index=False)

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
styles = {"Total-database input (r12 = +.01)": ("0.55", "--"), "Pooled applicant input (r12 = -.09)": ("0.0", "-")}
for label, r12 in SCENARIOS.items():
    R, d = composite(w, r12)
    color, ls = styles[label]
    axes[0].plot(w, R, label=label, color=color, ls=ls, lw=1.8)
    axes[1].plot(w, d, label=label, color=color, ls=ls, lw=1.8)
axes[0].set_ylabel("Composite validity (R)")
axes[1].set_ylabel("Composite Black\u2013White d")
for ax in axes:
    ax.set_xlabel("Weight on GMA (w)")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "figure2_simulation.png", dpi=300, bbox_inches="tight")

lo = pd.DataFrame(rows)
print(lo.to_string(index=False))
print("\nSaved table4_simulation.csv and figure2_simulation.png")
