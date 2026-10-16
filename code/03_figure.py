"""Figure 1: meta-analytic g x Big Five correlations by assessment stakes (95% CIs)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

t1 = pd.read_csv(OUT / "table1_meta_by_stakes.csv")

groups = [
    ("High stakes (applicants)", "#c0392b"),
    ("Low stakes (incumbents)", "#2471a3"),
    ("Low stakes (students)", "#7fb3d5"),
    ("Low stakes (community)", "#a9cce3"),
    ("Military (mixed stakes)", "#7d7d7d"),
]
traits = ["Conscientiousness", "Neuroticism", "Openness", "Extraversion", "Agreeableness"]

fig, ax = plt.subplots(figsize=(8.5, 6))
y = 0
ylabels, ypos = [], []
for trait in traits:
    for grp, color in groups:
        row = t1[(t1.trait == trait) & (t1.group == grp)]
        if row.empty:
            continue
        r = row.mean_r.iat[0]
        lo, hi = row.CI95_lo.iat[0], row.CI95_hi.iat[0]
        bold = grp.startswith("High")
        ax.errorbar(
            r, y, xerr=[[r - lo], [hi - r]], fmt="o",
            color=color, capsize=3, markersize=7 if bold else 5,
            markeredgecolor="black" if bold else color, elinewidth=1.6 if bold else 1.1,
        )
        short = "military (mixed stakes)" if grp.startswith("Military") else grp.split(" (")[1][:-1]
        ylabels.append(f"{short}  (k={row.k.iat[0]})")
        ypos.append(y)
        y += 1
    y += 1.2

ax.axvline(0, color="gray", lw=0.8, ls="--")
ax.set_yticks(ypos)
ax.set_yticklabels(ylabels, fontsize=8)
ax.invert_yaxis()

# Trait section labels
y = 0
for trait in traits:
    n = sum(1 for grp, _ in groups if not t1[(t1.trait == trait) & (t1.group == grp)].empty)
    ax.text(-0.33, y + (n - 1) / 2, trait, fontsize=10, fontweight="bold", va="center", ha="right")
    y += n + 1.2

ax.set_xlabel("Meta-analytic correlation with g (95% CI)")
ax.set_xlim(-0.2, 0.3)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(left=False)
# No in-image title: APA 7 places the figure number and title above the image
# in the document (built by 06_build_docx.py from the markdown figure block).
fig.tight_layout()
fig.savefig(OUT / "figure1_stakes_moderation.png", dpi=300, bbox_inches="tight")
print("Saved", OUT / "figure1_stakes_moderation.png")
