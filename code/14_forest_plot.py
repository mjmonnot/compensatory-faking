"""Forest plots of per-sample personality-GMA correlations by assessment context.

Builds the main-text Conscientiousness forest plot (Figure 2) and the
supplemental forest plots for the remaining Big Five traits (Figures S2-S5).
Layout chosen by MJM 2026-08-19: applicant vs. incumbent sections, no
weight column, no regional split.

Each row is one independent sample: observed r, 95% CI via Fisher r-to-z,
square marker area proportional to N. Group summaries are the same
N-weighted bare-bones means as Table 1 (meta_lib.bare_bones).
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from meta_lib import bare_bones

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

APP_COLOR, INC_COLOR = "#c0392b", "#2471a3"

COUNTRY_SHORT = {
    "United States of America": "US",
    "Australia / New Zealand": "ANZ",
    "Canada": "CAN",
    "United Kingdom": "UK",
    "combination (Estonia & Russia)": "EST/RUS",
    "<Not Reported>": "NR",
}

df = pd.read_csv(ROOT / "data" / "analysis_effects.csv")
z95 = 1.96
zr = np.arctanh(df.r)
se = 1 / np.sqrt(df.N - 3)
df["lo"] = np.tanh(zr - z95 * se)
df["hi"] = np.tanh(zr + z95 * se)


def short_label(row, with_country=False):
    """'Surname et al. (year)' from the database's author string."""
    first = re.split(r"[,&]| and ", str(row.first_author))[0].strip()
    multi = bool(re.search(r"[,&]| and ", str(row.first_author)))
    lab = f"{first}{' et al.' if multi else ''} ({int(row.year)})"
    if with_country:
        lab += f", {COUNTRY_SHORT.get(row.country, row.country)}"
    return lab


def plot_forest(sections, outfile, show_weight=False):
    """sections: list of (title, color, dataframe, country_flag) top to bottom."""
    n_rows = sum(len(s[2]) for s in sections)
    fig_h = 1.2 + 0.099 * (n_rows + 3.6 * len(sections))
    fig, ax = plt.subplots(figsize=(7.3, fig_h))
    trans = ax.get_yaxis_transform()  # x in axes fraction, y in data units

    col_n, col_w, col_r = (1.01, 1.095, 1.17) if show_weight else (1.01, None, 1.075)
    y = 0
    ylabels, ypos, seen = [], [], {}
    for title, color, sub, country_labels in sections:
        sub = sub.sort_values("r")
        n_grp = sub.N.sum()
        ax.text(-0.26, y, f"{title} (k = {len(sub)})", fontsize=7.5,
                fontweight="bold", va="center", ha="left", transform=trans)
        y += 1.5
        sizes = np.clip(np.sqrt(sub.N) / 3.5, 2.0, 6.2)
        for (_, row), ms in zip(sub.iterrows(), sizes):
            ax.plot([row.lo, row.hi], [y, y], color=color, lw=0.7,
                    solid_capstyle="butt", zorder=2)
            ax.plot(row.r, y, "s", color=color, markersize=ms, zorder=3)
            lab = short_label(row, with_country=country_labels)
            if lab in seen:  # same first author and year: disambiguate a, b, ...
                seen[lab] += 1
                suffix = "abcdefghijklmnopqrstuvwxyz"[seen[lab] % 26]
                lab = f"{lab[:-1]}{suffix})" if not country_labels \
                    else f"{lab} ({suffix})"
            else:
                seen[lab] = 0
            ylabels.append(lab)
            ypos.append(y)
            ax.text(col_n, y, f"{int(row.N):,}", fontsize=5.4, va="center",
                    ha="right", family="monospace", transform=trans)
            if show_weight:
                ax.text(col_w, y, f"{100 * row.N / n_grp:4.1f}", fontsize=5.4,
                        va="center", ha="right", family="monospace", transform=trans)
            ax.text(col_r, y, f"{row.r:+.2f} [{row.lo:+.2f}, {row.hi:+.2f}]",
                    fontsize=5.4, va="center", ha="left", family="monospace",
                    transform=trans)
            y += 1
        bb = bare_bones(sub.r.to_numpy(), sub.N.to_numpy())
        ax.fill([bb.ci_lo, bb.mean_r, bb.ci_hi, bb.mean_r],
                [y, y - 0.45, y, y + 0.45], color=color, zorder=3)
        ylabels.append("N-weighted mean")
        ypos.append(y)
        ax.text(col_r, y, f"{bb.mean_r:+.2f} [{bb.ci_lo:+.2f}, {bb.ci_hi:+.2f}]",
                fontsize=5.9, va="center", ha="left", family="monospace",
                fontweight="bold", transform=trans)
        y += 3.1

    ax.axvline(0, color="gray", lw=0.8, ls="--", zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=5.4)
    ax.set_ylim(-2.6, y - 1.6)
    ax.invert_yaxis()
    lim = max(0.7, max(max(abs(s[2].lo.min()), abs(s[2].hi.max()))
                       for s in sections) + 0.04)
    ax.set_xlim(-lim, lim)
    tick_max = np.floor(lim / 0.2) * 0.2
    ax.set_xticks(np.arange(-tick_max, tick_max + 0.01, 0.2))
    ax.set_xlabel("Observed correlation with GMA (95% CI)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    hdr_y = -1.7
    ax.text(col_n, hdr_y, "N", fontsize=6.3, fontweight="bold", ha="right",
            transform=trans)
    if show_weight:
        ax.text(col_w, hdr_y, "Wt %", fontsize=6.3, fontweight="bold",
                ha="right", transform=trans)
    ax.text(col_r, hdr_y, "r [95% CI]", fontsize=6.3, fontweight="bold",
            ha="left", transform=trans)

    fig.tight_layout()
    fig.savefig(OUT / outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved", OUT / outfile)


def by_trait(trait):
    c = df[df.trait == trait]
    app = c[c.stakes == "High stakes (applicants)"]
    inc = c[c.stakes == "Low stakes (incumbents)"]
    return app, inc


# --- Main-text Figure 2 (Conscientiousness) ---------------------------------
app, inc = by_trait("Conscientiousness")
plot_forest([("Applicant samples", APP_COLOR, app, False),
             ("Incumbent samples", INC_COLOR, inc, False)],
            "figure_forest_conscientiousness.png")

# --- Supplemental figures S2-S5 (remaining Big Five, baseline layout) -------
for trait in ["Neuroticism", "Openness", "Extraversion", "Agreeableness"]:
    a, i = by_trait(trait)
    plot_forest([("Applicant samples", APP_COLOR, a, False),
                 ("Incumbent samples", INC_COLOR, i, False)],
                f"figure_forest_{trait.lower()}.png")

# --- Supplemental figure S6: Conscientiousness applicants split by region ---
NONUS_COLOR = "#7d3c98"
capp, _ = by_trait("Conscientiousness")
plot_forest(
    [("US applicant samples", APP_COLOR,
      capp[capp.country == "United States of America"], True),
     ("Non-US applicant samples", NONUS_COLOR,
      capp[capp.country != "United States of America"], True)],
    "figure_forest_conscientiousness_us_nonus.png")
