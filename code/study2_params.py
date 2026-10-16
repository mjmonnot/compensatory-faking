"""Parameter module for the Study 2 trait-driven faking model.

Single source of numeric inputs for code/11_study2_trait_model.py. Every
parameter carries a comment naming its intended empirical source. Values that
still await verified anchors are marked:

    # PROVISIONAL — replace from literature/study2_anchor_table.md

(that anchor table is being compiled in parallel). Swapping any value here
requires no change to the model code — 11_study2_trait_model.py contains no
numeric constants of its own. Values NOT marked PROVISIONAL are the same
sourced anchors already used by code/09_selection_simulation.py or this
study's own meta-analytic estimates (output/region_contrasts.csv,
output/table2_contrasts.csv).
"""

# ---------------------------------------------------------------- latent trait structure
# Trait vector order everywhere: (g, C, H, SE).
#   g  = general mental ability
#   C  = true (unfaked) conscientiousness
#   H  = honesty-humility / integrity composite
#   SE = confidence / self-esteem dimension (compound trait in this project)

# Incumbent (honest-baseline) g-C correlations, this study
# (output/region_contrasts.csv for US/Non-US; Pooled = Table 1 incumbent mean,
# output/table1_meta_by_stakes.csv, r = .032 — corrected 2026-08-20; the prior
# value .010 was the total-database observed r, not an incumbent estimate).
RHO_GC_BASE = {"US": 0.013, "Pooled": 0.032, "Non-US": 0.054}

# Integrity x GMA = .00 (Ones 1993, via Schmidt & Hunter 1998); HEXACO H x g
# = .02 (Anglim et al. 2022, k = 23, N = 32,165). Verified in anchor table 2b/2a.
RHO_GH = 0.00

# Self-esteem x g, incumbent/employee samples, computed from Stanek-Ones
# database for this project (anchor table 2d: +.011, k = 16, N = 12,053).
RHO_GSE = 0.01

# Honesty-humility x Big Five: two meta-analytic sources are carried and the
# manuscript reports both (2026-10-12).
#   "howard"  (default): Howard & Van Zandt (2020, JRP, Table 1, Big Five row),
#             HEXACO honesty-humility, corrected rho: C .245 (k = 71,
#             N = 42,857), Neuroticism -.138 (k = 68, N = 42,095; reversed
#             onto ES), A .503 (k = 79, N = 44,267).
#   "overt":  Sackett & Wanek (1996, Table 3), overt integrity tests,
#             corrected: C .39, ES .28, A .34 (quoted by Lee, Ashton, &
#             de Vries, 2005, Human Performance, p. 182; observed .26/.18/.23).
# Select with the environment variable STUDY2_HH_SOURCE (howard | overt).
# Scripts append HH_SUFFIX to their output file names so the two runs never
# overwrite each other. The earlier .42/.33/.45 inputs (Blonigen et al.,
# 2011, previously mis-attributed to Ones, 1993) are archived in
# output/archive_hh42_run/.
import os as _os
HH_SOURCE = _os.environ.get("STUDY2_HH_SOURCE", "howard").lower()
_HH = {"howard": (0.25, 0.14, 0.50), "overt": (0.39, 0.28, 0.34)}
if HH_SOURCE not in _HH:
    raise ValueError(f"STUDY2_HH_SOURCE must be one of {sorted(_HH)}, got {HH_SOURCE!r}")
HH_SUFFIX = "" if HH_SOURCE == "howard" else f"_hh_{HH_SOURCE}"
RHO_CH = _HH[HH_SOURCE][0]

# Intended source: self-esteem/CSE x C meta.
RHO_CSE = 0.30   # PROVISIONAL — replace from literature/study2_anchor_table.md

# Intended source: H x self-esteem correlation (HEXACO nomological net).
RHO_HSE = 0.20   # PROVISIONAL — replace from literature/study2_anchor_table.md

# ---------------------------------------------------------------- performance model
# y = b_g*g + b_C*C [+ b_H*H] + e; coefficients solved so operational
# validities (zero-order r with y) reproduce the targets below.
V_G = 0.31       # Sackett, Zhang, Berry, & Lievens (2022, JAP, Table 4), cognitive ability
V_C = 0.19       # Sackett et al. (2022, Table 4), conscientiousness overall
# Integrity-test operational validity: Van Iddekinge et al. (2012) corrected
# .15-.18 (job performance); Ones et al. (1993) .41 is the optimistic bound.
# Midpoint of the defensible modern range; sensitivity should span {.15, .41}.
V_H = 0.18       # anchor table 2b (range, not point) — sensitivity dimension

# ---------------------------------------------------------------- subgroup differences
# Black-White standardized differences (positive = White higher), for AIR.
D_G = 0.79       # Sackett et al. (2022); Roth et al. (2011) applicant data
D_C = -0.07      # Dahlke & Sackett (2017)
# Intended source: integrity-test subgroup d (Ones & Viswesvaran 1998, near zero).
D_H = 0.04       # PROVISIONAL — replace from literature/study2_anchor_table.md
# Intended source: self-esteem Black-White d (near zero / slightly negative).
D_SE = 0.00      # PROVISIONAL — replace from literature/study2_anchor_table.md
P_MIN = 1 / 6    # minority applicant-pool proportion (common convention, as in 09)

# ---------------------------------------------------------------- faking propensity
# P(fake_i) = logistic(gamma_0 + gamma_g*g_i + gamma_C*C_i + gamma_H*H_i).
# The gammas are NOT set directly: they are solved inside the model so the
# simulated pool reproduces the target point-biserial correlations between
# each trait and the faking indicator (plus prevalence, via gamma_0). This
# anchors WHO FAKES to observed trait-faking correlations rather than to
# assumed logit slopes. Targets come from the anchor table (Section 1).
PARAM_SETS = {
    # Field-anchored: Holtrop et al. (2021, IJSA), n = 128 genuine firefighter
    # applicants, within-person RADS faking scores. g: r = -.19 (p = .03);
    # C (low-stakes): r = +.17 (ns); H: r = -.04 (ns). Point estimates used;
    # the ns-to-zero variant is a sensitivity set below.
    "field": {"r_gF": -0.19, "r_CF": 0.17, "r_HF": -0.04},
    # Lab-anchored: Dunlop et al. (2022, JBP), N = 189 incentivized simulation.
    # g: partial r = +.14 (positive in all regressions); H: r = +.21.
    # C: no verified lab personality-test anchor (McFarland & Ryan magnitudes
    # need full-text check); -.15 is the midpoint of the anchor table's
    # recommended lab range (-.30 to 0, interview-derived). PROVISIONAL for C.
    "lab": {"r_gF": 0.14, "r_CF": -0.15, "r_HF": 0.21},
    # Sensitivity variants:
    "field_ns0": {"r_gF": -0.19, "r_CF": 0.0, "r_HF": 0.0},   # ns anchors -> 0
    "lab_negH": {"r_gF": 0.14, "r_CF": -0.15, "r_HF": -0.20},  # classical H sign
}
DEFAULT_PARAM_SET = "field"

# Expectancy weights inside the need term (inflation magnitude only).
# Intended source: ability-to-identify-criteria / expectancy literature.
W_G_NEED = 0.5   # PROVISIONAL — replace from literature/study2_anchor_table.md
W_C_NEED = 0.5   # PROVISIONAL — replace from literature/study2_anchor_table.md

# Midpoint of the 30-50% within-person applicant range (Griffith, Chmielowski,
# & Yoshita 2007: 30-50% depending on CI; Donovan, Dwight, & Schneider 2014:
# 49.7% on >=1 dimension). The Griffith & Converse (2011) chapter synthesis
# concludes 30% +/-10% (verified from uncorrected proof, 2026-08-25), so .40
# is also that synthesis's upper bound — do NOT cite the chapter for 30-50%.
P_FAKE = 0.40
# Sensitivity: within-person range endpoints (.30, .50) plus the Griffith &
# Converse (2011) synthesis lower bound (.20, added 2026-08-25). .20 is listed
# last so scripts with sequential rng keep earlier conditions bit-identical.
P_FAKE_SENS = (0.30, 0.50, 0.20)

# ---------------------------------------------------------------- inflation magnitude
# infl_i = m0 + m_need*need_i + m_skill*g_i + N(0, sigma) for fakers.
# (m0, m_need, m_skill) are CALIBRATED by minimum distance, not set here.
# SE-scale inflation for fakers: s0 + m_skill*g_i + N(0, sigma); the skill
# coefficient is shared across scales (one faking-as-skill mechanism).
#
# SE-scale mean-inflation intercept. No anchor exists (Birkeland et al. 2006
# does not report SE); held at 0 until the anchor table supplies one.
S0_SE = 0.0      # PROVISIONAL — replace from literature/study2_anchor_table.md

SIGMA_INFL = 0.50            # idiosyncratic faking noise, default (as in 09)
SIGMA_SENS = (0.25, 1.00)    # sensitivity values (as in 09)

# ---------------------------------------------------------------- calibration targets
D_SHIFT = 0.45               # Birkeland et al. (2006): applicant-incumbent C shift
# Applicant (high-stakes) g-C correlations, this study (region_contrasts.csv).
TARGET_R_APPLICANT = {"US": 0.007, "Pooled": -0.074, "Non-US": -0.272}
# Compound-trait (Self-Esteem/Trust) applicant-incumbent correlation shift,
# this study (table2 contrasts): +.17.
DELTA_R_SE_G = 0.17
# Applicant/honest observed-C SD ratio (U ratio). Hu & Connelly (2021, IJSA,
# Table 4, full text received from author 2026-08-13): Conscientiousness
# U = 0.86, k = 18, N = 3,346, SD_U = 0.17, 80% CrI [0.64, 1.08]. Within-person
# applicant-vs-honest design; closest published anchor for the model's
# applicant-pool/incumbent-pool ratio. Uncorrected for range restriction
# (none possible per the source).
SD_RATIO_TARGET = 0.86       # Hu & Connelly (2021, Table 4) — ANCHORED
SD_RATIO_SENS = (0.64, 1.00)  # 80% CrI lower bound; legacy no-reduction value

# Weighted-least-squares moment weights (inverse squared SEs of the target
# moments where the source reports them).
MOMENT_WEIGHTS = {
    # Birkeland Table 3 C shift: d = .45, 95% CI [.303, .591] -> SE = .0735
    # (full-text check 2026-08-20; replaces the provisional .05).
    "d_shift": 1 / 0.0735**2,
    "r_gC": 1 / 0.03**2,        # approx. SE of the Study 1 applicant C mean
    "delta_r_SE": 1 / 0.05**2,  # PROVISIONAL — replace from literature/study2_anchor_table.md
    # Weighted by the between-study heterogeneity SD (SD_U = 0.17, Hu &
    # Connelly 2021, Table 4), not the SE of the mean (0.04): the calibration
    # matches ONE synthetic applicant pool, and single pools plausibly span
    # the 80% CrI [0.64, 1.08]. SE-based weighting (1/0.04^2) is the
    # sensitivity variant; it treats the mean U as a point constraint.
    "sd_ratio": 1 / 0.17**2,
}
SD_RATIO_WEIGHT_SENS = 1 / 0.04**2  # SE-of-mean weighting, sensitivity only

# ---------------------------------------------------------------- selection design
W_COMPOSITE = 0.5            # equal weights on standardized g and C (as in 09)
SRS = (0.10, 0.30)           # selection ratios (as in 09)
N_POOL = 200_000             # applicant pool size (as in 09)
SEED = 2026                  # rng seed (as in 09)
INCLUDE_SE = True            # toggle the SE dimension and its moment


# ---- 12_study2_expanded_model.py additions ----
# Expanded latent vector (g, C, ES, A, H, SE): Emotional Stability and
# Agreeableness added so the Birkeland et al. (2006) shift pattern
# (ES d = .44 shifts, A ~ null) becomes a genuine over-identifying test
# instead of being satisfied structurally (model notes section 4).

# -- new trait correlations --
# ES x g: Schilling et al. (2021, IJSA, Table 2) nonselection baseline
# rho = .039 (anchor table 2a). Sensitivity: Anglim et al. (2022, Psych
# Bulletin) neuroticism x intelligence rho = -.09, i.e. ES x g = +.09.
RHO_GES = 0.04
RHO_GES_SENS = 0.09          # Anglim et al. (2022) alternative anchor

# A x g: Schilling et al. (2021, Table 2) nonselection rho = -.019
# (anchor table 2a; Anglim et al. 2022 report A ~ 0 as well).
RHO_GA = -0.02

# Big Five intercorrelations: van der Linden, te Nijenhuis, & Bakker (2010,
# J. of Research in Personality, Table 2; K = 212, N = 144,117), CORRECTED
# true-score rho: C-N = -.43 (=> C-ES +.43), C-A = +.43, A-N = -.36
# (=> ES-A +.36). Verified from fetched full text 2026-08-12. Note these are
# construct-level (corrected) values; the uncorrected observed rs are
# C-N -.32, C-A .31, A-N -.26 — using corrected values is consistent with
# treating the latent vector as true scores, but the choice observed-vs-
# corrected is a framing decision: flag PROVISIONAL at that level.
RHO_CES = 0.43   # PROVISIONAL (corrected-vs-observed choice) — van der Linden et al. 2010
RHO_CA = 0.43    # PROVISIONAL (corrected-vs-observed choice) — van der Linden et al. 2010
RHO_ESA = 0.36   # PROVISIONAL (corrected-vs-observed choice) — van der Linden et al. 2010

# ES x H and A x H: see the HH_SOURCE block above (Howard & Van Zandt, 2020,
# or Sackett & Wanek, 1996, overt tests).
RHO_ESH = _HH[HH_SOURCE][1]
RHO_AH = _HH[HH_SOURCE][2]

# ES x SE: core-self-evaluations literature (SE and ES are both CSE
# indicators, typically r ~ .5 between self-esteem and neuroticism-reversed).
RHO_ESSE = 0.50  # PROVISIONAL — replace from literature/study2_anchor_table.md

# A x SE: self-esteem x agreeableness, small positive in Big Five metas.
RHO_ASE = 0.20   # PROVISIONAL — replace from literature/study2_anchor_table.md

# -- subgroup differences for the new traits (Black-White d, White higher) --
# Foldes, Duehr, & Ones (2008) report near-zero ES and A differences; exact
# values not re-verified for this project. ES/A are not scored and do not
# enter the performance model, so these affect nothing downstream.
D_ES = 0.00      # PROVISIONAL — replace from literature/study2_anchor_table.md
D_A = 0.00       # PROVISIONAL — replace from literature/study2_anchor_table.md

# -- new calibration targets (Birkeland et al. 2006, applicant vs incumbent) --
D_SHIFT_ES = 0.44            # emotional stability shift (verified abstract text)
# Agreeableness: Birkeland Table 3 point estimate d = .16, 95% CI [-.01, .32],
# ns (full-text check 2026-08-20). Corrected from 0.00 (abstract-only reading)
# after three independent reviews flagged the mismatch.
D_SHIFT_A = 0.16
# Moment weights for the new shift targets: inverse squared SEs recovered
# from the Birkeland Table 3 95% CIs (full-text check 2026-08-20).
MOMENT_WEIGHTS_12 = {
    "d_shift_ES": 1 / 0.0804**2,  # ES: d = .44, CI [.278, .593] -> SE = .0804
    "d_shift_A": 1 / 0.0852**2,   # A:  d = .16, CI [-.010, .324] -> SE = .0852
}

# -- variant D: social-desirability loading vector lambda --
# Anchor: Ones, Viswesvaran, & Reiss (1996, JAP) meta-analytic correlations
# of social-desirability scales with the Big Five. VERIFIED (abstract +
# multiple secondary full texts): SD x Emotional Stability = .37,
# SD x Conscientiousness = .20. Their Table 6 values for A/E/O are NOT
# retrievable via open web (2026-08-12); the SDR literature consistently
# places A's loading in the same band as C's (moralistic-bias content), so
# lambda_A = .20 is used and flagged. SE has no published SD rating; set
# equal to ES (self-esteem items are maximally evaluative, and SE tracks the
# CSE/ES cluster) and flagged. The model normalizes by lambda_C, so only the
# RATIOS matter: lambda_ES/lambda_C = 1.85 and lambda_A/lambda_C = 1.00 are
# the empirically testable content of variant D.
LAMBDA_SD = {
    "C": 0.20,   # Ones et al. (1996) — verified
    "ES": 0.37,  # Ones et al. (1996) — verified
    "A": 0.20,   # PROVISIONAL — Ones et al. Table 6 not retrievable; SDR-literature band
    "SE": 0.37,  # PROVISIONAL — no published SD rating; set equal to ES
}
LAMBDA_A_SENS = (0.10, 0.30)  # sweep the crux (provisional) loading

# -- design --
PARAM_SETS_12 = ("field_ns0", "field", "lab")  # field_ns0 = decided primary
PRIMARY_PARAM_SET_12 = "field_ns0"
