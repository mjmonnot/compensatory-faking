# Study 2 trait-driven faking model — technical notes

**ADDENDUM (2026-08-13): SD-ratio target ANCHORED.** Hu & Connelly (2021) full
text received from B. Connelly (`literature/Hu & Connelly (2021). IJSA.pdf`).
`SD_RATIO_TARGET` = 0.86 (Conscientiousness U ratio, Table 4, k = 18,
N = 3,346), weighted by between-study heterogeneity (SD_U = 0.17), not the SE
of the mean (0.04) — the calibration matches one pool and pools vary (80% CrI
[0.64, 1.08]); the SE-based weight was tried first and let the single
variance target dominate all other targets. All of 11/12/13 re-run
(logs `output/log_1*_rerun_sdratio086*.txt`). Consequences: (a) adjudication
conclusions unchanged (D-variant still rejected by an order of magnitude;
+.17 r_CF still rejected in US, profile cap +.11 unchanged; pooled/non-US now
also bounded, cap ≈ +.20); (b) the anchored variance target now *favors* the
censored (ceiling) inflation form in US/Pooled (dist 2.9→0.9, 1.7→0.8) and
the linear form in Non-US (1.45→3.58) — form choice is now empirically
guided, so US/Pooled losses are reported as 9–19% with the upper half favored
(censored and freed-skill forms agree at ≈15–19%); (c) feasibility frontier
moved from ≈−.25 to ≈−.26. Residual caveat: U is within-person
applicant-vs-honest and uncorrected; the model's quantity is between-pool.
Manuscript v6 updated accordingly (Method targets, Tables 5–6, Results,
Limitations).

Status: **anchored v2 + expanded v3** (2026-08-12; v3 = six-trait expansion,
section 7, `code/12_study2_expanded_model.py`). The faking-propensity parameters and the
key trait correlations are now pinned to the verified anchors in
`literature/study2_anchor_table.md`; parameters still flagged PROVISIONAL in
`code/study2_params.py` are the remaining sensitivity dimensions (rho_CSE,
rho_HSE, d_H, d_SE, need weights, SD-ratio target, moment weights, s0).
Code: `code/11_study2_trait_model.py`; results:
`output/study2_prototype_results.csv`. Companion/baseline model:
`code/09_selection_simulation.py` (random-faker mechanism).

**Framing decisions (MJM, 2026-08-12).** Study 2 is a full research article.
(1) Primary framing = the adjudication itself (all four anchor sets presented,
applicant-pool moments discriminate), with `field_ns0` as the primary
specification for outcome tables. (2) The N/E/A/O structural-null limitation
(section 4) is resolved by expanding the latent vector — middle version:
add Emotional Stability and Agreeableness so the Birkeland shift pattern
(ES d = .44 shifts, A ~ null) becomes a genuine over-identifying test;
full Big Five expansion only if the six-trait model calibrates cleanly.

Change from the v1 prototype: the propensity is no longer parameterized by
assumed logit slopes. It is logistic in (g, C, H) directly, and all four
gammas are **solved** so the simulated pool reproduces anchored point-biserial
trait–faking correlations plus prevalence (smooth moments via
E[F|traits]; Cov(x,F) = Cov(x,p), Var(F) = p̄(1−p̄)). Anchor sets:

| set | r(g,F) | r(C,F) | r(H,F) | source |
|---|---|---|---|---|
| field | −.19 | +.17 | −.04 | Holtrop et al. (2021), n = 128 genuine applicants |
| lab | +.14 | −.15 (provisional) | +.21 | Dunlop et al. (2022), N = 189 incentivized |
| field_ns0 | −.19 | 0 | 0 | field with ns anchors zeroed |
| lab_negH | +.14 | −.15 | −.20 | lab with classical negative H sign |

Verified trait-correlation anchors now in place: rho_gH = .00 (Ones 1993 via
Schmidt & Hunter 1998; Anglim et al. 2022 give .02), rho_CH = .42 (Ones),
rho_gSE = .01 (this study's incumbent estimate), V_H = .18 (Van Iddekinge et
al. 2012 corrected .15–.18; Ones .41 as the optimistic sensitivity bound).

## 1. Model

Latent traits per applicant, order (g, C, H, SE), multivariate normal with unit
variances and correlation matrix R(context); rho_gC is the incumbent baseline
from this study (US .013, pooled .010, non-US .054); rho_gH (.00), rho_CH
(.42), and rho_gSE (.01) are verified anchors; rho_CSE (.30) and rho_HSE
(.20) remain provisional. Minority applicants (proportion 1/6)
have all four means shifted down by the subgroup ds (d_g = .79, d_C = −.07
sourced; d_H, d_SE provisional).

True performance (H path toggled on/off, common residual shock):

    y = b_g·g + b_C·C [+ b_H·H] + e,   e ~ N(0, 1 − v'b)

with b = R_pred⁻¹·v solved exactly so zero-order operational validities hit
v = (.31, .19[, .20]) (Sackett et al. 2022; integrity validity provisional).
Verified in-simulation: achieved r(y, ·) = (.32, .19, .20).

Faking propensity (WHO fakes — the departure from 09's random 40%):

    P(fake_i) = logistic(γ₀ + γ_g·g_i + γ_C·C_i + γ_H·H_i)

All four gammas are solved numerically so the pool reproduces the anchored
point-biserial correlations (r_gF, r_CF, r_HF) and the prevalence target
(default .40; sensitivity .30/.50) — the propensity is exactly anchored, not
part of the WLS. Solved slopes at defaults, US: field γ = (−.41, +.50, −.30);
lab γ = (+.32, −.68, +.78). The need term −(w_g·g + w_C·C) survives only
inside the inflation-magnitude equation.

Inflation magnitude (fakers only):

    infl_i  = m₀ + m_need·need_i + m_skill·g_i + N(0, σ)     → C_obs = C + F·infl
    inflSE_i = s₀ + m_skill·g_i + N(0, σ)                    → SE_obs = SE + F·inflSE

m_need is the compensation mechanism, m_skill the faking-as-skill mechanism;
the skill coefficient is **shared** across the C and SE scales (one skill
mechanism), which is what lets the SE moment identify it. s₀ has no empirical
anchor (Birkeland et al. 2006 does not report SE) and is fixed at 0.

Selection mirrors 09: composite = .5·g + .5·standardized C_obs, top-down at
SR .10/.30, n = 200,000, seeded rng; SE_obs is *not* scored (it exists only to
carry the compound-trait identification moment); H is *not* scored (no
integrity test administered) but optionally feeds performance.

## 2. Calibration

Minimum-distance / weighted least squares (`scipy.optimize.least_squares` on
√w-scaled moment errors, common random numbers held fixed across optimizer
steps) over four applicant-pool moments:

| # | moment | target | source |
|---|--------|--------|--------|
| 1 | C mean shift d (obs vs honest, pooled SD) | .45 | Birkeland et al. 2006 |
| 2 | applicant r(g, C_obs) | US +.007 / pooled −.074 / non-US −.272, fit separately | this study |
| 3 | applicant Δr(g, SE_obs) | +.17 | this study, compound-trait contrast |
| 4 | applicant/incumbent SD ratio of observed C | 1.00 | no anchor retrievable (Hu & Connelly 2021 paywalled, direction-only: slightly < 1); swept at {.85, .95, 1.00} |

Weights are provisional inverse squared SEs (d: .05, r: .03, Δr_SE: .05,
SD ratio: .10). Prevalence is a fifth moment matched exactly by γ₀ inside the
loop.

## 3. Identification status

| parameter | how determined | status |
|---|---|---|
| rho_gC (baseline) | this study's incumbent estimates | pinned |
| rho_gH, rho_CH, rho_gSE | verified external anchors (anchor table 2a/2b/2d) | pinned |
| rho_CSE, rho_HSE | no verified anchor | provisional |
| b_g, b_C, b_H | solved exactly from operational validities | pinned given v (V_H = .18, sensitivity {.15, .41}) |
| γ₀, γ_g, γ_C, γ_H | solved exactly from anchored trait–faking correlations + prevalence | pinned given the anchor set (field/lab) |
| w_g, w_C (need weights, inflation only) | external | provisional |
| m₀ | mainly moment 1 (mean shift) | calibrated |
| m_need | mainly moment 2 (r_gC) | calibrated |
| m_skill | mainly moment 3 (SE shift), via the shared-coefficient restriction | calibrated |
| s₀ (SE intercept) | none | fixed at 0, unidentified |
| σ | sensitivity parameter {.25, .50, 1.00} | design choice (as in 09) |

The system is **conditionally over-identified**: 4 moments constrain 3 free
inflation parameters, given exactly-anchored propensity parameters. Because
the propensity is now pinned by data rather than placeholders, the WLS
distance becomes a genuine adjudication statistic across anchor sets:

| context | field | lab | field_ns0 | lab_negH |
|---|---|---|---|---|
| US | 5.11 | **0.23** | 1.97 | 0.23 |
| Pooled | 2.93 | **0.23** | — | — |
| Non-US | 2.79 | **2.42** | 0.42 | 2.39 |

Reading: the moments clearly reject the field set's **+.17 C–faking anchor**
(Holtrop's ns point estimate, n = 128) — "higher-C people fake more" inflates
the observed-C SD (achieved ratio 1.14–1.18 vs target 1.0) and depresses the
mean shift (d = .40 vs .45). Zeroing that one anchor (field_ns0) restores fit
(US 1.97, non-US 0.42), and the lab set's −.15 fits equally well. The g and H
propensity slopes are barely discriminated (field_ns0 ≈ lab ≈ lab_negH in
distance) — so the moments adjudicate the C slope but not the g/H slopes,
which must stay externally pinned.

Calibrated inflation values at defaults (field, prevalence .40, σ = .50):
US (m₀, m_need, m_skill) = (1.23, 0.60, 0.51); non-US = (1.14, 1.91, 0.43).
The prototype's apportioning result survives anchoring: m_skill is stable
across contexts (≈ .43–.51) and the US/non-US difference loads on the
compensation mechanism (m_need 0.60 vs 1.91). Negative-inflation
(self-sabotage) share of fakers, now tracked as a diagnostic: 4–8% (US)
but 18–26% (non-US), confirming the −.27 target strains the linear
inflation form.

## 4. N/E/A/O null constraints (limitation)

The meta-analytic result that N, E, A, O show no applicant–incumbent shift
cannot enter the moment vector because those traits are not in the latent
vector. The model satisfies the nulls **structurally**: only the C and SE
scales are faked, so nothing else shifts by construction. This is an
assumption, not a fitted implication — the model as specified cannot fail
this test, so the nulls currently carry no identifying information. A full
Study 2 would need either (a) N/E/A/O in the latent vector with desirability
loadings and the nulls as genuine over-identifying moments, or (b) an
argument for why fakers target C- and SE-type content specifically (e.g.,
job-desirability perceptions, ATIC).

**Resolved (v3, middle version)** — see section 7: ES and A are now in the
latent vector, the Birkeland ES/A shifts are calibration moments, and a
desirability-proportional (trait-general) inflation variant is tested
against a targeted variant. The A null and the ES/C shift ratio reject the
trait-general variant in every context.

## 5. Headline results (anchored v2)

> **Caveat added after the section 8 validation batch (2026-08-12):** the
> US/Pooled performance-loss and hires-changed point estimates below are
> **form-dependent**. Two independent relaxations of the linear inflation
> form (censoring at zero, 8.1; freeing the shared m_skill, 8.5) fit the
> same US/Pooled moments as well or better and roughly **double** the damage
> (US SR .10: ~19% loss / ~40–42% changed vs 10% / 35.5% here). Present
> US/Pooled outcomes as ranges across functional forms, or anchor the
> SD-ratio target first. Non-US outcomes are robust (8.0–9.2% across forms),
> and all qualitative claims (integrity composition, AIR stability) hold in
> every form.

The anchored results are **more sober than the v1 prototype's** — anchoring
the propensity to observed trait–faking correlations (which are modest,
|r| ≤ .21) shrinks the dramatic placeholder effects. Do not cite the v1
numbers (1-SD integrity swap, +5.4 to +7.3 extra loss); they were artifacts
of an assumed γ_H = 1.0 slope far steeper than any observed correlation.

1. **Aggregate performance loss: trait-driven ≈ random under US/pooled
   calibrations.** Extra true-performance loss over the random benchmark
   (H-path on) is +0.0 to +2.2 points (US), +1.3 to +1.5 (pooled). WHO fakes
   matters much less for aggregate hire quality than placeholder slopes
   implied, because realistic trait–faking correlations are small.
2. **Integrity composition of changed hires still differs meaningfully.**
   Under the field set, faked-in hires average H ≈ −.01 to −.21 vs +.23 to
   +.46 under random faking on the same pool — a quarter- to half-SD
   integrity gap, concentrated exactly in the hires the faking created.
   Displaced applicants average H ≈ +.33 to +.58.
3. **The field-vs-lab contrast matters most in the non-US calibration.**
   With lab-anchored propensity (fakers are low-C), hitting the −.27 target
   requires enormous compensatory inflation among low-C fakers, producing
   37–42% changed hires and 38–48% performance loss — implausible, and a
   useful argument that lab-anchored faking parameters cannot describe
   genuine applicant pools (echoing Schilling et al.'s field/lab moderation).
4. **AIR is essentially unchanged by trait-driven faking** (US SR .10: .39
   honest vs .38 faked), because inflation is spent on low-H/high-need
   applicants of any group rather than boosting low-g applicants per se.

## 6. What remains to validate

1. ~~Swap verified anchors and rerun~~ — **done (v2)**: propensity solved
   from anchored trait–faking correlations; rho_gH/rho_CH/rho_gSE/V_H pinned.
   Still open from the anchor table: rho_CSE, rho_HSE, d_H, d_SE, s₀, and
   moment weights from actual meta-analytic SEs; ~~V_H sensitivity {.15, .41}
   not yet run~~ — **done (validation batch), see §8.2: V_H is immaterial**.
   The SD-ratio target is now a **swept** sensitivity dimension
   ({.85, .95, 1.00}): Hu & Connelly (2021) and McFarland & Ryan (2000) full
   texts are NOT retrievable via the open web (verified 2026-08-12; library
   or author-email access required — see anchor table for the search trail).
   Sweep result: selection outcomes are robust to the target (US SR .10 perf
   loss 7.8–9.1% across {.85, .95, 1.00}; H composition of changed hires
   essentially unchanged). The field set's achieved ratio stays ≈ 1.13–1.18
   regardless of target (the +.17 C anchor forces variance inflation), so a
   true ratio below 1 would further penalize the field set in the
   adjudication without altering outcome conclusions.
2. ~~The negative-inflation artifact: refit with inflation censored at 0~~ —
   **done (validation batch), see §8.1**: the non-US −.27 target is NOT fully
   reachable without negative inflation (tops out at −.253, distance 0.42 →
   2.70); and censoring exposes a broader functional-form sensitivity of the
   outcome magnitudes (§8.1, §8.5).
3. ~~Profile the WLS distance over the propensity grid~~ — **done
   (validation batch), see §8.4**: finer r_CF grid run; only the US context
   adjudicates r_CF (heuristic 95% region roughly r_CF ≲ +.11, rejecting the
   field +.17 anchor); Pooled/Non-US accept the whole [−.30, +.20] grid.
4. ~~Single seed at n = 200,000~~ — **done (validation batch), see §8.3**:
   R = 25 ADEMP-style replications; Monte Carlo SEs are negligible relative
   to every reported contrast.
5. ~~The shared-m_skill restriction across C and SE scales is untested~~ —
   **done (validation batch), see §8.5**: the restriction is rejected in
   US/Pooled (freed slopes 1.27/1.07 vs 0.42) and outcome magnitudes are
   sensitive to it — the most consequential finding of the batch.
6. Add a scored-integrity variant (H in the selection composite) to connect
   to the integrity-testing literature, and consider modeling N/E/A/O per §4
   (partially addressed by the v3 expansion, section 7; E and O still open).

## 7. Expanded model (v3): ES + A, targeted vs desirability-proportional inflation

Code: `code/12_study2_expanded_model.py`; parameters: the
"12_study2_expanded_model.py additions" section of `code/study2_params.py`;
results: `output/study2_expanded_results.csv`. Run 2026-08-12: n = 200,000,
seed 2026, common random numbers across variants (same pool, same faking
indicator F, same inflation-noise draws per scale).

### 7.1 Latent vector and anchors

Latent traits per applicant, order **(g, C, ES, A, H, SE)**, multivariate
normal, unit variances. New correlation anchors (full sourcing in
`study2_params.py`):

| pair | value | source | status |
|---|---|---|---|
| ES–g | +.04 (sens. +.09) | Schilling et al. 2021 nonselection ρ = .039 (anchor table 2a); Anglim et al. 2022 N–g ρ = −.09 as sensitivity | pinned, two anchors |
| A–g | −.02 | Schilling et al. 2021 nonselection ρ = −.019 | pinned |
| C–ES | +.43 | van der Linden et al. 2010, Table 2 corrected ρ (C–N = −.43), K = 212, N = 144,117; verified from full text 2026-08-12 | PROVISIONAL at the corrected-vs-observed level (uncorrected r = .32) |
| C–A | +.43 | van der Linden et al. 2010 (C–A corrected ρ = .43; uncorrected .31) | PROVISIONAL (same caveat) |
| ES–A | +.36 | van der Linden et al. 2010 (A–N corrected ρ = −.36; uncorrected −.26) | PROVISIONAL (same caveat) |
| ES–H | +.33 | Ones integrity × Emotional Stability (anchor table 2b) | pinned |
| A–H | +.45 | Ashton & Lee 2005 (multiple R = .58); Ludeke et al. 2019 (r = .47 with BFAS A); Ones integrity–A = .40 | PROVISIONAL — no composite-level point estimate |
| ES–SE | +.50 | CSE literature (self-esteem and ES are both CSE indicators) | PROVISIONAL |
| A–SE | +.20 | Big Five–self-esteem metas, small positive | PROVISIONAL |

**Positive definiteness:** the extended 6×6 matrix is comfortably PD in all
three contexts (min eigenvalue ≈ 0.461; 0.454 under ES–g = .09). **No
nearest-PD adjustment was applied** (the eigenvalue-clip fallback exists in
the code and reports the max element change if ever triggered).

Subgroup ds for the new traits are set to 0 (PROVISIONAL; Foldes et al. 2008
report near-zero ES/A Black–White differences, not re-verified). ES and A are
not scored and carry no direct performance path (y remains regression of
performance on g, C[, H] hitting the Sackett validities), so these values
affect nothing downstream. WHO fakes is exactly model 11's mechanism; the
solved gammas reproduce all anchored trait–faking correlations to < .005.

### 7.2 The two inflation variants

**Variant T (targeted):** fakers inflate only C, ES, SE — job-relevant
targeting per Birkeland's own interpretation. C and ES each get a free
mean-inflation intercept; the need and skill slopes are shared:

    infl_C  = m0_C  + m_need·need + m_skill·g + N(0,σ)
    infl_ES = m0_ES + m_need·need + m_skill·g + N(0,σ)
    infl_SE = s₀ + m_skill·g + N(0,σ)          (as model 11; s₀ = 0, no anchor)

Free: (m0_C, m0_ES, m_need, m_skill). The A scale is untouched, so the A
null stays structural for T — what changed is that it is now a *moment*, so
T and D are scored on the same criterion.

**Variant D (desirability-proportional / trait-general):** fakers inflate
all modeled Big Five scales + SE proportionally to a social-desirability
loading vector λ, one free scale parameter κ:

    infl_j = (λ_j/λ_C)·(κ + m_need·need + m_skill·g + N(0,σ)),  j ∈ {C, ES, A, SE}

λ anchor: Ones, Viswesvaran, & Reiss (1996) meta-analytic SD–Big Five
correlations — **λ_ES = .37 and λ_C = .20 verified** (abstract + secondary
full texts); λ_A = .20 and λ_SE = .37 **PROVISIONAL** (Ones Table 6 not
retrievable via open web 2026-08-12; λ_A sits in the SDR-literature band and
is swept at {.10, .30}; λ_SE set equal to λ_ES). Only the ratios matter
after normalization: λ_ES/λ_C = 1.85, λ_A/λ_C = 1.00, λ_SE/λ_C = 1.85.
Free: (κ, m_need, m_skill). D's testable content: it predicts d_ES ≈
1.85·d_C and d_A ≈ d_C — the observed pattern (d_ES/d_C ≈ .98, d_A ≈ 0) can
now reject it.

### 7.3 Calibration: six moments, WLS

Moment vector extends section 2 with the Birkeland ES and A shifts: d_C =
.45, **d_ES = .44, d_A = .00**, applicant r(g, C_obs) by context, applicant
Δr(g, SE_obs) = +.17, observed-C SD ratio = 1.00. New shift weights 1/.05²
(provisional, analogous to d_C). Variant T: 4 free parameters, 6 moments
(d_A structurally zero for T → effectively 5 informative moments,
over-identified by 1). Variant D: 3 free parameters, 6 moments
(over-identified by 3).

### 7.4 Headline result: minimized weighted distance, T vs D

| context | set | **T** | **D** | model-11 (4-trait) reference |
|---|---|---|---|---|
| US | field_ns0 (primary) | **2.03** | **46.7** | 1.97 |
| US | field | 5.18 | 47.5 | 5.11 |
| US | lab | 0.24 | 45.8 | 0.23 |
| Pooled | field_ns0 | **0.69** | **55.7** | — |
| Pooled | field | 2.94 | 56.6 | 2.93 |
| Pooled | lab | 0.24 | 56.2 | 0.23 |
| Non-US | field_ns0 | **0.43** | **101.6** | 0.42 |
| Non-US | field | 2.83 | 106.2 | 2.79 |
| Non-US | lab | 2.35 | 100.1 | 2.42 |

Variant T reproduces model 11's fit almost exactly (the ES scale is absorbed
by its free intercept, m0_ES ≈ 1.08–1.38, and the C-side calibration is
unchanged: US field_ns0 (m0_C, m_need, m_skill) = (1.21, .52, .49) vs model
11's (1.20, .52, .49)). **Variant D is rejected in every context and under
every anchor set** — its best distance (45.8) is ~20× T's worst primary
distance.

**Which moments discriminate.** Under field_ns0:

1. **The Agreeableness null** — D's achieved d_A = +.25/+.27/+.27 (US/
   Pooled/Non-US) against a target of 0: a ~5-weighted-SD violation that
   contributes ≈ 25 to the distance on its own.
2. **The C shift, collaterally** — to limit the A violation the optimizer
   shrinks κ, dragging d_C down to .26/.31/.31 vs .45 (≈ 3–4 weighted SDs).
   D cannot fix one without worsening the other because a single κ scales
   every shift: with λ_A/λ_C = 1, d_A ≈ d_C is *forced*, and the data say
   d_A ≈ 0 while d_C = .45.
3. **The ES/C ratio** — D wants d_ES ≈ 1.85·d_C; observed is ≈ 0.98. Even
   with κ shrunk, achieved d_ES/d_C ≈ 1.3–1.6.
4. **Δr(g, SE_obs), Pooled/Non-US** — λ-proportional SE inflation carries
   the full compensatory need term; the strongly negative applicant r_gC
   targets force m_need up (≈ 2.8–3.6), which pushes r(g, SE_obs) *down*
   (achieved Δr = −.01/−.18 vs +.17). Targeting (T) avoids this because the
   SE scale keeps the skill-only form.

**Sensitivities.** The rejection is robust: with λ_A halved to .10, D's
distance is still 24.7 (US) / 79.8 (Non-US); at λ_A = .30 it is 61.7/112.4.
ES–g = .09 (Anglim) changes nothing (T 2.09 vs 2.03; D 46.4 vs 46.7, US).
For D to survive the A null, λ_A/λ_C would have to be ≈ 0 — i.e.,
desirability loadings so C/ES-specific that "desirability-proportional"
collapses into targeting.

**Interpretation.** With independently anchored desirability loadings, the
trait-general distortion account (fakers inflate everything in proportion to
its desirability) is **quantitatively incompatible** with the Birkeland
shift pattern: the observed nulls on A (and the ~1:1 ES/C shift ratio) now
carry real identifying information, and they reject variant D decisively.
The data favor job-relevant *targeting* — which is Birkeland et al.'s own
reading (rank-order of shifts changes with job type). Caveats for the
manuscript: λ_A and λ_SE are provisional (though the rejection spans the
plausible λ_A band); the corrected van der Linden intercorrelations are a
framing choice; and variant T still satisfies the A null structurally — the
null adjudicates *between* mechanisms rather than testing T in isolation.

### 7.5 Outcomes under the primary spec (field_ns0, variant T)

Selection composite unchanged (.5·g + .5·standardized C_obs; ES/A/SE not
scored). Six-trait vs four-trait (model 11) outcomes at SR .10:

| outcome | US v3 (T) | US model 11 | Non-US v3 (T) | Non-US model 11 |
|---|---|---|---|---|
| % hires changed | 35.8 | 35.5 | 20.2 | 20.2 |
| perf loss % (H path) | 10.3 | 10.0 | 10.0 | 8.0 |
| AIR honest → faked | .389 → .381 | .387 → .372 | .399 → .365 | .387 → .354 |
| mean H faked-in | +.11 | +.11 | −.15 | −.13 |
| mean H displaced | +.46 | +.45 | +.60 | +.62 |

Calibrated inflation parameters are essentially identical to model 11
(differences ≤ .01), so the expansion does not disturb the v2 headline
results; the outcome deltas (largest: Non-US perf loss 8.0 → 10.0) are
within Monte Carlo re-draw noise — the six-dimensional pool is a different
random realization under the same seed, and the honest-baseline AIR moved
comparably (.387 → .399). At SR .30 (new for field_ns0): US 22.0% changed /
9.0% loss; Pooled 22.2% / 8.6%; Non-US 25.8% / 17.9%. Variant D outcomes are
reported in the CSV for completeness but should not be headline numbers —
that specification is rejected by the applicant-pool moments.

### 7.6 What section 7 leaves open

1. λ_A and λ_SE need verified point estimates (Ones et al. 1996 Table 6 or
   Konstabel et al. 2006 domain desirability ratings — both paywalled).
2. E and O remain outside the latent vector (middle version); the Birkeland
   E/O shifts (.11/.13) are small but nonzero and would mildly *favor* a
   weak trait-general component in a full Big Five expansion — worth noting
   in limitations before claiming targeting is the whole story.
3. The corrected-vs-observed intercorrelation choice (van der Linden) should
   be a sensitivity run.
4. A hybrid variant (targeted intercepts + small λ-proportional component)
   would quantify *how much* trait-general distortion the A null tolerates.

## 8. Validation batch (v2 model)

Code: `code/13_study2_validation.py`; results:
`output/study2_validation_results.csv`. Run 2026-08-12: primary spec
field_ns0 (r_gF = −.19, r_CF = 0, r_HF = 0), prevalence .40, σ = .50,
n = 200,000, seed 2026; the three context pools are draw-for-draw identical
to model 11's. Executes section 6 items 2–5 plus the V_H sensitivity from
item 1. Nothing was forced; two of the five analyses produced findings that
qualify the v2 headline numbers (8.1 and 8.5).

### 8.1 Censored-inflation refit (infl⁺ = max(infl, 0), C and SE scales)

**Non-US verdict (the key question): the applicant r(g, C_obs) = −.272
target cannot be fully reached without self-sabotage.** With inflation
floored at zero, the best achievable r under field_ns0 is **−.253**
(distance 0.42 → 2.70, ~6.4×), with collateral misses on Δr_SE (.12 vs .17)
and the SD ratio (1.12 vs 1.00); under the field and lab anchor sets the
shortfall is worse (−.245 at distance 6.02; −.205 at distance 9.51). So the
negative-inflation tail (18–26% of non-US fakers in the uncensored fit) is
not an incidental artifact — it is load-bearing for the last ~.02 of the
−.27 correlation, and purely non-negative within-person compensatory
inflation tops out around −.25. The distance does not "blow up" to outright
infeasibility; report it as a feasibility *frontier*, not a wall.

**US/Pooled surprise: censoring *improves* the fit** (US 1.97 → 0.04,
Pooled 0.65 → 0.03) because the floor supplies the observed-C variance
compression the linear form cannot (achieved SD ratio 1.02 vs 1.11) — but
the fitted mechanism becomes far more aggressive (m_need 2.5–2.9, m_skill
1.6, 25–30% of fakers floored on C, ~63% on SE) and the outcomes roughly
double: US SR .10 performance loss 19.2% vs 10.0% uncensored, hires changed
39.8% vs 35.5%, AIR .387 → .330 (vs .372 uncensored), mean H of faked-in
hires −.26 vs +.11. See §8.5 for why this matters: the four moments do not
pin down the outcome magnitudes across functional forms.

### 8.2 V_H sensitivity {.15, .18, .41}

Immaterial. Only the performance regression is re-solved (calibration does
not involve y), and under field_ns0 the faking indicator is uncorrelated
with H by construction, so the H path barely discriminates the mechanisms:
US SR .10 perf_loss_pct_Hpath moves 9.98 → 9.97 → 9.89 across
{.15, .18, .41} and the extra-loss-vs-random comparison moves +0.77 → +0.82
→ +1.21; Non-US SR .10 moves 8.01 → 8.05 → 8.41 with extra loss −6.9 →
−6.9 → −7.2 (trait-driven faking remains *less* damaging than random there
because the random benchmark needs a steep g-slope to hit −.27). No claim in
the paper is sensitive to the Van Iddekinge–vs–Ones choice.

### 8.3 Monte Carlo SEs (ADEMP; Morris, White, & Crowther 2019)

R = 25 replications (seeds 2026–2050, well under the 30-min budget: ~1.2
min total), fresh 200k pool and full recalibration (propensity solve + WLS)
per replication. Monte Carlo SEs are negligible relative to every contrast
discussed: pct_hires_changed SE ≤ .06 points, perf_loss_pct_Hpath SE ≤ .19,
AIR_faked SE ≤ .002, mean_H_faked_in/displaced SE ≤ .003, calibrated
(m₀, m_need, m_skill) SEs ≤ .004. MC means match the single-seed values to
within ~2 SE (e.g., US SR .10 loss 10.32 ± .15 vs seed-2026's 9.97; Non-US
SR .30 18.14 ± .13 vs 18.04). The single-seed design was adequate; cite the
MC SEs in the manuscript rather than rerunning everything replicated.

### 8.4 r_CF profile grid (r_gF = −.19, r_HF = −.04 fixed)

Distance profiles are shallow and left-sloping: minima at r_CF = −.30 (US,
at the grid edge; dist .015), −.15 (Pooled, .001), −.10 (Non-US, .005).
Because the minima are ≈ 0, the 2×-minimum criterion is degenerate
(absurdly narrow regions) — use the additive min + 3.84 heuristic (1-df
chi-square analogy; heuristic only, the WLS criterion is not a calibrated
deviance). Acceptance regions: **US r_CF ∈ (< −.30, +.112]** — the only
context that adjudicates, and it rejects the field set's +.17 anchor while
accepting everything from 0 leftward; **Pooled and Non-US accept the entire
[−.30, +.20] grid**. Conclusion: the applicant-pool moments rule out
strongly positive C–faking correlations (US), are indifferent across the
plausible negative range, and mildly prefer lab-direction values — the
field_ns0 (r_CF = 0) primary spec sits comfortably inside every region.

### 8.5 Freeing the shared m_skill (m_skill_C ≠ m_skill_SE, just-identified)

**The shared-mechanism restriction is rejected in US/Pooled and the outcome
magnitudes are sensitive to it — the most consequential finding of this
batch.** The just-identified system has a unique solution (verified by
20-start multi-start): US (m₀, m_need, m_skill_C, m_skill_SE) =
(1.23, 2.00, **1.27**, **0.42**), distance 1.97 → 0.03; Pooled
(1.17, 2.00, **1.07**, **0.43**), 0.65 → 0.00; Non-US (0.96, 2.01, **0.49**,
**0.42**), 0.42 → 0.41. In Non-US the slopes nearly coincide and the
restriction is non-binding (outcomes move ≤ 1.7 points), supporting the
shared mechanism there. In US/Pooled the C-scale slope is ~2.5–3× the
SE-scale slope, and the freed fit — like the censored fit of §8.1 — matches
the SD-ratio moment via a heavy-need/heavy-skill configuration that roughly
doubles the damage estimates (US SR .10 loss 18.8% vs 9.97% shared, hires
changed 42.1% vs 35.5%).

**Cross-cutting caveat for the manuscript (8.1 + 8.5).** Two independent
relaxations (censoring; freeing the skill slope) that let the model actually
*hit* all four US/Pooled moments both land on substantially more damaging
mechanisms than the primary shared-skill linear fit, which leaves ~1–2 units
of weighted distance on the table (mostly the SD ratio). The qualitative
claims survive everywhere — faked-in hires are lower-H than under random
faking (indeed more extreme under the refits: mean H of faked-in −.17 to
−.38 vs +.11), and the Non-US compensation story strengthens — but the
US/Pooled *point estimates* of performance loss and hires changed should be
presented as form-dependent (≈ 10–19% and ≈ 35–42% at SR .10), or the
primary spec re-argued (e.g., anchor the SD-ratio target before letting it
drive the mechanism). Non-US outcomes are robust across all forms at SR .10
(8.0–9.2% loss under uncensored/freed; 18.1% only under the censored fit
whose feasibility is itself the finding).

## 9. Six-target recalibration campaign (2026-08-21; triage 5b, 6b–6e)

Executed by `code/16_study2_sixtarget_refit.py` on model 12's machinery
(pools bit-identical); results in `output/study2_sixtarget_refit.csv`,
manuscript v13, Supplemental Tables S9–S11. Section 8's relaxations were
four-target (v2 model); this refit makes them apples-to-apples with the
six-target primary.

- **6b (six-target refit of censored/freed forms).** Section 8's conclusion
  survives almost unchanged: above the constant Agreeableness penalty
  (≈ 3.53), US linear 2.70 → censored 0.83 / freed 0.77; Pooled 1.46 →
  0.74 / 0.61; Non-US disfavors censoring (3.56) and is indifferent to
  freeing (1.38 vs 1.39). US/Pooled relaxed losses 14.0–17.3% (was
  "14–19%" under four targets); aggregate US/pooled range now 8–17%.
  Censored US/pooled integrity gaps 0.56–0.81 SD.
- **6c (uniform reporting rule).** All anchor sets × contexts reported
  (Table S11) with a 2×-context-minimum fit flag; only US/field fails it.
  Lab set is the best-fitting in US/Pooled (3.78/3.79) with unremarkable
  losses (5–12%); its non-US implied loss is 40.0–47.5%, reported and
  argued implausible rather than withheld.
- **6d (rho_CSE/rho_HSE ±.10 grid).** 27 refits; weighted distances move
  < 0.2 from defaults; loss, H-gap, and AIR directions hold in every cell.
  The provisional-values sweep claim is now supported, not asserted.
- **6e (ES–GMA held-out falsification).** Model-implied Δr(g, ES_obs):
  US −.02 (inside Study 1's ES-side CI of +.01 [−.03, .06]), Pooled −.10
  and Non-US −.28 (outside), identical across all three forms. The
  shared-slope tying of ES inflation to the C need/skill components
  overpredicts compensatory ES distortion where compensatory pressure is
  strong; ES elevation is better read as a mean shift. Reported as an
  informative failure (Results robustness, General Discussion,
  Limitations); the T-vs-D comparison and integrity results do not turn
  on this component, but a scale-specific ES equation is the first thing
  within-person applicant data should revise.
- **5b (prevalence sweeps, both models).** 30/50% endpoints, full re-solve
  and refit per prevalence (Study 1: `09`, Table S9; Study 2: script 16).
  Lower prevalence ⇒ larger per-faker inflation ⇒ worse outcomes (Study 1
  losses 12–33% at 30% vs 7–15% at 50%); all directions invariant. Sigma
  sweeps extended to US/Pooled in `09` (S7): directions invariant.

## ADDENDUM (2026-08-25): prevalence nuke-proofing — 20% condition and union-rate definition

Provenance audit of the prevalence anchors (triggered by MJM; sources
verified against the G&C 2011 proof and the Donovan et al. 2014 full text,
see `study2_anchor_table.md` Section 1a):

- **Definition pinned.** P_FAKE is a **union ("faked on ≥1 scale") rate**,
  matching the model's single binary faker indicator. Within-person
  union rates: Griffith et al. (2007) 30–50% (CI-dependent, *N* = 60, C
  scale only); Donovan et al. (2014) 49.7% (*N* = 162, goal-orientation
  scales, per-dimension 14–25%). Per-scale rates run 14–36% across the
  family and are NOT the model's quantity. Stated in Study 1 Method and
  locked by two new fact-check regressions.
- **20% condition added** to `P_FAKE_SENS` (= .30, .50, .20; .20 appended
  last to preserve `09`'s sequential-rng bit-identity) so the sweep spans
  the G&C 2011 synthesis lower bound (30% ±10% ⇒ 20–40%). Results
  (Table S9): Study 2 degrades gradually (worst cell: non-US SR .30 loss
  35.3%); Study 1's random-faker model requires per-faker mean inflation
  of 3.0–4.8 SD at 20% (losses to 78.6%), argued in text as an upper
  bound on harm, not a realistic scenario — bounded scales forbid it.
- All previously reported .30/.40/.50 cells verified bit-identical after
  the reruns (`09`, `11`, `16`; logs `log_*_prev020.txt`).
- ~~Outstanding: Griffith et al. (2007) full text still "abstract only"~~ —
  **RESOLVED 2026-08-26**: full text received from P. Converse
  (`literature/Griffith, Chmielowski, _ Yoshita (2007).pdf`). Verified:
  faker rates are criterion-dependent at 49% (½-SD threshold), 31%
  (1.96 × SEM), 22% (SED); the abstract's 30–50% is the 31–49% band
  rounded (corrects the earlier secondary-source reading of "22–31%").
  Design: 141 applicants, two temp agencies; 60 hired temps retested by
  mail one month later (43% response), honest + fake-good sets. Single
  C scale (CSCS, 30 items) — so Griffith's rate is a single-scale rate,
  while Donovan et al.'s 49.7% is a union rate; the manuscript's
  union-rate sentence attributes the 14–25% per-scale band to Donovan
  only and remains accurate. No manuscript changes required; the 30–50%
  citation and the 40% midpoint stand as written.
