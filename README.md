# Compensatory Faking and the Relationship Between Cognitive Ability and Personality in Personnel Selection

Code, aggregate results, and manuscript for

> Monnot, M. J. (2026). *Compensatory faking and the relationship between cognitive ability and
> personality in personnel selection: Meta-analytic evidence and trait-driven simulations.*
> SSRN. https://doi.org/10.2139/ssrn.7611561

Preprint: [https://doi.org/10.2139/ssrn.7611561](https://doi.org/10.2139/ssrn.7611561)
(PDF copy in `manuscript/`). Interactive companion (faking-cost calculator):
[mjmonnot.github.io/faking-costs](https://mjmonnot.github.io/faking-costs/).

## What the paper finds

Study 1 meta-analyzes the Stanek and Ones (2023) personality-ability database by assessment
stakes. The Conscientiousness-GMA correlation reverses sign between incumbents (r = .03) and
genuine applicants (r = -.07; delta = -.11, 95% CI [-.18, -.03]); the other Big Five factors do
not shift. The pattern is consistent with *compensatory faking* (lower-ability applicants
inflating Conscientiousness more) and inconsistent with trait-general distortion. A Monte Carlo
model fit to those estimates indicates that at 40% faking prevalence, distortion changes 22-36%
of top-down hiring decisions and lowers mean hire performance by 8-19%.

Study 2 assigns faking from applicant traits anchored to published trait-faking correlations.
Applicants who gain hiring slots through modeled faking average 0.2-0.4 SD lower in
honesty-humility than those they displace (0.3-0.7 under overt integrity-test correlations).

## Repository layout

| Path | Contents |
|---|---|
| `code/` | Fully scripted pipeline, raw database to every table and figure (Python 3.14) |
| `output/` | Aggregate tables (`table*.csv`, robustness checks, Study 2 results), figures, run logs |
| `manuscript/` | Markdown sources (`manuscript_v24.md`, `title_page_v10.md`, `supplement_v6.md`) and the preprint and supplement PDFs |
| `literature/` | Study 2 anchor table and model notes (parameter provenance) |
| `database/`, `data/` | Placeholders; see the READMEs inside. The database is not redistributed |

## Reproducing the analyses

1. Download the Stanek and Ones (2023) database from https://doi.org/10.13020/7YS8-ET75 and put
   the xlsx in `database/` (see `database/README.md`).
2. `pip install -r requirements.txt`
3. Run in order:

```
python code/00_load_database.py          # xlsx -> database/db.pkl
python code/01_prepare_data.py           # reversals, g x Big Five filter, stakes coding, compositing -> data/analysis_effects.csv
python code/02_run_analyses.py           # Hunter-Schmidt estimates by stakes, contrasts, sensitivity -> output/table1, table2, table3
python code/04_presubmission_checks.py   # leave-one-out, publication bias, reliability grid, facets
python code/05_composite_simulation.py   # validity-diversity simulation -> table4
python code/07_facet_scan.py             # facet and compound-trait scan
python code/08_moderator_scrub.py        # region, publication type, era, measure moderators
python code/09_selection_simulation.py   # calibrated selection consequences -> table7, figure3
python code/10_explainer_figure.py       # mechanism explainer -> figure4
python code/11_study2_trait_model.py     # Study 2: anchored trait-driven faking model
python code/12_study2_expanded_model.py  # Study 2: six-trait expansion, mechanism comparison
python code/13_study2_validation.py      # Study 2: replications, profile sweep, form relaxations
python code/16_study2_sixtarget_refit.py # Study 2: six-target refit
python code/17_study1_mcse.py            # Monte Carlo SEs for the Study 1 simulation
python code/14_forest_plot.py            # Figure 2 and supplemental forest plots S2-S5
python code/06_build_docx.py             # rebuilds the manuscript, title page, and supplement DOCX from the markdown
python code/07_fact_check.py             # deterministic consistency checks on the manuscript
```

Study 2 runs under two integrity-correlation sources (`STUDY2_HH_SOURCE=howard | overt` in
`code/study2_params.py`); outputs under the overt source carry the `_hh_overt` suffix.

## Data statement

Study 1 is a secondary analysis of the publicly available Personality Traits and Cognitive
Abilities Relations Database (Stanek & Ones, 2023; https://doi.org/10.13020/7YS8-ET75), used
under CC BY-NC-ND 3.0 US. The database, and sample-level derivatives of it
(`data/analysis_effects.csv`, the applicant audit list), are not redistributed; only aggregate
estimates are committed. Study 2 is a computer simulation with no human-participant data. The
database authors were not involved in this study.

## License

Code: MIT. Manuscript text, tables, and figures: CC BY-NC-ND 4.0. See `LICENSE`.

## Contact

Matthew J. Monnot, Vocentive, San Francisco. matt@vocentive.com.
ORCID 0000-0002-9489-1777.
