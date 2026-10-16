"""Deterministic manuscript smoke test. No API calls, no model, free to run.

Checks the latest manuscript_v*.md against the project's own sources of truth:

1. Citation cross-reference, both directions (in-text vs. reference list).
2. Abstract word count (<= 250) and keyword count (<= 5).
3. Style lint: banned AI-speak words (see manuscript/mjm_style_notes.md),
   em-dash and rhetorical-colon budgets.
4. Regression checks for previously caught factual errors (Dunlop n = 189,
   Roth et al. 2011 title, 22%-36% hiring range, etc.).
5. Numeric agreement with code/study2_params.py for values the simulation pins.
6. Figure markers point at files that exist in output/.
7. Reference list alphabetical order.

Exit code 0 = all pass; 1 = at least one failure. Run directly or via
06_build_docx.py, which calls this after every build.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS_DIR = ROOT / "manuscript"

EM_DASH_BUDGET = 6
RHETORICAL_COLON_BUDGET = 16

BANNED_WORDS = [
    "provenance", "signature", "casualty", "arbiter", "test bed", "testbed",
    "untraced", "landscape", "tapestry", "delve", "underscores", "pivotal",
    "crucially", "notably,", "leverages", "adjudicat",
]

# Regressions: strings that must be present / absent (errors caught in past passes).
MUST_CONTAIN = [
    ("two-sided applicant C credibility interval (7a)",
     re.escape("[\u2212.28, +.13]")),
    ("US Openness contrast flagged nonestimable (7a)", "could not be estimated"),
    ("Sociability in facet enumeration (7a)", "Sociability"),
    ("observed-vs-corrected convention stated (7b)",
     "are observed correlations"),
    ("OPP keying verification sentence (3c)",
     "reverse-scored toward Conscientiousness"),
    ("Dunlop et al. lab n = 189", r"\*n\* = 189"),
    ("Roth et al. (2011) correct title",
     "input values can affect research conclusions in human resource management"),
    ("hiring-decision range 22%",
     "22%"),
    ("RQ renumbering intact (four research questions)", "Research Question 4"),
    ("suppression labeled correctly", "cooperative suppression"),
    ("v9mjm title (MJM 2026-08-20)",
     "Relationship Between Cognitive Ability and Personality in Personnel "
     "Selection"),
    ("v21 title leads with the coined construct (MJM 2026-10-12)",
     "# Compensatory Faking and the Relationship Between Cognitive Ability"),
    ("hypothesis named 'compensatory faking', distinguished from the "
     "Moutafi/Wood-Englert compensation hypothesis (corpus scan 2026-10-12)",
     "compensatory faking hypothesis"),
    ("plain 'jointly solved' phrasing replaces 'calibrated' and 'method of "
     "moments' (corpus scans 2026-08-20/21)", "jointly solved so that"),
    ("Birkeland A shift stated with point estimate and CI (Table 3, "
     "full-text check 2026-08-20; leading zeros per APA scrub 2026-08-25)",
     r"\*d\* = 0\.16, 95% CI \[−0\.01, 0\.32\]"),
    ("pooled incumbent baseline corrected to +.03 (Table 1 incumbent mean)",
     r"pooled \+\.03"),
    ("database end year corrected to 2014 (analysis_effects.csv max year)",
     "1925 and 2014"),
    ("Birkeland moment weights disclosed as meta-analytic SEs",
     "inverse squared standard errors"),
    ("excluded sample types disclosed (10c; reworded per MJM 2026-08-22)",
     "outside the four stakes-classifiable categories"),
    ("meta-analysis gap statement, approved wording (2026-08-22)",
     "No meta-analysis has compared"),
    ("one-parameter sensitivity analysis heuristic caveat present "
     "(9h; renamed from 'profile analysis' 2026-08-27)",
     "a heuristic screen rather than a formal inference procedure"),
    ("unrounded-contrast convention disclosed (9i)",
     "unrounded subgroup means"),
    ("compensation limit phrased as 'lower bound' not 'frontier' (corpus "
     "scan 2026-08-21)", r"lower bound near −\.26"),
    ("incumbent benchmark justified in Method (2a)", "focal comparison group"),
    ("pooled low-stakes contrasts in main text (2b)",
     "directionally identical but nonsignificant"),
    ("non-US instrument-family concentration disclosed, Roth-2001 style (3a)",
     "contributed a large portion of the non-US applicant data"),
    ("self-selection rival engaged in Discussion (4d)",
     "shaped by self-selection"),
    ("abstract states regional boundary (5a)",
     "concentrated in non-US selection settings"),
    ("US-demography caveat on adverse-impact projections (7c)",
     "US demographic assumptions"),
    ("ADEMP claim scoped to 'organized around' (MJM 2026-08-22)",
     "organized around the ADEMP framework"),
    ("estimand tripartite definition up front (MJM 2026-08-22)",
     r"estimand \(the quantity a simulation is designed to learn\)"),
    ("Study 1 Monte Carlo error reported from 25 replications "
     "(17_study1_mcse.py)", r"seeds 2026\u20132050"),
    ("Study 2 MCSEs reported for all four estimands",
     r"\.002 for adverse-impact ratios"),
    ("Study 2 Method opens with ADEMP mapping",
     "reported in the ADEMP structure introduced in Study 1"),
    ("Holtrop uniqueness scoped to genuine JOB applicants after "
     "Weissenbacher & Krammer 2025 (teacher-education applicants, "
     "within-person, intelligence positively predicted faking) was found "
     "(MJM re-check 2026-08-27)",
     "Only one study of genuine job applicants to date has measured"),
    ("Weissenbacher, Jud, & Krammer 2025 counterexample cited "
     "(Crossref author list 2026-10-12; was missing Jud)",
     "Wei\u00dfenbacher et al., 2025"),
    ("Stanek & Ones dataset cited separately from the PNAS article "
     "(CC BY-NC-ND attribution; DataCite 10.13020/7YS8-ET75)",
     r"Stanek and Ones \(2023c\)"),
    ("dataset license named in the manuscript",
     "CC BY-NC-ND 3.0 US"),
    ("relaxations refit against all six targets (6b)",
     "refit against all six empirical targets"),
    ("uniform anchor-set reporting rule (6c)", "uniform reporting rule"),
    ("prevalence-sweep direction (5b)",
     "lower prevalence concentrates larger inflation"),
    ("prevalence = union/any-scale rate, matching the binary faker "
     "indicator (nuke-proofing pass 2026-08-25)",
     "proportion of applicants who fake at all"),
    ("20% prevalence condition flagged as upper bound on harm, not a "
     "realistic scenario (2026-08-25)",
     "upper bound on harm rather than a realistic scenario"),
    ("ES-GMA held-out falsification check reported (6e)",
     "exposed the targeted mechanism to falsification"),
    ("integrity gap reported under both H sources: HEXACO linear range "
     "(Howard & Van Zandt 2020 run, 2026-10-12)", "0\\.16\u20130\\.45 \\*SD\\*"),
    ("integrity gap reported under both H sources: overt-test linear range "
     "(Sackett & Wanek 1996 via Lee et al. 2005 run, 2026-10-12)",
     "0\\.29\u20130\\.70 \\*SD\\*"),
    ("Sackett & Wanek cited as secondary source (APA 7; only Lee et al. "
     "2005 PDF in hand)", r"Sackett & Wanek, 1996, Table 3, as cited in Lee et al\., 2005"),
    ("Table 6 carries both H-gap columns", r"\| H gap, HEXACO \| H gap, overt tests \|"),
]
MUST_NOT_CONTAIN = [
    ("Table 2 Sociability p printed .080; facet_scan.csv p = .0794 -> .079 "
     "(caught by Grok stats validation, 2026-10-12)",
     r"Sociability \(E facet\) \|[^\n]*\| \.080 \|"),
    ("30-50% misattributed to G&C 2011 chapter: chapter concludes 30% \u00b110% "
     "(pp. 36, 47 of proof); 30-50% is Griffith et al. 2007 / Donovan et al. "
     "2014 (caught from chapter proof, 2026-08-25)",
     r"\(Griffith & Converse, 2011; Griffith et al\., 2007\)"),
    ("40% prevalence cited to G&C 2011 chapter alone (same error)",
     r"prevalence \(Griffith & Converse, 2011\)"),
    ("ADEMP overclaim: 'follow' implies full estimator-evaluation design; "
     "scoped to 'organized around' (MJM 2026-08-22)",
     "reporting follow the ADEMP"),
    ("one-sided credibility interval (7a)", "interval extending to"),
    ("random faked-in 'near the pool average' overstatement (7a)",
     "hires were near the pool average"),
    ("promissory interrater-agreement sentence (4a: operational definition, no IRR)",
     "interrater agreement will be reported"),
    ("stale Dunlop n = 201", r"\*n\* = 201"),
    ("stale hiring range 23%\u201336%", "23%\u201336%"),
    ("stale Roth title (data errors)",
     "data errors affect research conclusions"),
    ("imprecise hire-change range (use 20%-36%)", "quarter to a third"),
    ("imprecise integrity gap (use 0.3-0.8 SD)", "one third to four fifths"),
    ("'only within-person study' overclaim", "only within-person study"),
    ("'single study' framing dropped: five within-person applicant designs "
     "exist (Griffith 2007; Ellingson 2007; Peterson 2011; Arthur 2010; "
     "Donovan 2014); uniqueness is measured predictors (MJM triple-check "
     "2026-08-22)", "comes from a single study to date"),
    ("old Study 2 heading (MJM v17mjm hand edit 2026-08-22: 'Trait-Driven "
     "Faking and Hiring Outcomes')", "Its Consequences for Who Is Hired"),
    ("'propensity' not corpus language (MJM 2026-08-26; anchor papers say "
     "predictors/determinants of faking; Morris propensity-score collision); "
     "use anchor sets / trait-faking correlations / who fakes",
     "(?i)propensit"),
    ("'sweep/swept' simulation jargon (MJM 2026-08-27); use 'varied across'",
     r"(?i)\bswe(ep|pt|eping)\b"),
    ("'profile analysis' names a distinct MANOVA technique (MJM 2026-08-27); "
     "use 'one-parameter sensitivity analysis'", "profile analysis"),
    ("evidence/pools don't 'sit poorly' (MJM 2026-08-27)", "sit poorly"),
    ("estimates don't 'decide'; studies aren't people (MJM 2026-08-27); "
     "say the sets can be tested against the estimates", "can decide between"),
    ("'the fit of each parameter set' reads as fit statistics "
     "(MJM 2026-08-27); say how closely each set reproduced the targets",
     "fit of each parameter set"),
    ("old unscoped Holtrop uniqueness claim (superseded 2026-08-27)",
     "Only one such study to date"),
    ("desirability treated as a quantity a scale possesses (MJM 2026-08-27); "
     "say correlations with social desirability scales (Ones et al., 1996)",
     r"(scale's social desirability|desirability loading)"),
    ("'misfit' is IRT/SEM vocabulary, absent from corpus and from Morris "
     "et al. 2019 (MJM 2026-08-27); say the targets were missed / distance "
     "reported", "(?i)mis-?fit"),
    ("'held-out' is ML jargon, absent from Morris et al. 2019 and corpus "
     "(Morris audit 2026-08-27); say 'a quantity the fit never used'",
     "(?i)held-out"),
    ("'validation results/battery' dangles since the Method says robustness "
     "(Morris audit 2026-08-27)", "validation (results|battery)"),
    ("Schilling field-cell overclaim (their field C is significantly positive)",
     "leaves open what operational selection settings"),
    ("suppressor mislabel (use cooperative suppression; Tzelgov & Henik 1991)",
     "partial suppressor"),
    ("overclaim 'first meta-analytic evidence' (Schilling 2021 exists)",
     "first meta-analytic evidence"),
    ("competing hypotheses (convert to research questions per MJM 2026-08-20)",
     "Hypothesis 1a"),
    ("'transportability' misuse (Tang 2026 is not a transportability study)",
     "transportab"),
    ("metaphorical 'register' (style notes)", "can register"),
    ("metaphorical 'registering' (style notes)", "without registering"),
    ("'matched' samples overclaim (no matching procedure was used)",
     "matched lower-stakes"),
    ("'first' comparison overclaim (Schilling 2021 Table 4 reports per-dimension "
     "field vs. nonselection estimates)", "the first construct-by-construct"),
    ("region claim overreach: 6 of 10 US applicant samples are negative; the "
     "reversal is a subgroup-mean result", "confined to non-US"),
    ("region claim overreach (same issue as 'confined to non-US')",
     "concentrated entirely in non-US"),
    ("econ-speak 'calibrat*' (corpus scan 2026-08-20: only Sackett et al. 2022 "
     "uses it, for validity re-estimation; sim papers say 'based on' "
     "meta-analytic estimates)", "(?i)calibrat"),
    ("RQ parenthetical in a table title (APA 7: keep titles substantive; map "
     "RQs in the note)", r"\(Research Question \d\)\*"),
    ("Birkeland A mischaracterized as d ≈ 0 (Table 3 reports d = .16)",
     "≈ 0; Birkeland"),
    ("Birkeland A 'essentially none' overclaim (d = .16, same band as E/O)",
     "essentially none for Agreeableness"),
    ("stale pooled incumbent baseline (.01 was the total-database r, not an "
     "incumbent estimate)", r"pooled \+\.01"),
    ("'novel lens' first-claim hygiene (removed 2026-08-20)", "novel lens"),
    ("Schilling nonselection mischaracterized as null (O/ES/A nonselection "
     "CIs exclude 0)", "but not in nonselection"),
    ("stale database end year (max year in analysis file is 2014)",
     "1925 and 2013"),
    ("stale limitations end year", "ends at 2013"),
    ("US case attached to pooled/non-US ranges (US losses are 8%-10%)",
     "even in the US case"),
    ("Dunlop demographics overstated (gender only, partial r = .14, p = .051)",
     "once demographics were controlled"),
    ("De Fruyt 2006 overlap error (it IS in the database, coded "
     "fluid/induction)", "none of the samples appear in the Stanek"),
    ("causal 'stakes effect' phrasing (identification sweep 8a)",
     "stakes effect"),
    ("stale Table 5 distance from the d_A = 0 run", "46.83"),
    ("econometrics 'moment(s)' (corpus scan 2026-08-21: the 14 local PDFs use "
     "the word only in 'product-moment correlation'; say 'targets' or "
     "'empirical targets')", r"(?i)(?<!product-)(?<!product )moment"),
    ("econ-speak 'frontier' (corpus scan 2026-08-21: local PDFs say "
     "'lower/upper bound'; plural 'Frontiers' journal name is exempt)",
     r"(?i)\bfrontier\b"),
    ("'fixed reference point' overclaim (GMA distributions vary across "
     "contexts through selection and practice)", "fixed reference point"),
    ("trait-as-agent phrasing: traits don't shift, statistics differ "
     "(MJM 2026-08-21)", "Agreeableness does not shift"),
    ("trait-as-agent phrasing, Study 2 variant", "it must shift in applicant"),
    ("editorializing 'discover accidentally' (7c)", "discover accidentally"),
    ("stale four-target relaxation loss range (six-target refit gives "
     "14%\u201317%)", "14%\u201319%"),
    ("stale four-target censored integrity gap (six-target refit gives "
     "0.56\u20130.81)", "0.56\u20130.80"),
    ("stale lab non-US loss range from the prototype model (six-target refit "
     "gives 39%\u201348% under the Howard run)", "40%\u201350% of hire performance"),
    ("stale integrity gap from the .42/.33/.45 run (Blonigen et al. 2011 "
     "values mis-attributed to Ones 1993; replaced 2026-10-12)",
     "0\\.31\u20130\\.75"),
    ("stale ceiling-form gap from the .42 run", "0\\.56\u20130\\.81"),
    ("stale linear loss range from the .42 run", "8\\.1%\u20139\\.9%"),
    ("stale random-faker H range from the .42 run", "\u22120\\.01 to \\+0\\.46"),
    ("stale Table 5 primary distances from the .42 run (now 6.20/4.95/4.90)",
     r"\| 6\.23 \| 4\.99 \| 4\.91 \|"),
    ("stale lab non-US loss range from the .42 run", "40%\u201348% of hire performance"),
    ("AI-speak 'propagat*' (corpus scan 2026-08-22: 0 hits in local PDFs)",
     "(?i)propagat"),
    ("AI-speak 'sanity check' (corpus scan 2026-08-22: 0 hits in local PDFs)",
     "(?i)sanity"),
    ("'patient samples' itemization (MJM 2026-08-22: fold into the "
     "stakes-classifiable category wording)", "patient samples"),
    ("overstated 'have not been examined' (Schilling 2021 has an "
     "implications-for-selection section; say discussed but not quantified)",
     "have not been examined"),
    ("'directional hypotheses' phrasing (MJM removed 2026-08-22)",
     "directional hypotheses"),
    ("bare software package list (MJM 2026-08-22: name language + version "
     "only; packages did not perform the estimation)", "pandas, NumPy"),
    ("taxonomy label 'sub-Big-Five' (corpus: Stanek-Ones say facets, "
     "aspects, compound traits)", "sub-Big-Five"),
    ("overclaim 'decisive' (0 corpus hits; report significance instead)",
     "(?i)decisive"),
    ("'plug-in' model name (0 corpus hits; say analytic model)", "plug-in"),
    ("'mechanism model' (0 corpus hits; say Monte Carlo simulation)",
     "mechanism model"),
    ("undefined 'Era analyses' opener (era coding now defined in Method)",
     "Era analyses"),
    ("label-style 'Applied Aim:' heading (corpus headings are substantive "
     "noun phrases)", "Applied Aim"),
    ("AI-speak 'is sharpest' pattern claim (MJM 2026-08-22)", "is sharpest"),
    ("dataset DOI attached to the PNAS article (2023a); the data set is 2023c",
     r"2023a\) database \(https://doi\.org/10\.13020"),
    ("two-author Weißenbacher citation (Jud is the second author)",
     "Wei\u00dfenbacher & Krammer"),
]

# Values the manuscript must state consistently with code/study2_params.py.
PARAM_CHECKS = [
    ("D_SHIFT", r"D_SHIFT\s*=\s*([\d.]+)", "Birkeland C shift d"),
    ("D_SHIFT_A", r"D_SHIFT_A\s*=\s*([\d.]+)", "Birkeland A shift d"),
    ("P_FAKE", r"P_FAKE\s*=\s*([\d.]+)", "faking prevalence"),
    ("D_G", r"D_G\s*=\s*([\d.]+)", "GMA subgroup d"),
]

MULTIWORD_CITE_OK = {
    "De",       # De Corte
    "Van",      # Van Iddekinge, van der Linden (capitalized at sentence start)
    "Equal",    # Equal Employment Opportunity Commission et al.
    "Local",    # Local Law No. 144
    "Society",  # SIOP (if ever re-cited)
}


def latest_manuscript() -> Path:
    cands = sorted(MS_DIR.glob("manuscript_v*.md"),
                   key=lambda p: int(re.search(r"v(\d+)", p.stem).group(1)))
    return cands[-1]


def main() -> int:
    failures, warnings = [], []
    path = latest_manuscript()
    md = path.read_text(encoding="utf8")
    parts = md.split("# References")
    body, back = parts[0], parts[1]
    # Tables, figure notes, and appendix follow the reference list; include
    # them in prose-quality scans (reference-list titles are excluded since
    # they quote other authors' wording).
    t = md.find("**Table 1**")
    prose = body + (md[t:] if t != -1 else "")

    # 1. Citation cross-reference.
    ref_entries = re.findall(r"^([A-Z][A-Za-z\u00c0-\u017f\u2019'-]+),", back, re.M)
    ref_surnames = set(ref_entries)
    cited = set()
    for m in re.finditer(r"\(([^()]*?\d{4}[a-z]?[^()]*?)\)", body):
        for chunk in m.group(1).split(";"):
            mm = re.match(r"\s*(?:e\.g\., |cf\. |see |via )?([A-Z][A-Za-z\u00c0-\u017f\u2019'-]+)", chunk)
            if mm and re.search(r"\d{4}", chunk):
                cited.add(mm.group(1))
    for m in re.finditer(r"([A-Z][A-Za-z\u00c0-\u017f\u2019'-]+)(?: et al\.| and [A-Z][a-z]+| & [A-Z][a-z]+)? \(\d{4}[a-z]?\)", body):
        cited.add(m.group(1))
    missing = sorted(c for c in cited
                     if c not in ref_surnames and c not in MULTIWORD_CITE_OK)
    if missing:
        failures.append(f"cited in text but no reference entry: {missing}")
    plain_body = re.sub(r"\s+", " ", body)
    orphans = sorted(s for s in ref_surnames if s not in plain_body)
    if orphans:
        failures.append(f"reference entries never cited in body: {orphans}")

    # 2. Abstract and keywords.
    m = re.search(r"## Abstract\s+(.+?)\s+\*Keywords:\*(.+)", md, re.S)
    if m:
        n_words = len(re.sub(r"[*]", "", m.group(1)).split())
        if n_words > 250:
            failures.append(f"abstract is {n_words} words (limit 250)")
        n_kw = len(m.group(2).split("\n")[0].split(","))
        if n_kw > 5:
            failures.append(f"{n_kw} keywords (limit 5)")
    else:
        failures.append("could not locate abstract/keywords block")

    # 3. Style lint (body + tables/figures/appendix; reference list excluded).
    for w in BANNED_WORDS:
        hits = len(re.findall(re.escape(w), prose, re.I))
        if hits:
            failures.append(f"banned word '{w}' appears {hits}x in prose")
    dashes = body.count("\u2014")
    if dashes > EM_DASH_BUDGET:
        warnings.append(f"{dashes} em dashes in body (budget {EM_DASH_BUDGET})")
    colons = len(re.findall(r"[a-z\)]: [a-z]", body))
    if colons > RHETORICAL_COLON_BUDGET:
        warnings.append(f"{colons} mid-sentence colons (budget {RHETORICAL_COLON_BUDGET})")

    # 4. Regression checks.
    for label, pat in MUST_CONTAIN:
        if not re.search(pat, md):
            failures.append(f"missing expected content: {label}")
    for label, pat in MUST_NOT_CONTAIN:
        if re.search(pat, md):
            failures.append(f"stale content present: {label}")

    # 5. Agreement with simulation parameters.
    params_src = (ROOT / "code" / "study2_params.py").read_text(encoding="utf8")
    for name, pat, label in PARAM_CHECKS:
        pm = re.search(pat, params_src)
        if not pm:
            warnings.append(f"could not read {name} from study2_params.py")
            continue
        val = float(pm.group(1))
        as_stat = f"{val:.2f}".lstrip("0") if val < 1 else f"{val:g}"
        pct = f"{val:.0%}"
        if as_stat not in md and pct not in md:
            failures.append(
                f"{label}: params say {val} but neither '{as_stat}' nor '{pct}' found in manuscript")

    # 6. Figure files exist.
    for fig in re.findall(r"<!--figure:([^>]+?)-->", md):
        if not (ROOT / "output" / fig.strip()).exists():
            failures.append(f"figure marker points at missing file: {fig.strip()}")

    # 6b. Title page consistency (latest title_page_v*.md).
    tp_cands = sorted(MS_DIR.glob("title_page_v*.md"),
                      key=lambda p: int(re.search(r"v(\d+)", p.stem).group(1)))
    if tp_cands:
        tp = tp_cands[-1].read_text(encoding="utf8")
        if "hypothes" in tp.lower():
            failures.append(
                f"title page ({tp_cands[-1].name}) still says 'hypotheses'; "
                "manuscript uses research questions")
        if "Stanek & Ones, 2023)" in tp:
            failures.append(
                f"title page ({tp_cands[-1].name}) cites 'Stanek & Ones, "
                "2023' without the a/b suffix (manuscript uses 2023a)")

    # 7. Reference list order.
    if ref_entries != sorted(ref_entries, key=str.casefold):
        pairs = [(a, b) for a, b in zip(ref_entries, ref_entries[1:])
                 if str.casefold(a) > str.casefold(b)]
        warnings.append(f"reference list not alphabetical near: {pairs[:3]}")

    print(f"fact-check target: {path.name}")
    for w in warnings:
        print(f"  WARN  {w}")
    for f in failures:
        print(f"  FAIL  {f}")
    if not failures:
        print(f"  OK    all checks passed ({len(warnings)} warnings)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
