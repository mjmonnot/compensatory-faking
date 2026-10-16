"""Build the v24 submission package from markdown sources:

- manuscript/Manuscript_A_v24.docx        <- manuscript_v24.md (full: unmasked APA title page embedded)
- manuscript/Manuscript_A_v24_masked.docx <- manuscript_v24.md (masked review copy: title-only page 1)
- manuscript/Title_Page_v10.docx          <- title_page_v10.md (separate, unmasked)
- manuscript/Supplement_v6.docx           <- supplement_v6.md

v24 = v23 with the abstract cut 250 -> 192 words and the running head shortened
to < 40 characters, both for Personnel Psychology submission requirements
(2026-10-14). Prose otherwise identical to v23 (the SSRN preprint).

APA 7 formatting: Times New Roman 12 pt, double-spaced, 0.5-in first-line indents
for body prose; Level 1 headings centered bold, Level 2 flush-left bold, Level 3
flush-left bold italic; tables with horizontal rules only (top, below header,
bottom), single-spaced; <!--pagebreak--> markers honored; ~x~ renders subscript;
<!--figure:NAME.png--> inserts output/NAME.png in place (APA 7 figure pages:
bold label, italic title, image, note).

Tables use a fixed layout with column widths sized to content so cells hold a
single line: statistics columns never wrap (non-breaking spaces + noWrap);
columns whose longest cell exceeds PROSE_COL_CHARS are treated as prose, share
the leftover width, and may wrap. Pages whose table cannot fit the portrait
text width (6.5 in) are set in landscape via section breaks; table font steps
down 10 pt -> 9 pt before any proportional width scaling.
"""

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.text import (WD_ALIGN_PARAGRAPH, WD_LINE_SPACING,
                            WD_TAB_ALIGNMENT)
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"

# Running head: abbreviated title, all caps. APA 7 allows <= 50 characters,
# but Personnel Psychology requires a running title < 40 characters (checked
# against the author guidelines 2026-10-14), so the tighter bound governs.
RUNNING_HEAD = "COMPENSATORY FAKING IN SELECTION"

DOCS = [
    (MS / "manuscript_v24.md", MS / "Manuscript_A_v24.docx", []),
    (MS / "title_page_v10.md", MS / "Title_Page_v10.docx", []),
    (MS / "supplement_v6.md", MS / "Supplement_v6.docx", []),
]

INLINE = re.compile(r"(\*\*.+?\*\*|\*.+?\*|`.+?`|~.+?~)")

PORTRAIT_USABLE = 6.5    # text width in inches at 1-in margins
LANDSCAPE_USABLE = 9.0
CHAR_IN = {10: 0.072, 9: 0.065}   # average char width for Times New Roman
PAD_IN = 0.15                     # per-column cell padding allowance
PROSE_COL_CHARS = 45              # columns longer than this may wrap
PROSE_MIN_IN = 2.0                # minimum width reserved for a prose column
NBSP = "\u00A0"


def add_runs(par, text: str) -> None:
    """Render **bold**, *italic*, `code`, and ~subscript~ inline markup."""
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            par.add_run(piece[2:-2]).bold = True
        elif piece.startswith("*") and piece.endswith("*"):
            par.add_run(piece[1:-1]).italic = True
        elif piece.startswith("`") and piece.endswith("`"):
            run = par.add_run(piece[1:-1])
            run.font.name = "Courier New"
        elif piece.startswith("~") and piece.endswith("~") and len(piece) > 2:
            run = par.add_run(piece[1:-1])
            run.font.subscript = True
        else:
            par.add_run(piece)


def no_break_hyphens(par) -> None:
    """Replace ASCII hyphens with Word non-breaking hyphens inside a run.

    w:noWrap stops Word from breaking a cell at spaces but not at hyphens, so a
    narrow statistics column can still split "Non-US" across two lines.
    """
    for run in par.runs:
        if "-" not in run.text:
            continue
        parts = run.text.split("-")
        r = run._r
        for t in r.findall(qn("w:t")):
            r.remove(t)
        for k, part in enumerate(parts):
            if part:
                t = OxmlElement("w:t")
                t.text = part
                t.set(qn("xml:space"), "preserve")
                r.append(t)
            if k < len(parts) - 1:
                r.append(OxmlElement("w:noBreakHyphen"))


def force_times_new_roman(doc: Document) -> None:
    """Remove theme-font attributes (Calibri/Calibri Light) and pin Times New Roman.

    python-docx's style.font.name sets w:ascii/w:hAnsi, but Word's asciiTheme/
    hAnsiTheme attributes take precedence when present, so headings and body render
    in the theme fonts unless the theme attributes are stripped.
    """
    theme_attrs = [qn(f"w:{a}") for a in
                   ("asciiTheme", "hAnsiTheme", "cstheme", "eastAsiaTheme")]

    def pin(rpr_parent):
        rpr = rpr_parent.find(qn("w:rPr"))
        if rpr is None:
            rpr = OxmlElement("w:rPr")
            rpr_parent.insert(0, rpr)
        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts")
            rpr.insert(0, rf)
        for a in theme_attrs:
            rf.attrib.pop(a, None)
        for a in ("w:ascii", "w:hAnsi", "w:cs"):
            rf.set(qn(a), "Times New Roman")

    # document-wide run defaults
    doc_defaults = doc.styles.element.find(qn("w:docDefaults"))
    if doc_defaults is not None:
        rpr_default = doc_defaults.find(qn("w:rPrDefault"))
        if rpr_default is not None:
            pin(rpr_default)
    # every paragraph/character style
    for style in doc.styles:
        el = style.element
        if el.tag == qn("w:style"):
            pin(el)


def style_doc(doc: Document) -> None:
    force_times_new_roman(doc)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    normal.paragraph_format.space_after = Pt(0)
    for h in ("Heading 1", "Heading 2", "Heading 3"):
        st = doc.styles[h]
        st.font.name = "Times New Roman"
        st.font.color.rgb = None
        st.font.bold = True
        st.font.size = Pt(12)
        st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
        st.paragraph_format.space_after = Pt(0)
        st.paragraph_format.space_before = Pt(0)
    doc.styles["Heading 3"].font.italic = True


def set_orientation(section, landscape: bool) -> None:
    if landscape:
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width, section.page_height = Inches(11), Inches(8.5)
    else:
        section.orientation = WD_ORIENT.PORTRAIT
        section.page_width, section.page_height = Inches(8.5), Inches(11)
    for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, m, Inches(1))


def add_running_head(doc: Document, text: str, section=None) -> None:
    """APA 7 header: running head flush left, page number flush right."""
    sec = doc.sections[0] if section is None else section
    sec.header.is_linked_to_previous = False
    par = sec.header.paragraphs[0]
    par.text = ""
    par.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    usable = sec.page_width - sec.left_margin - sec.right_margin
    par.paragraph_format.tab_stops.add_tab_stop(usable, WD_TAB_ALIGNMENT.RIGHT)
    run = par.add_run(text + "\t")
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    # PAGE field for automatic page numbers
    fld = par.add_run()
    fld.font.name = "Times New Roman"
    fld.font.size = Pt(12)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for el in (begin, instr, end):
        fld._r.append(el)


def render_title_page(doc: Document, md_path: Path) -> None:
    """Render an APA 7 professional title page (Publication Manual Ã‚Â§2.3-2.7)
    into an existing document: bold centered title three to four lines below
    the top margin, centered author and affiliation, then the author note in
    the bottom half of the page under a centered bold label."""
    lines = [ln.strip() for ln in md_path.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    title = next(ln.lstrip("#").strip() for ln in lines if ln.startswith("#"))
    body = [ln for ln in lines if not ln.startswith("#")]

    for _ in range(2):  # title sits two to three double-spaced lines down
        # (reduced from 3 on 2026-10-14: the PsyArXiv-required ethics statement
        # lengthened the author note; the correspondence email must stay on p. 1)
        doc.add_paragraph()
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.add_run(title).bold = True
    doc.add_paragraph()  # one blank double-spaced line before the byline

    note_seen = False
    for ln in body:
        if ln in ("**Author Note**", "Author Note"):
            note_seen = True
            # No blank line before the note: with the data-availability paragraph
            # plus the ethics statement, the author note must start high or the
            # correspondence line (and contact email) lands on page 2 -- PsyArXiv
            # rejected the preprint for that (2026-10-13).
            pass
            par = doc.add_paragraph()
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            par.add_run("Author Note").bold = True
            continue
        par = doc.add_paragraph()
        add_runs(par, ln)
        if note_seen:
            par.paragraph_format.first_line_indent = Inches(0.5)
        else:
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER


def build_title_page(md_path: Path, out_path: Path) -> None:
    doc = Document()
    style_doc(doc)
    add_running_head(doc, RUNNING_HEAD)
    render_title_page(doc, md_path)
    doc.save(out_path)
    print("Saved", out_path)


def set_cell_border(cell, top: bool = False, bottom: bool = False) -> None:
    """APA 7: horizontal rules only. Left, right, and unspecified edges are nil."""
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge, on in (("top", top), ("bottom", bottom), ("left", False), ("right", False)):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single" if on else "nil")
        el.set(qn("w:sz"), "6" if on else "0")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000" if on else "auto")
        borders.append(el)


def clear_table_borders(table) -> None:
    """Drop the Table Grid style's vertical and interior rules."""
    tbl_pr = table._tbl.tblPr
    existing = tbl_pr.find(qn("w:tblBorders"))
    if existing is not None:
        tbl_pr.remove(existing)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "nil")
        el.set(qn("w:sz"), "0")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "auto")
        borders.append(el)
    tbl_pr.append(borders)


def parse_table(lines: list[str]) -> list[list[str]]:
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    return [r for r in rows if not all(set(c) <= {"-", ":", " "} for c in r)]


def visible_len(cell: str) -> float:
    """Approximate rendered width in average-character units.

    Capitals and m/w run about 1.35x an average lowercase letter in Times New
    Roman; punctuation and spaces about half. A plain character count let
    "Non-US" overflow a column sized from the header "Context".
    """
    text = re.sub(r"[*`~]", "", cell)
    width = 0.0
    for ch in text:
        if ch.isupper() or ch in "mwMW%":
            width += 1.35
        elif ch in " .,;:'|()[]-\u2212\u2013":
            width += 0.55
        else:
            width += 1.0
    return width


def col_maxlens(rows: list[list[str]]) -> list[int]:
    n_cols = max(len(r) for r in rows)
    return [max((visible_len(r[j]) for r in rows if len(r) > j), default=1)
            for j in range(n_cols)]


def plan_widths(rows: list[list[str]], usable: float) -> tuple[list[float], list[bool], int]:
    """Column widths, per-column wrap permission, and table font size.

    Statistics columns (longest cell <= PROSE_COL_CHARS) get content-sized fixed
    widths and never wrap. Prose columns share the leftover width and may wrap.
    Font steps 10 pt -> 9 pt, then widths scale proportionally as a last resort.
    """
    maxlens = col_maxlens(rows)
    prose = [m > PROSE_COL_CHARS for m in maxlens]
    for font in (10, 9):
        content = [m * CHAR_IN[font] for m in maxlens]
        need = (sum(c + PAD_IN for c, p in zip(content, prose) if not p)
                + PROSE_MIN_IN * sum(prose))
        if need <= usable or font == 9:
            break
    # Compress per-column padding (down to 0.06 in) before shrinking columns
    # below content width, so noWrap cells keep room for a single line.
    pad = PAD_IN
    if not any(prose) and sum(content) + PAD_IN * len(content) > usable:
        pad = max(0.06, (usable - sum(content)) / len(content))
    fixed = [c + pad for c in content]
    widths = list(fixed)
    n_prose = sum(prose)
    if n_prose:
        leftover = usable - sum(w for w, p in zip(widths, prose) if not p)
        per = max(PROSE_MIN_IN, leftover / n_prose)
        for j, p in enumerate(prose):
            if p:
                widths[j] = per
    total = sum(widths)
    if total > usable:
        widths = [w * usable / total for w in widths]
    return widths, prose, font


def table_fits_portrait(rows: list[list[str]]) -> bool:
    maxlens = col_maxlens(rows)
    prose = [m > PROSE_COL_CHARS for m in maxlens]
    need = sum(m * CHAR_IN[10] + PAD_IN for m, p in zip(maxlens, prose) if not p)
    need += PROSE_MIN_IN * sum(prose)
    return need <= PORTRAIT_USABLE


def add_table(doc: Document, lines: list[str], usable: float) -> None:
    rows = parse_table(lines)
    if not rows:
        return
    widths, prose, font = plan_widths(rows, usable)
    n_cols = len(widths)
    table = doc.add_table(rows=len(rows), cols=n_cols)
    table.autofit = False
    clear_table_borders(table)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    table._tbl.tblPr.append(layout)
    # Word's default 0.08 in left/right cell margins would eat the padding
    # budget planned above; pin them to 0.03 in so the plan is real.
    cell_mar = OxmlElement("w:tblCellMar")
    for side in ("left", "right"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), "43")
        el.set(qn("w:type"), "dxa")
        cell_mar.append(el)
    table._tbl.tblPr.append(cell_mar)
    for j, col in enumerate(table.columns):
        col.width = Inches(widths[j])
    for i, row in enumerate(rows):
        for j in range(n_cols):
            cell_text = row[j] if j < len(row) else ""
            cell = table.rows[i].cells[j]
            cell.width = Inches(widths[j])
            par = cell.paragraphs[0]
            if not prose[j]:
                cell_text = cell_text.replace(" ", NBSP)
            add_runs(par, cell_text)
            if not prose[j]:
                no_break_hyphens(par)
            for run in par.runs:
                run.font.size = Pt(font)
                if run.font.name != "Courier New":
                    run.font.name = "Times New Roman"
            par.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            par.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            set_cell_border(cell, top=(i == 0), bottom=(i == 0 or i == len(rows) - 1))
            if not prose[j]:
                tc_pr = cell._tc.get_or_add_tcPr()
                if tc_pr.find(qn("w:noWrap")) is None:
                    tc_pr.append(OxmlElement("w:noWrap"))
        # A row with text only in the stub is a section heading (APA spanner),
        # not a data row with missing cells.
        if row and row[0].strip() and all(not c.strip() for c in row[1:]):
            table.rows[i].cells[0].merge(table.rows[i].cells[n_cols - 1])


def split_pages(lines: list[str]) -> list[list[str]]:
    pages, current = [], []
    for line in lines:
        if line.strip() == "<!--pagebreak-->":
            pages.append(current)
            current = []
        else:
            current.append(line)
    pages.append(current)
    return pages


def page_tables(page: list[str]) -> list[list[str]]:
    tables, current = [], []
    for line in page:
        if line.strip().startswith("|"):
            current.append(line)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def page_is_landscape(page: list[str]) -> bool:
    return any(not table_fits_portrait(parse_table(tbl))
               for tbl in page_tables(page) if parse_table(tbl))


FIG_MARKER = re.compile(r"^<!--figure:(.+?)-->$")
FIG_MAX_W = 6.2   # inches; text width minus a little slack
FIG_MAX_H = 7.4   # inches; leaves room for the figure number/title above
                  # and the note below on an 11-in page with 1-in margins
                  # (PEPS editorial office 2026-10-15: Figure S2 was clipped)


def add_figure(doc: Document, name: str) -> None:
    """Insert output/<name> scaled to fit both the text width and the page
    height, preserving aspect ratio (tall forest plots were overflowing)."""
    from PIL import Image
    path = ROOT / "output" / name
    with Image.open(path) as im:
        w_px, h_px = im.size
    width = min(FIG_MAX_W, FIG_MAX_H * w_px / h_px)
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def render_page(doc: Document, page: list[str], usable: float, state: dict) -> None:
    i = 0
    while i < len(page):
        line = page[i]
        m = FIG_MARKER.match(line.strip())
        if m:
            add_figure(doc, m.group(1).strip())
            i += 1
            continue
        if line.strip().startswith("|"):
            tbl = []
            while i < len(page) and page[i].strip().startswith("|"):
                tbl.append(page[i])
                i += 1
            add_table(doc, tbl, usable)
            continue
        stripped = line.strip()
        if not stripped or stripped == "---":
            i += 1
            continue
        if stripped.startswith("#"):
            level = min(len(stripped) - len(stripped.lstrip("#")), 3)
            text = stripped.lstrip("#").strip()
            par = doc.add_heading("", level=level)
            add_runs(par, text)
            # APA section labels (Abstract, References, etc.) are centered
            # regardless of outline level; only true Level 2+ headings are
            # flush left.
            if level == 1 or text in ("Abstract", "Figure Captions"):
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if level == 1:
                state["first_h1_seen"] = True
        elif stripped.startswith(("- ", "* ")):
            par = doc.add_paragraph(style="List Bullet")
            add_runs(par, stripped[2:])
        elif re.match(r"^\d+\.\s", stripped):
            par = doc.add_paragraph(style="List Number")
            add_runs(par, re.sub(r"^\d+\.\s", "", stripped))
        else:
            par = doc.add_paragraph()
            add_runs(par, stripped)
            is_meta = (
                stripped.startswith(("*Note.*", "**Table", "**Figure", "**Panel", "Panel "))
                or not state["first_h1_seen"])
            # APA 7: the keywords line is indented 0.5 in like body prose.
            if stripped.startswith("*Keywords:*"):
                par.paragraph_format.first_line_indent = Inches(0.5)
            elif not is_meta:
                par.paragraph_format.first_line_indent = Inches(0.5)
        i += 1


def build(md_path: Path, out_path: Path, figures: list[tuple[str, str]],
          title_page_md: Path | None = None) -> None:
    doc = Document()
    style_doc(doc)

    pages = split_pages(md_path.read_text(encoding="utf-8").splitlines())
    first_section_used = False
    if title_page_md is not None:
        # Page 1 is the full APA title page; drop the manuscript's bare title page.
        set_orientation(doc.sections[0], False)
        add_running_head(doc, RUNNING_HEAD)
        render_title_page(doc, title_page_md)
        pages = pages[1:]
        first_section_used = True

    state = {"first_h1_seen": False}
    for k, page in enumerate(pages):
        landscape = page_is_landscape(page)
        if k == 0 and not first_section_used:
            sec = doc.sections[0]
            set_orientation(sec, landscape)
            add_running_head(doc, RUNNING_HEAD)
        else:
            sec = doc.add_section(WD_SECTION.NEW_PAGE)
            set_orientation(sec, landscape)
            add_running_head(doc, RUNNING_HEAD, sec)
        usable = LANDSCAPE_USABLE if landscape else PORTRAIT_USABLE
        render_page(doc, page, usable, state)

    if figures:
        sec = doc.add_section(WD_SECTION.NEW_PAGE)
        set_orientation(sec, False)
        add_running_head(doc, RUNNING_HEAD, sec)
        for n, (label, fig) in enumerate(figures):
            if n:
                doc.add_page_break()
            lab = doc.add_paragraph()
            lab.add_run(label).bold = True
            add_figure(doc, fig)

    doc.save(out_path)
    print("Saved", out_path)


if __name__ == "__main__":
    for md, out, figs in DOCS:
        if md.name.startswith("title_page"):
            build_title_page(md, out)
        elif md.name.startswith("manuscript"):
            # Full copy with the unmasked APA title page embedded.
            build(md, out, figs, title_page_md=MS / "title_page_v10.md")
            # Masked review copy for JAP submission: page 1 carries the
            # title only (the md's own H1); no author-identifying content.
            masked = out.with_name(out.stem + "_masked" + out.suffix)
            build(md, masked, figs)
        else:
            build(md, out, figs)

    # Smoke test: every build runs the deterministic fact-check (no API cost).
    import subprocess
    import sys
    rc = subprocess.call([sys.executable, str(ROOT / "code" / "07_fact_check.py")])
    if rc != 0:
        sys.exit("Build completed but fact-check FAILED; see report above.")


