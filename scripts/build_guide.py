"""Build the learning guide PDF:  report/guide/Learning_Guide.md  ->  HTML (print stylesheet)  ->  report/DR_Learning_Guide.pdf

    python scripts/make_guide_figures.py     # diagrams + figures from the real project artifacts (run first, CPU, ~1 min)
    python scripts/build_guide.py            # this script (needs Chrome or Edge; uses its headless print engine)

Source syntax (Markdown plus a few directives, all resolved here so the PDF can be regenerated after any new run):
  {{val:path.to.value|.3f}}         a number from results/metrics.json (or test_summary / simulation); unresolved -> "N/A" + listed in build/missing.txt
  {{fig:file.png|Caption|80|label}}  a numbered figure (width in % of the text column); looked up in report/guide/figures, then report/figures
  {{tab:label|Caption}}              a numbered table caption placed above the following Markdown table
  {{ref:label}}                      "Figure 3" / "Table 2" for a labelled figure or table
  ::: kind Title ... :::             callout box; kind in define, why, how, project, example, warning, takeaway, math, viva
  $...$ and $$...$$                  mathematics, typeset with matplotlib's mathtext (no TeX installation needed)
Headings: '#' = part opener (new page), '##' = numbered chapter, '###' = numbered section. A heading ending in ' {-}' is not numbered.
The table of contents gets real page numbers in a second pass (read back from the PDF's own link targets).
"""
from __future__ import annotations

import base64
import html
import io
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

GUIDE = ROOT / "report" / "guide"
SRC = GUIDE / "Learning_Guide.md"
BUILD = GUIDE / "build"
OUT_PDF = ROOT / "report" / "DR_Learning_Guide.pdf"
FIG_DIRS = [GUIDE / "figures", ROOT / "report" / "figures"]
TITLE = "Diabetic Retinopathy Screening — A Learning Guide to the Project"

BOX = {"define": "Definition", "why": "Why we need it", "how": "How it works", "project": "In our project", "example": "Illustrative example",
       "warning": "Caution", "takeaway": "Key takeaway", "math": "The mathematics", "viva": "Viva question"}


# ----------------------------------------------------------------------------- data placeholders
def load_results() -> dict:
    res = json.loads((ROOT / "results" / "metrics.json").read_text())
    for name, key in (("test_summary.json", "test_summary"), ("simulation.json", "simulation")):
        f = ROOT / "results" / name
        if f.is_file():
            res[key] = json.loads(f.read_text())
    for name, key in (("walkthrough.json", "walkthrough"), ("figure_facts.json", "figfacts")):  # outputs of scripts/make_guide_figures.py
        f = GUIDE / "figures" / name
        if f.is_file():
            res[key] = json.loads(f.read_text())
    return res


def resolve_values(text: str, res: dict, missing: list) -> str:
    from drscreen.report import _get, fmt_value

    def sub(m):
        path, _, fmt = m.group(1).partition("|")
        v = _get(res, path.strip())
        if v is None:
            missing.append(path)
            return "N/A"
        if isinstance(v, str):
            return v
        return fmt_value(v, fmt.strip() or ".3f", "N/A")

    return re.sub(r"\{\{val:([^}]+)\}\}", sub, text)


def find_figure(name: str) -> Path | None:
    for d in FIG_DIRS:
        if (d / name).is_file():
            return d / name
    return None


def number_figures_and_tables(text: str, missing: list) -> str:
    labels, nfig, ntab = {}, 0, 0
    for m in re.finditer(r"\{\{(fig|tab):([^}]+)\}\}", text):
        parts = m.group(2).split("|")
        if m.group(1) == "fig":
            nfig += 1
            if len(parts) >= 4 and parts[3].strip():
                labels[parts[3].strip()] = f"Figure {nfig}"
        else:
            ntab += 1
            labels[parts[0].strip()] = f"Table {ntab}"
    counters = {"fig": 0, "tab": 0}

    def fig_or_tab(m):
        kind, parts = m.group(1), m.group(2).split("|")
        counters[kind] += 1
        if kind == "tab":
            return f'<div class="tcap"><b>Table {counters["tab"]}.</b> {parts[1].strip() if len(parts) > 1 else ""}</div>'
        name, cap = parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "")
        width = parts[2].strip() if len(parts) > 2 and parts[2].strip() else "100"
        p = find_figure(name)
        if p is None:
            missing.append("figure " + name)
            return f'<div class="figmissing">Figure {counters["fig"]} not available: {html.escape(name)} was not produced by the current artifacts.</div>'
        return (f'<figure style="width:{width}%"><img src="{p.resolve().as_uri()}" alt="{html.escape(cap)}">'
                f'<figcaption><b>Figure {counters["fig"]}.</b> {cap}</figcaption></figure>')

    text = re.sub(r"\{\{(fig|tab):([^}]+)\}\}", fig_or_tab, text)

    def ref(m):
        lab = m.group(1).strip()
        if lab not in labels:
            missing.append("ref " + lab)
            return "(missing reference)"
        return labels[lab]

    return re.sub(r"\{\{ref:([^}]+)\}\}", ref, text)


# ----------------------------------------------------------------------------- mathematics
_MATH_CACHE: dict = {}


def math_svg(tex: str, display: bool) -> str:
    tex = " ".join(tex.split())
    key = (tex, display)
    if key in _MATH_CACHE:
        return _MATH_CACHE[key]
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import rcParams
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import math_to_image

    rcParams["mathtext.fontset"] = "stix"
    rcParams["svg.fonttype"] = "path"
    buf = io.BytesIO()
    try:
        math_to_image(f"${tex}$", buf, prop=FontProperties(size=17 if display else 12.5), dpi=72, format="svg")
    except Exception as e:  # show the source rather than failing the whole build
        print(f"[math] could not typeset {tex!r}: {e}")
        out = f'<code class="mathsrc">{html.escape(tex)}</code>'
        _MATH_CACHE[key] = out
        return out
    svg = buf.getvalue().decode("utf-8")
    svg = svg[svg.find("<svg"):]
    svg = re.sub(r"<metadata>.*?</metadata>", "", svg, flags=re.S)
    m = re.search(r'width="([\d.]+)pt" height="([\d.]+)pt"', svg)
    if m:  # scale points to CSS so text-size maths matches the body
        w, h = float(m.group(1)), float(m.group(2))
        svg = svg.replace(m.group(0), f'width="{w * 0.96:.1f}pt" height="{h * 0.96:.1f}pt"', 1)
    uid = f"m{len(_MATH_CACHE)}"
    svg = re.sub(r'id="([^"]+)"', lambda mm: f'id="{uid}{mm.group(1)}"', svg)
    svg = re.sub(r'(href|xlink:href)="#([^"]+)"', lambda mm: f'{mm.group(1)}="#{uid}{mm.group(2)}"', svg)
    out = (f'<div class="math-block">{svg}</div>' if display else f'<span class="math-inline">{svg}</span>')
    _MATH_CACHE[key] = out
    return out


# ----------------------------------------------------------------------------- markdown -> html
def markdown_to_html(text: str):
    from markdown_it import MarkdownIt
    from mdit_py_plugins.container import container_plugin
    from mdit_py_plugins.dollarmath import dollarmath_plugin

    md = MarkdownIt("commonmark", {"html": True, "typographer": True}).enable(["table", "strikethrough", "replacements", "smartquotes"])
    md.use(dollarmath_plugin, allow_space=True, double_inline=False)

    def render_box(kind):
        def render(self, tokens, idx, options, env):
            tok = tokens[idx]
            if tok.nesting == 1:
                title = tok.info.strip()[len(kind):].strip()
                label = BOX[kind]
                head = f'<span class="box-kind">{label}</span>' + (f'<span class="box-title">{md.renderInline(title)}</span>' if title else "")
                return f'<aside class="box box-{kind}"><div class="box-head">{head}</div>\n'
            return "</aside>\n"
        return render

    for k in BOX:
        md.use(container_plugin, name=k, render=render_box(k))
    md.add_render_rule("math_inline", lambda self, tokens, idx, options, env: math_svg(tokens[idx].content, False))
    md.add_render_rule("math_block", lambda self, tokens, idx, options, env: math_svg(tokens[idx].content.strip(), True))
    md.add_render_rule("math_block_label", lambda self, tokens, idx, options, env: math_svg(tokens[idx].content.strip(), True))

    text = re.sub(r"(?m)^[ \t]*\$\$(.+?)\$\$[ \t]*$", lambda m: "\n$$\n" + m.group(1).strip() + "\n$$\n", text)  # one-line display maths
    tokens = md.parse(text)
    toc, nch, nsec, npart = [], 0, 0, 0
    romans = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    for i, tok in enumerate(tokens):
        if tok.type != "heading_open":
            continue
        inline = tokens[i + 1]
        raw = inline.content
        numbered = not raw.endswith("{-}")
        title = raw[:-3].strip() if not numbered else raw
        level = int(tok.tag[1])
        hid = f"h{len(toc)}"
        num = ""
        if level == 1:
            npart += 1 if numbered else 0
            num = f"Part {romans[npart - 1]}" if numbered else ""
            tok.attrSet("class", "part")
        elif level == 2:
            if numbered:
                nch += 1
                nsec = 0
                num = str(nch)
        elif level == 3 and numbered and nch:
            nsec += 1
            num = f"{nch}.{nsec}"
        tok.attrSet("id", hid)
        inline.content = title
        inline.children = md.parseInline(title)[0].children
        if num:
            prefix = f'<span class="hnum">{num}</span>'
            from markdown_it.token import Token

            t = Token("html_inline", "", 0)
            t.content = prefix
            inline.children.insert(0, t)
        toc.append({"level": level, "id": hid, "num": num, "title": title})
    body = md.renderer.render(tokens, md.options, {})
    return body, toc


# ----------------------------------------------------------------------------- page assembly
CSS = (GUIDE / "guide.css")


def toc_html(toc: list, pages: dict | None) -> str:
    rows = []
    for e in toc:
        if e["level"] > 2:
            continue
        pg = str(pages.get(e["id"], "")) if pages else "000"
        cls = "toc-part" if e["level"] == 1 else "toc-ch"
        num = f'<span class="toc-num">{html.escape(e["num"])}</span>' if e["num"] else '<span class="toc-num"></span>'
        title = re.sub(r"\*\*|`", "", e["title"])
        rows.append(f'<a class="toc-row {cls}" href="#{e["id"]}">{num}<span class="toc-title">{html.escape(title)}</span><span class="toc-dots"></span><span class="toc-pg">{pg}</span></a>')
    return '<nav class="toc"><h2 class="toc-head">Contents</h2>' + "".join(rows) + "</nav>"


def page_html(body: str, toc: list, pages: dict | None, cover: bool = False) -> str:
    css = CSS.read_text(encoding="utf-8")
    if cover:
        return f'<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body class="cover-body">{body}</body></html>'
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(TITLE)}</title><style>{css}</style></head>'
            f'<body>{toc_html(toc, pages)}<main>{body}</main></body></html>')


# ----------------------------------------------------------------------------- printing (headless Chrome over DevTools)
def print_pdf(html_path: Path, pdf_path: Path, header_footer: bool) -> None:
    from ui_screenshots import CDP, CHROMES

    chrome = next((c for c in CHROMES if Path(c).is_file()), None)
    if chrome is None:
        raise SystemExit("Chrome or Edge is required to print the guide")
    port = 9400 + int(time.time()) % 200
    proc = subprocess.Popen([chrome, "--headless=new", "--disable-gpu", f"--remote-debugging-port={port}", "--remote-allow-origins=*",
                             f"--user-data-dir={tempfile.mkdtemp(prefix='drguide_')}", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                c = CDP(port)
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise SystemExit("could not attach to Chrome")
        c.call("Page.enable")
        c.call("Page.navigate", url=html_path.resolve().as_uri())
        for _ in range(120):
            if c.js("document.readyState === 'complete' && [...document.images].every(i => i.complete)"):
                break
            time.sleep(0.5)
        time.sleep(0.5)
        foot = ('<div style="font-family:Segoe UI,Arial,sans-serif;font-size:7.5px;color:#6b726e;width:100%;padding:0 18mm;display:flex;justify-content:space-between">'
                '<span>Diabetic Retinopathy Screening · Learning Guide</span><span><span class="pageNumber"></span></span></div>')
        r = c.call("Page.printToPDF", landscape=False, printBackground=True, paperWidth=8.27, paperHeight=11.69,
                   marginTop=0.0 if not header_footer else 0.75, marginBottom=0.0 if not header_footer else 0.7,
                   marginLeft=0.0 if not header_footer else 0.79, marginRight=0.0 if not header_footer else 0.79,
                   displayHeaderFooter=header_footer, headerTemplate="<span></span>", footerTemplate=foot, preferCSSPageSize=False)
        pdf_path.write_bytes(base64.b64decode(r["data"]))
    finally:
        proc.terminate()


def toc_pages(pdf_path: Path, toc: list) -> dict:
    """Read each TOC link's destination page back from the printed PDF."""
    import pymupdf

    doc = pymupdf.open(pdf_path)
    ids = [e["id"] for e in toc if e["level"] <= 2]
    pages: dict = {}
    links = []
    for pno in range(min(6, len(doc))):
        for ln in doc[pno].get_links():
            if ln.get("kind") == pymupdf.LINK_GOTO or "page" in ln:
                links.append((pno, ln["from"].y0, ln.get("page", -1)))
    links.sort()
    for hid, (_, _, target) in zip(ids, links):
        if target is not None and target >= 0:
            pages[hid] = target + 1
    return pages


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    res = load_results()
    missing: list = []
    src = SRC.read_text(encoding="utf-8")
    src = re.sub(r"\{\{include:([^}]+)\}\}", lambda m: (GUIDE / m.group(1).strip()).read_text(encoding="utf-8"), src)  # parts/*.md
    cover_src, _, body_src = src.partition("<!-- END COVER -->")
    body_src = number_figures_and_tables(resolve_values(body_src, res, missing), missing)
    cover_src = resolve_values(cover_src, res, missing)
    body, toc = markdown_to_html(body_src)

    cover_html = BUILD / "cover.html"
    cover_html.write_text(page_html(cover_src, [], None, cover=True), encoding="utf-8")
    print_pdf(cover_html, BUILD / "cover.pdf", header_footer=False)

    body_html = BUILD / "guide.html"
    pages = None
    for attempt in range(3):  # pass 1 measures page numbers, pass 2 prints them (a third pass only if the TOC length changed)
        body_html.write_text(page_html(body, toc, pages), encoding="utf-8")
        print_pdf(body_html, BUILD / "body.pdf", header_footer=True)
        new = toc_pages(BUILD / "body.pdf", toc)
        if pages == new:
            break
        pages = new
    import pymupdf

    out = pymupdf.open(BUILD / "cover.pdf")
    body_doc = pymupdf.open(BUILD / "body.pdf")
    out.insert_pdf(body_doc, links=False)
    for pno in range(len(body_doc)):  # re-create the internal links (table of contents) with the one-page cover offset
        for ln in body_doc[pno].get_links():
            if ln.get("page", -1) is not None and ln.get("page", -1) >= 0:
                out[pno + 1].insert_link({"kind": pymupdf.LINK_GOTO, "from": ln["from"], "page": ln["page"] + 1, "to": ln.get("to", pymupdf.Point(0, 0))})
    out.set_metadata({"title": TITLE, "subject": "Learning guide and research report for the DR screening research prototype", "creator": "scripts/build_guide.py"})
    out.save(OUT_PDF, garbage=3, deflate=True)
    (BUILD / "missing.txt").write_text("\n".join(sorted(set(missing))) + ("\n" if missing else ""))
    print(json.dumps({"pdf": str(OUT_PDF.relative_to(ROOT)), "pages": len(out), "toc_entries": len([e for e in toc if e["level"] <= 2]),
                      "figures": len(re.findall(r"<figure", body)), "missing": sorted(set(missing))}, indent=1))


if __name__ == "__main__":
    main()
