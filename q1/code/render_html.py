"""Render FINDINGS.md to a self-contained HTML report with the figures inline."""
from __future__ import annotations

import base64
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
MD = ROOT / "FINDINGS.md"
OUT = ROOT / "FINDINGS.html"

CSS = """
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 32px 24px 80px; line-height: 1.55;
       color: #1d2330; background: #ffffff; }
h1 { font-size: 30px; border-bottom: 3px solid #228b73; padding-bottom: 8px; }
h2 { font-size: 23px; margin-top: 38px; color: #155f4c; border-bottom: 1px solid #d9e2df; padding-bottom: 4px;}
h3 { font-size: 18px; margin-top: 26px; color: #2b3a4a; }
h4 { font-size: 15px; margin-top: 20px; color: #445; }
table { border-collapse: collapse; margin: 14px 0; font-size: 13.5px; width: 100%; }
th, td { border: 1px solid #d5dcd9; padding: 6px 9px; text-align: left; vertical-align: top; }
th { background: #eef5f2; }
tr:nth-child(even) td { background: #fafcfb; }
code { background: #f1f3f5; padding: 1px 5px; border-radius: 4px; font-size: 13px; }
pre { background: #f7f9fa; border: 1px solid #e3e8ea; border-radius: 6px; padding: 12px 14px;
      overflow-x: auto; font-size: 12.5px; }
pre code { background: none; padding: 0; }
blockquote { border-left: 4px solid #228b73; margin: 12px 0; padding: 4px 14px; color: #334; background: #f6faf9;}
figure { margin: 18px 0; }
figure img { width: 100%; border: 1px solid #e2e6e8; border-radius: 6px; }
figcaption { font-size: 13px; color: #556; margin-top: 6px; }
hr { border: none; border-top: 1px solid #e2e6e8; margin: 30px 0; }
strong { color: #10312a; }
"""


def inline(text):
    """Minimal markdown inline formatting."""
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    return text


def img_b64(path):
    data = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode()


def convert(md: str) -> str:
    out = []
    lines = md.split("\n")
    i = 0
    in_code = False
    code_buf = []
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            if in_code:
                out.append("<pre><code>" + html.escape("\n".join(code_buf)) + "</code></pre>")
                code_buf = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and set(
                lines[i + 1].replace("|", "").replace(" ", "")) <= set("-"):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            out.append("<table><thead><tr>" +
                       "".join(f"<th>{inline(c)}</th>" for c in header) +
                       "</tr></thead><tbody>")
            for r in rows:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            out.append("</tbody></table>")
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            i += 1
            continue
        if line.strip() == "---":
            out.append("<hr>")
            i += 1
            continue
        if re.match(r"^\s*[-*]\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append(re.sub(r"^\s*[-*]\s+", "", lines[i]))
                i += 1
            out.append("<ul>" + "".join(f"<li>{inline(t)}</li>" for t in items) + "</ul>")
            continue
        if re.match(r"^\s*\d+\.\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                items.append(re.sub(r"^\s*\d+\.\s+", "", lines[i]))
                i += 1
            out.append("<ol>" + "".join(f"<li>{inline(t)}</li>" for t in items) + "</ol>")
            continue
        if line.strip():
            out.append(f"<p>{inline(line)}</p>")
        i += 1
    return "\n".join(out)


def inject_figures(body: str) -> str:
    """Insert every figure that exists, in a gallery, before the last section."""
    figs = sorted(FIG.glob("*.png"))
    if not figs:
        return body
    items = []
    for f in figs:
        cap = f.stem.replace("_", " ").replace("A1 ", "A1 - ").replace("B1 ", "B1 - ")
        items.append(
            f"<figure><img src='{img_b64(f)}' alt='{html.escape(cap)}'>"
            f"<figcaption>{html.escape(cap)}</figcaption></figure>")
    gallery = ("<h2 id='figures'>Figure gallery</h2>"
               "<p>Every figure referenced in the notes, in one place.</p>"
               + "".join(items))
    return body.replace("<h2 id=\"reproducing\">", gallery + "<h2 id=\"reproducing\">")


def main():
    md = MD.read_text()
    body = convert(md)
    body = inject_figures(body)
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Q1 River Forecasting - design investigation</title><style>{CSS}</style></head>
<body>{body}</body></html>"""
    OUT.write_text(doc)
    print("wrote", OUT, len(doc) // 1024, "kB")


if __name__ == "__main__":
    main()
