"""Render the Markdown experiment report to a print-ready HTML and PDF.

Uses the Python `markdown` package for conversion and headless Microsoft Edge
for the HTML -> PDF step, because Microsoft Word is not installed on this host.
"""

import re
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(r"D:\zijie")
BUILD = ROOT / "submission_build"
SRC = ROOT / "实验汇报总结_ByteFormer_MNIST.md"
HTML_PATH = BUILD / "_report.html"
PDF_PATH = BUILD / "report" / "实验报告.pdf"

EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Microsoft YaHei", "Segoe UI", sans-serif; font-size: 10.5pt;
       line-height: 1.75; color: #1f1f1f; margin: 0; }
h1 { font-size: 20pt; color: #1f4e79; text-align: center; margin: 0 0 4pt 0;
     border-bottom: 2.5pt solid #1f4e79; padding-bottom: 6pt; }
h2 { font-size: 14pt; color: #1f4e79; margin: 18pt 0 6pt 0;
     border-left: 5pt solid #2e75b6; padding-left: 8pt; }
h3 { font-size: 12pt; color: #2e75b6; margin: 13pt 0 5pt 0; }
p { margin: 5pt 0; text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 10pt; }
th { background: #1f4e79; color: #fff; font-weight: bold; padding: 5pt 7pt;
     border: 0.75pt solid #b8c4d4; text-align: center; }
td { padding: 5pt 7pt; border: 0.75pt solid #b8c4d4; }
tr:nth-child(even) td { background: #f4f7fb; }
code { font-family: Consolas, monospace; font-size: 9.5pt;
       background: #f2f2f2; padding: 1pt 3pt; border-radius: 2pt; }
pre { background: #f6f8fa; border: 0.75pt solid #d8dee6; border-radius: 3pt;
      padding: 7pt 9pt; font-size: 9pt; line-height: 1.5; overflow-x: auto; }
pre code { background: none; padding: 0; }
blockquote { margin: 7pt 0; padding: 6pt 10pt; background: #fdf6e3;
             border-left: 4pt solid #e0a800; color: #4a4a4a; font-size: 10pt; }
ul, ol { margin: 5pt 0 5pt 0; padding-left: 20pt; }
li { margin: 2.5pt 0; }
strong { color: #1f4e79; }
"""


def main() -> int:
    if not SRC.exists():
        print(f"[FAIL] source report not found: {SRC}")
        return 1

    text = SRC.read_text(encoding="utf-8")
    # Drop the stale "结果文件" section: it points at paths that no longer exist.
    text = re.sub(r"\n## 十一、结果文件\n.*\Z", "\n", text, flags=re.S)

    html_body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "nl2br"]
    )
    HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    HTML_PATH.write_text(
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<title>码流图像分类实验报告</title><style>{CSS}</style></head>"
        f"<body>{html_body}</body></html>",
        encoding="utf-8",
    )
    print(f"[HTML] {HTML_PATH}")

    edge = next((p for p in EDGE_CANDIDATES if p.exists()), None)
    if edge is None:
        print("[WARN] Microsoft Edge not found; PDF not generated")
        return 0

    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(edge), "--headless=new", "--disable-gpu", "--no-first-run",
        "--no-pdf-header-footer", f"--print-to-pdf={PDF_PATH}",
        HTML_PATH.as_uri(),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if PDF_PATH.exists():
        print(f"[PDF ] {PDF_PATH} ({PDF_PATH.stat().st_size} bytes)")
    else:
        print(f"[FAIL] PDF missing; rc={proc.returncode}")
        print(proc.stderr[-800:])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
