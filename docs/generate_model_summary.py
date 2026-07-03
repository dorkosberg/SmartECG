#!/usr/bin/env python3
"""Build print-ready HTML and PDF from SmartECG_Model_Summary.md"""

import os
import sys

import markdown
from xhtml2pdf import pisa

ROOT = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(ROOT, 'SmartECG_Model_Summary.md')
HTML_PATH = os.path.join(ROOT, 'SmartECG_Model_Summary.html')
PDF_PATH = os.path.join(ROOT, 'SmartECG_Model_Summary.pdf')

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
  <meta charset="UTF-8">
  <title>SmartECG — סיכום מודל מפורט</title>
  <style>
    @page {{ size: A4; margin: 2cm; }}
    body {{
      font-family: Arial, Helvetica, sans-serif;
      color: #222;
      line-height: 1.65;
      font-size: 11pt;
      direction: rtl;
      text-align: right;
    }}
    h1 {{ color: #c0392b; font-size: 22pt; border-bottom: 2px solid #e8b4b0; padding-bottom: 8px; }}
    h2 {{ color: #1a5276; font-size: 15pt; margin-top: 24px; page-break-after: avoid; }}
    h3 {{ color: #2980b9; font-size: 12pt; margin-top: 16px; }}
    table {{ width: 95%; border-collapse: collapse; margin: 12px 0; font-size: 9pt; table-layout: fixed; }}
    th {{ background: #1a5276; color: white; padding: 6px; text-align: right; word-wrap: break-word; }}
    td {{ border: 1px solid #ddd; padding: 5px 6px; word-wrap: break-word; }}
    tr:nth-child(even) td {{ background: #f8f9fa; }}
    code, pre {{
      font-family: Menlo, Consolas, monospace;
      background: #f4f4f4;
      direction: ltr;
      text-align: left;
    }}
    pre {{ padding: 10px; border-radius: 6px; font-size: 9pt; white-space: pre-wrap; }}
    code {{ padding: 2px 5px; border-radius: 3px; }}
    hr {{ border: none; border-top: 1px solid #ddd; margin: 20px 0; }}
    ul, ol {{ padding-right: 22px; }}
    .cover {{
      text-align: center;
      padding: 60px 0 40px;
      page-break-after: always;
    }}
    .cover h1 {{ border: none; font-size: 28pt; }}
    .cover p {{ color: #555; font-size: 12pt; }}
    blockquote {{ border-right: 4px solid #c0392b; margin: 0; padding: 8px 16px; background: #fdf2f0; }}
    @media print {{
      body {{ font-size: 10pt; }}
      h2 {{ page-break-before: auto; }}
      pre, table {{ page-break-inside: avoid; }}
    }}
  </style>
</head>
<body>
  <div class="cover">
    <h1>SmartECG</h1>
    <p>סיכום מפורט — מודל, עיבוד מקדים, נתונים וחלוקה</p>
    <p>1D CNN · MIT-BIH · Normal / Abnormal</p>
    <p>Afeka · Machine Learning · יוני 2026</p>
  </div>
  {body}
</body>
</html>
"""


def build_html():
    with open(MD_PATH, 'r', encoding='utf-8') as md_file:
        md_text = md_file.read()

    body = markdown.markdown(
        md_text,
        extensions=['tables', 'fenced_code', 'nl2br'],
    )
    html = HTML_TEMPLATE.format(body=body)

    with open(HTML_PATH, 'w', encoding='utf-8') as html_file:
        html_file.write(html)

    return html


def build_pdf(html):
    with open(PDF_PATH, 'wb') as pdf_file:
        status = pisa.CreatePDF(html, dest=pdf_file, encoding='utf-8')
    return not status.err


def main():
    if not os.path.exists(MD_PATH):
        print(f'MD not found: {MD_PATH}')
        sys.exit(1)

    print('Building HTML...')
    html = build_html()
    print(f'HTML saved: {HTML_PATH}')

    print('Building PDF...')
    if build_pdf(html):
        print(f'PDF saved: {PDF_PATH}')
    else:
        print('PDF generation failed. Open the HTML in a browser and use Print → Save as PDF.')
        sys.exit(1)


if __name__ == '__main__':
    main()
