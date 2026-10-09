#!/usr/bin/env python3
"""
Convert MASTER_THESIS_TECHNICAL_REPORT.md to a publication-grade, print-ready HTML document
and automatically compile it to a PDF via Microsoft Edge.
"""

import os
import re
import subprocess
from markdown_it import MarkdownIt

def build_report():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    md_path = os.path.join(base_dir, "reports", "MASTER_THESIS_TECHNICAL_REPORT.md")
    html_path = os.path.join(base_dir, "reports", "MASTER_THESIS_TECHNICAL_REPORT.html")
    pdf_path = os.path.join(base_dir, "reports", "MASTER_THESIS_TECHNICAL_REPORT.pdf")

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Initialize MarkdownIt with table support
    md = MarkdownIt("gfm-like", {"linkify": False, "html": True})
    content_html = md.render(md_text)

    # Convert mermaid codeblocks into clean diagram boxes
    content_html = re.sub(
        r'<pre><code class="language-mermaid">(.*?)</code></pre>',
        r'<div class="mermaid-diagram"><pre class="mermaid-code"><code>\1</code></pre></div>',
        content_html,
        flags=re.DOTALL
    )

    # Wrap tables in clean table containers
    content_html = content_html.replace('<table>', '<div class="table-wrap"><table>')
    content_html = content_html.replace('</table>', '</table></div>')

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Master Technical Thesis Report - Multimodal Deepfake Forensics</title>
    
    <!-- MathJax 3 for mathematical equation rendering -->
    <script>
    window.MathJax = {{
        tex: {{
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
            processEscapes: true
        }},
        options: {{
            skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
        }}
    }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

    <style>
        /* Modern Publication-Grade Stylesheet */
        @page {{
            size: A4;
            margin: 18mm 14mm 18mm 14mm;
        }}

        *, *::before, *::after {{
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 10pt;
            line-height: 1.55;
            color: #1e293b;
            background-color: #ffffff;
            margin: 0;
            padding: 24px 32px;
            max-width: 960px;
            margin-left: auto;
            margin-right: auto;
        }}

        /* Document Hierarchy */
        h1 {{
            font-size: 18pt;
            font-weight: 700;
            color: #0f2d59;
            line-height: 1.25;
            border-bottom: 2.5px solid #0f2d59;
            padding-bottom: 8px;
            margin-top: 24px;
            margin-bottom: 12px;
            page-break-after: avoid;
        }}

        h2 {{
            font-size: 13.5pt;
            font-weight: 600;
            color: #1e3a8a;
            border-bottom: 1px solid #cbd5e1;
            padding-bottom: 5px;
            margin-top: 22px;
            margin-bottom: 10px;
            page-break-after: avoid;
        }}

        h3 {{
            font-size: 11.5pt;
            font-weight: 600;
            color: #1e293b;
            margin-top: 16px;
            margin-bottom: 6px;
            page-break-after: avoid;
        }}

        h4 {{
            font-size: 10.5pt;
            font-weight: 600;
            color: #334155;
            margin-top: 14px;
            margin-bottom: 4px;
            page-break-after: avoid;
        }}

        p, li {{
            margin-top: 4px;
            margin-bottom: 7px;
            text-align: justify;
        }}

        ul, ol {{
            padding-left: 22px;
            margin-top: 4px;
            margin-bottom: 8px;
        }}

        li {{
            margin-bottom: 3px;
        }}

        hr {{
            border: 0;
            border-top: 1px solid #e2e8f0;
            margin: 20px 0;
        }}

        /* Academic Clean Tables */
        .table-wrap {{
            width: 100%;
            margin: 12px 0 16px 0;
            overflow: visible;
            page-break-inside: avoid;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 8.5pt;
            line-height: 1.4;
            text-align: left;
            border: 1px solid #94a3b8;
            page-break-inside: avoid;
        }}

        th {{
            background-color: #f1f5f9;
            color: #0f2d59;
            font-weight: 600;
            padding: 6px 8px;
            border: 1px solid #cbd5e1;
            vertical-align: middle;
        }}

        td {{
            padding: 5px 8px;
            border: 1px solid #e2e8f0;
            vertical-align: top;
        }}

        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}

        /* Code Blocks */
        pre {{
            background-color: #f8fafc;
            color: #0f172a;
            border: 1px solid #cbd5e1;
            border-radius: 4px;
            padding: 10px 12px;
            font-size: 8.5pt;
            line-height: 1.4;
            overflow-x: auto;
            page-break-inside: avoid;
            margin: 10px 0;
        }}

        code {{
            font-family: "Cascadia Code", "Consolas", "Courier New", monospace;
            font-size: 8.8pt;
            background-color: #f1f5f9;
            color: #0f172a;
            padding: 1px 4px;
            border-radius: 3px;
            border: 1px solid #e2e8f0;
        }}

        pre code {{
            background-color: transparent;
            border: none;
            padding: 0;
            color: inherit;
        }}

        /* Blockquotes and Callouts */
        blockquote {{
            border-left: 3.5px solid #2563eb;
            background-color: #f8fafc;
            margin: 10px 0;
            padding: 8px 14px;
            border-radius: 0 4px 4px 0;
            page-break-inside: avoid;
            font-style: italic;
            color: #334155;
        }}

        blockquote p {{
            margin: 0;
        }}

        /* Diagrams */
        .mermaid-diagram {{
            background: #f8fafc;
            border: 1px dashed #94a3b8;
            border-radius: 4px;
            padding: 8px 12px;
            margin: 10px 0;
            page-break-inside: avoid;
        }}

        .mermaid-diagram pre {{
            background: transparent;
            border: none;
            padding: 0;
            margin: 0;
            font-size: 8pt;
            color: #334155;
        }}

        /* Print Media Override */
        @media print {{
            body {{
                max-width: 100% !important;
                padding: 0 !important;
                font-size: 9.5pt;
                color: #000000;
            }}

            h1 {{
                font-size: 16pt;
                margin-top: 14px;
                margin-bottom: 8px;
                color: #000000;
                border-bottom: 2px solid #000000;
                page-break-after: avoid;
            }}

            h2 {{
                font-size: 12.5pt;
                margin-top: 14px;
                margin-bottom: 6px;
                color: #000000;
                border-bottom: 1px solid #666666;
                page-break-after: avoid;
            }}

            h3 {{
                font-size: 11pt;
                margin-top: 10px;
                margin-bottom: 4px;
                page-break-after: avoid;
            }}

            table, tr, td, th {{
                page-break-inside: avoid !important;
            }}

            .table-wrap {{
                page-break-inside: avoid !important;
                overflow: visible !important;
            }}

            pre, blockquote, .mermaid-diagram {{
                page-break-inside: avoid !important;
            }}

            a {{
                text-decoration: none;
                color: #000000;
            }}
        }}
    </style>
</head>
<body>
    {content_html}
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"Generated HTML: {html_path} ({os.path.getsize(html_path):,} bytes)")

    # Compile to PDF using Edge Headless
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if os.path.exists(edge_path):
        print("Compiling publication PDF via Microsoft Edge...")
        cmd = [
            edge_path,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-first-run",
            "--no-default-browser-check",
            f"--print-to-pdf={pdf_path}",
            html_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if os.path.exists(pdf_path):
            print(f"Successfully compiled PDF: {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")
        else:
            print("PDF generation returned, but file not found.")

if __name__ == "__main__":
    build_report()
