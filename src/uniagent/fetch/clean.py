"""HTML -> Markdown, tuned for recall.

Article extractors (trafilatura, readability) decide what is "main content", and on the
card/grid layouts universities use for facts pages they guess wrong: on facts.mit.edu
trafilatura kept 11.7k characters and dropped all of: founding year, enrollment and admit rate.
So we only remove what is never content (scripts, nav, footer, forms, hidden elements), convert
the rest of <body> to Markdown with tables intact, and leave precision to the per-field
retrieval step (extract/retrieve.py), which only sends the relevant chunks to the model.

Elements marked `hidden` / `aria-hidden="true"` are deliberately KEPT: sites use those
attributes for collapsed accordions and tabs, and that is exactly where facts like
"International applications close 30 November" live (UQ, McGill). A browser would show them
after one click; so should we.
"""

from __future__ import annotations

import re

from lxml import html as lxml_html
from markdownify import markdownify

_DROP_XPATH = "|".join(
    [
        f"//{t}"
        for t in (
            "script",
            "style",
            "noscript",
            "svg",
            "nav",
            "footer",
            "form",
            "iframe",
            "button",
            "template",
            "select",
        )
    ]
    + [
        "//*[@role='navigation']",
        "//*[contains(@class,'cookie')]",
        "//*[contains(@id,'cookie')]",
        "//*[contains(@class,'skip-link')]",
    ]
)


def html_to_markdown(html: str, url: str | None = None) -> str:
    if not html or not html.strip():
        return ""
    try:
        doc = lxml_html.fromstring(html)
    except (ValueError, lxml_html.etree.ParserError):
        return ""
    body = doc.find("body")
    root = body if body is not None else doc
    total = len(root.text_content()) or 1
    for node in doc.xpath(_DROP_XPATH):
        if node.getparent() is None:
            continue
        # Malformed markup can make the parser nest the whole page inside a <nav> (seen on
        # Edinburgh's Drupal theme). Never drop a "boilerplate" element that holds the main
        # content or most of the page's text.
        if node.xpath(".//main|.//article|.//*[@role='main']"):
            continue
        if len(node.text_content()) > 0.5 * total:
            continue
        node.drop_tree()
    md = markdownify(
        lxml_html.tostring(root, encoding="unicode"),
        heading_style="ATX",
        strip=["a", "img"],
        bullets="-",
    )
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()
