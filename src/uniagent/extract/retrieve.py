"""Chunk a page and pick the passages relevant to the fields being extracted.

v1 sent `page_text[:4000]` to the model, so any fact below the fold of a long fees page was
invisible. Here the cleaned Markdown is split along headings into ~1.2k-char chunks (tables are
kept intact so a fee row never loses its column headers), each field's keywords are scored
against every chunk with BM25, and the best chunks are packed into a fixed character budget,
in document order so the model still reads them as a coherent page.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from uniagent.schema import FieldSpec

_TOKEN = re.compile(r"[a-z0-9]+")
_HEADING = re.compile(r"^#{1,6}\s")


@dataclass
class Chunk:
    idx: int
    heading: str
    text: str


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def chunk_markdown(md: str, target: int = 1200) -> list[Chunk]:
    blocks: list[tuple[str, str]] = []  # (heading, block)
    heading = ""
    buf: list[str] = []

    def flush():
        if buf:
            blocks.append((heading, "\n".join(buf).strip()))
            buf.clear()

    in_table = False
    for line in md.splitlines():
        is_table = line.lstrip().startswith("|")
        if _HEADING.match(line):
            flush()
            heading = line.lstrip("#").strip()
            buf.append(line)
            continue
        if is_table != in_table:
            flush()
            if heading and not is_table:
                pass
            in_table = is_table
        if not line.strip() and not in_table:
            flush()
            continue
        buf.append(line)
    flush()

    chunks: list[Chunk] = []
    cur, cur_head = "", ""
    for head, block in blocks:
        if not block:
            continue
        if cur and (len(cur) + len(block) > target or head != cur_head):
            chunks.append(Chunk(len(chunks), cur_head, cur.strip()))
            cur = ""
        if not cur:
            cur_head = head
            if head and not block.startswith("#"):
                cur = f"## {head}\n"
        cur += block + "\n\n"
        if len(cur) > target * 2:  # huge single block (e.g. long table): cut on lines
            lines, part = cur.splitlines(), ""
            for ln in lines:
                if len(part) + len(ln) > target and part:
                    chunks.append(Chunk(len(chunks), cur_head, part.strip()))
                    part = (f"## {cur_head}\n" if cur_head else "") + (
                        lines[0] + "\n" if lines[0].startswith("|") else ""
                    )
                part += ln + "\n"
            cur = part
    if cur.strip():
        chunks.append(Chunk(len(chunks), cur_head, cur.strip()))
    return chunks


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, docs)) / max(1, len(docs))
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, d in zip(self.tf, self.docs, strict=True):
            s = 0.0
            for t in query:
                if t not in tf:
                    continue
                f = tf[t]
                s += (
                    self.idf.get(t, 0)
                    * f
                    * (self.k1 + 1)
                    / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
                )
            out.append(s)
        return out


def select_context(md: str, fields: list[FieldSpec], budget: int = 7000, per_field: int = 3) -> str:
    """Return the most relevant chunks for `fields`, packed into `budget` characters."""
    if len(md) <= budget:
        return md
    chunks = chunk_markdown(md)
    if not chunks:
        return md[:budget]
    bm25 = BM25([_tokens(c.heading + " " + c.text) for c in chunks])

    ranked_lists = []
    for f in fields:
        query = _tokens(" ".join(f.keywords) + " " + f.name.replace("_", " "))
        scores = bm25.scores(query)
        ranked_lists.append(sorted(range(len(chunks)), key=lambda i: -scores[i]))

    # Round-robin over fields so every field gets its best chunk before any gets its second.
    chosen: list[int] = []
    used = 0
    for rank in range(per_field):
        for ranked in ranked_lists:
            if rank >= len(ranked):
                continue
            i = ranked[rank]
            if i in chosen:
                continue
            size = len(chunks[i].text) + 2
            if used + size > budget:
                continue
            chosen.append(i)
            used += size
    return "\n\n".join(chunks[i].text for i in sorted(chosen))


def truncate_context(md: str, budget: int = 4000) -> str:
    """The v1 behaviour, kept as an ablation baseline."""
    return md[:budget]
