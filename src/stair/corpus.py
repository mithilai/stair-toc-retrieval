"""Turn a book into a STAIR corpus: leaf sections with their text.

One book is one corpus. The paper trains and evaluates per book — the 14k
input budget exists so a single book's entire ToC fits in the prompt — so
nothing here ever mixes books together.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from .toc import LeafSection, extract_toc, section_spans

# Page furniture that repeats across a book and carries no section signal.
_NOISE = re.compile(r"^\s*(\d+|[ivxlc]+|chapter \d+|page \d+ of \d+)\s*$", re.I)

# Front/back matter parses as leaf sections with real page spans, but no
# generated query will ever legitimately retrieve "Copyright". Left in, these
# inflate the docid set and quietly distort every metric (H10).
_MATTER = re.compile(
    r"^\s*("
    r"cover|copyright|title page|half title|colophon|dedication|"
    r"table of contents|contents|brief contents|toc|"
    r"about the author(s)?|about this book|about the cover|"
    r"acknowledg(e)?ments?|praise for|front ?matter|back ?matter|"
    r"index|bibliography|glossary|references|credits|"
    r"foreword|preface|epigraph|errata|"
    r"list of (figures|tables|illustrations)"
    r")\s*$",
    re.I,
)


def is_front_matter(title: str) -> bool:
    return bool(_MATTER.match(title.strip()))


def _clean(text: str) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    kept = [ln for ln in lines if not _NOISE.match(ln)]
    out = "\n".join(kept)
    out = re.sub(r"-\n(?=[a-z])", "", out)      # rejoin hyphenated line breaks
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def disambiguate(sections: list[LeafSection]) -> tuple[list[LeafSection], int]:
    """Make docids unique (H9).

    STAIR's identifier is the bare leaf title, but titles repeat — "Summary"
    appears once per chapter. A repeated title cannot identify a section, so
    exact-match scoring and the constrained-decoding trie both break on it.

    We keep the paper's bare title wherever it is already unique and fall back
    to the full hierarchical path only for collisions. That is the smallest
    deviation that makes the identifier actually identify. Returns the count
    so it can be reported rather than hidden.
    """
    counts: dict[str, int] = {}
    for s in sections:
        counts[s.docid] = counts.get(s.docid, 0) + 1

    n_fixed = 0
    for s in sections:
        if counts[s.docid] > 1:
            s.docid = s.path_docid
            n_fixed += 1

    # A path can still collide if the ToC itself repeats a full path.
    seen: dict[str, int] = {}
    for s in sections:
        if s.docid in seen:
            seen[s.docid] += 1
            s.docid = f"{s.docid} [{seen[s.docid]}]"
        else:
            seen[s.docid] = 1
    return sections, n_fixed


def build_corpus(pdf_path: str | Path, book_id: str | None = None,
                 drop_front_matter: bool = True,
                 unique_docids: bool = True,
                 min_chars: int = 200) -> list[LeafSection]:
    """Extract every leaf section of a book, with text."""
    import pymupdf

    pdf_path = Path(pdf_path)
    book_id = book_id or pdf_path.stem[:60]

    roots = extract_toc(str(pdf_path))
    if not roots:
        return []

    with pymupdf.open(str(pdf_path)) as doc:
        n_pages = doc.page_count
        sections: list[LeafSection] = []
        for node, start, end in section_spans(roots, n_pages):
            # pymupdf ToC pages are 1-based; page indices are 0-based.
            lo = max(0, start - 1)
            hi = min(n_pages, max(lo + 1, end - 1))
            text = "\n".join(doc[i].get_text() for i in range(lo, hi))
            sections.append(
                LeafSection(
                    book_id=book_id,
                    docid=node.title,
                    path=node.path,
                    page_start=start,
                    page_end=end,
                    text=_clean(text),
                )
            )

    if drop_front_matter:
        sections = [
            s for s in sections
            if not any(is_front_matter(t) for t in s.path)
        ]
    if min_chars:
        sections = [s for s in sections if s.n_chars >= min_chars]
    if unique_docids:
        sections, _ = disambiguate(sections)
    return sections


def toc_prompt(sections: list[LeafSection]) -> str:
    """Render the book's ToC as the global structure shown to the model.

    This string is STAIR's actual contribution — the model sees the whole
    hierarchy, not an isolated chunk. Its token count is what the 14k input
    budget is spent on, so it is also the thing that has to shrink to fit a
    smaller context window.
    """
    lines, seen = [], set()
    for s in sections:
        for depth, title in enumerate(s.path):
            key = tuple(s.path[: depth + 1])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"{'  ' * depth}{title}")
    return "\n".join(lines)


def corpus_stats(sections: list[LeafSection]) -> dict:
    if not sections:
        return {"n_leaves": 0}
    chars = [s.n_chars for s in sections]
    depths = [len(s.path) for s in sections]
    empty = sum(1 for c in chars if c < 200)
    return {
        "n_leaves": len(sections),
        "max_depth": max(depths),
        "mean_depth": round(sum(depths) / len(depths), 2),
        "mean_chars": int(sum(chars) / len(chars)),
        "median_chars": sorted(chars)[len(chars) // 2],
        "empty_sections": empty,
        "empty_pct": round(100 * empty / len(sections), 1),
        "duplicate_titles": len(sections) - len({s.docid for s in sections}),
        "toc_prompt_chars": len(toc_prompt(sections)),
    }


def save(sections: list[LeafSection], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for s in sections:
            fh.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    return out_path
