"""Audit the PDFs in books_AI/ — which are usable as a STAIR corpus, and why.

Usage:  python scripts/audit_books.py [folder]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Windows consoles default to cp1252 and die on non-Latin book titles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stair.corpus import build_corpus, corpus_stats  # noqa: E402

# Rough chars-per-token for English prose; good enough for a budget check.
CHARS_PER_TOKEN = 3.8


def main(folder: str = "books_AI") -> None:
    pdfs = sorted(Path(folder).glob("*.pdf"))
    if not pdfs:
        print(f"no PDFs in {folder}/")
        return

    for pdf in pdfs:
        print("=" * 70)
        print(pdf.name[:66])
        try:
            sections = build_corpus(pdf)
        except Exception as exc:
            print(f"  FAILED to parse: {type(exc).__name__}: {exc}")
            continue

        if not sections:
            print("  UNUSABLE — no embedded Table of Contents")
            continue

        st = corpus_stats(sections)
        toc_tokens = int(st["toc_prompt_chars"] / CHARS_PER_TOKEN)
        print(f"  leaf sections : {st['n_leaves']}")
        print(f"  ToC depth     : max {st['max_depth']}, mean {st['mean_depth']}")
        print(f"  section text  : mean {st['mean_chars']} chars, "
              f"median {st['median_chars']}")
        print(f"  empty/thin    : {st['empty_sections']} ({st['empty_pct']}%)")
        print(f"  dup titles    : {st['duplicate_titles']}  "
              f"(ambiguous docids — these break exact-match eval)")
        print(f"  ToC prompt    : ~{toc_tokens} tokens", end="")
        for budget, label in ((14000, "paper 14k"), (8000, "8k"), (4000, "4k")):
            print(f" | {label}: {'fits' if toc_tokens < budget else 'OVERFLOW'}", end="")
        print()

        print("  first leaves  :")
        for s in sections[:3]:
            print(f"    - {s.docid[:58]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "books_AI")
