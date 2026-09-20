"""Build the per-book corpora and report what filtering removed.

Every number here goes in the results table. Silently dropping sections is
how a benchmark quietly stops meaning anything, so the counts are printed and
saved alongside the data.

Usage:  python scripts/build_corpus.py [books_folder] [out_folder]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stair.corpus import build_corpus, corpus_stats, save  # noqa: E402
from stair.prompts import fit_toc  # noqa: E402

CHARS_PER_TOKEN = 3.8


def slug(name: str) -> str:
    keep = [c if c.isalnum() else "-" for c in name.lower()]
    out = "".join(keep)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")[:48]


def main(books: str = "books_AI", out: str = "data/corpora") -> None:
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for pdf in sorted(Path(books).glob("*.pdf")):
        raw = build_corpus(pdf, drop_front_matter=False, unique_docids=False,
                           min_chars=0)
        if not raw:
            print(f"SKIP  {pdf.name[:50]} — no embedded ToC")
            continue

        clean = build_corpus(pdf)
        book_id = slug(pdf.stem)
        st = corpus_stats(clean)

        dropped = len(raw) - len(clean)
        dup_before = len(raw) - len({s.docid for s in raw})
        dup_after = len(clean) - len({s.docid for s in clean})
        budget = fit_toc(clean, 4000)

        path = save(clean, outdir / f"{book_id}.jsonl")
        entry = {
            "book_id": book_id,
            "source": pdf.name,
            "leaves_raw": len(raw),
            "leaves_kept": len(clean),
            "dropped": dropped,
            "dup_titles_before": dup_before,
            "dup_titles_after": dup_after,
            "max_depth": st["max_depth"],
            "mean_chars": st["mean_chars"],
            "toc_tokens": int(st["toc_prompt_chars"] / CHARS_PER_TOKEN),
            "toc_strategy_at_4k": budget["strategy"],
            "recall_ceiling": budget["recall_ceiling"],
            "file": str(path),
        }
        manifest.append(entry)

        print(f"{book_id}")
        print(f"   leaves   {len(raw)} raw -> {len(clean)} kept  "
              f"({dropped} dropped: front matter + thin)")
        print(f"   docids   {dup_before} duplicate titles -> {dup_after} after "
              f"path disambiguation")
        print(f"   ToC      ~{entry['toc_tokens']} tok, fits 4k as "
              f"'{budget['strategy']}', ceiling {budget['recall_ceiling']}%")

    (outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\n{len(manifest)} books -> {outdir}/  (manifest.json written)")
    if manifest:
        tot = sum(m["leaves_kept"] for m in manifest)
        print(f"total retrievable sections: {tot}")


if __name__ == "__main__":
    a = sys.argv
    main(a[1] if len(a) > 1 else "books_AI", a[2] if len(a) > 2 else "data/corpora")
