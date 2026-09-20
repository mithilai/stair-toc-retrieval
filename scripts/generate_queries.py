"""Generate the query set for our own books with the paper's model.

The paper uses Mixtral 8x7B. We call the same model through OpenRouter rather
than substituting a weaker local one, because query quality bounds benchmark
quality — a bad generator produces a benchmark that measures the generator.

Every response is cached to disk, so this is safe to interrupt and rerun.

Usage:
    python scripts/generate_queries.py                 # all books
    python scripts/generate_queries.py --limit 5       # cheap dry run
    python scripts/generate_queries.py --book ai-eng   # one book
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stair.querygen import (  # noqa: E402
    MIXTRAL_8X7B,
    CachedBackend,
    generate_corpus_queries,
    load_env,
    mistral_generate,
    openrouter_generate,
)
from stair.toc import LeafSection  # noqa: E402


def load_corpus(path: Path) -> list[LeafSection]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(LeafSection(**json.loads(line)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="data/corpora")
    ap.add_argument("--out", default="data/queries")
    ap.add_argument("--model", default=None,
                    help="default: $MIXTRAL_MODEL, else open-mixtral-8x7b")
    ap.add_argument("--provider", default="local",
                    choices=["local", "mistral", "openrouter"])
    ap.add_argument("--book", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="sections per book; 0 = all. Use a small value first.")
    ap.add_argument("--n", type=int, default=3, help="queries per section")
    args = ap.parse_args()

    load_env()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    import os

    if args.model is None:
        args.model = {
            "local": "mistralai/Mistral-7B-Instruct-v0.2",
            "openrouter": "mistralai/mistral-medium-3.1",
        }.get(args.provider, os.environ.get("MISTRAL_MODEL") or MIXTRAL_8X7B)

    if args.provider == "local":
        from stair.querygen import LocalBackend

        print(f"loading {args.model} locally (4-bit)...", flush=True)
        call_obj = LocalBackend(args.model)
        raw_call = lambda p: call_obj(p)          # noqa: E731
    else:
        call = (mistral_generate if args.provider == "mistral"
                else openrouter_generate)
        raw_call = lambda p: call(p, model=args.model)   # noqa: E731

    backend = CachedBackend(
        raw_call,
        cache_path=f"{args.out}/.cache_{args.model.replace('/', '_')}.jsonl",
    )
    print(f"provider: {args.provider} | model: {args.model}")
    print(f"cache: {len(backend.cache)} responses already stored\n")

    for corpus_file in sorted(Path(args.corpora).glob("*.jsonl")):
        book = corpus_file.stem
        if args.book and args.book not in book:
            continue
        sections = load_corpus(corpus_file)
        if args.limit:
            sections = sections[: args.limit]

        print(f"{book}: {len(sections)} sections")
        t0 = time.time()
        result = generate_corpus_queries(sections, n=args.n, backend=backend)
        st = result["stats"]

        path = outdir / f"{book}.json"
        path.write_text(
            json.dumps({"book_id": book, "model": args.model, **result},
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"  {st['queries']} queries | "
              f"{st['dropped_title_leakage']} dropped for title leakage | "
              f"{st['failed_sections']} failed sections")
        print(f"  cache hits {backend.hits}, API calls {backend.misses} | "
              f"{time.time() - t0:.0f}s -> {path}\n")


if __name__ == "__main__":
    main()
