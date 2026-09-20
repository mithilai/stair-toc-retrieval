"""Evaluate trained adapters on the test split, with significance testing.

Three fixes over the first version:

1. **Per-query predictions are saved.** Generation is the expensive part; a
   comparison should never require regenerating it.
2. **Every pairwise gap gets a p-value** from the paper's randomization test.
   A 5-point difference on a small query set is noise, and the table should
   say so rather than leaving the reader to assume.
3. **Both H14 columns are reported** — answerable and all.

Usage:
    python scripts/evaluate_run.py --book whole-child
    python scripts/evaluate_run.py --book whole-child --limit 200
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

from stair.baselines import BM25Retriever  # noqa: E402
from stair.data import SUBSET_BOOKS, _hf, _jsonl, load_subset  # noqa: E402
from stair.evaluate import compare, evaluate  # noqa: E402
from stair.model import StairRetriever, load_model, vram_report  # noqa: E402
from stair.significance import compare_systems, format_comparisons  # noqa: E402


def orphan_count(book: str, split: str = "test") -> tuple[int, int]:
    domain = next(d for d, n in SUBSET_BOOKS if n == book)
    leaves = {str(r["leaf_id"])
              for r in _jsonl(_hf(f"gen/{domain}/{book}/leaf_docs.jsonl"))}
    qs = [r for r in _jsonl(_hf(f"gen/{domain}/{book}/questions.jsonl"))
          if r.get("split") == split]
    return sum(1 for r in qs if str(r["leaf_id"]) not in leaves), len(qs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="whole-child")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--four-bit", action="store_true", default=True)
    args = ap.parse_args()

    bench = load_subset(book=args.book, split="test")[0]
    queries = bench.queries
    if args.limit:
        queries = dict(list(queries.items())[: args.limit])
    gold = {k: v for k, v in bench.gold.items() if k in queries}
    orphans, published = orphan_count(args.book)

    print(f"book {args.book} | {len(bench.sections)} leaves | "
          f"{len(queries)} answerable queries", flush=True)
    print(f"published test split: {published}, of which {orphans} "
          f"({100 * orphans / published:.1f}%) have gold sections missing "
          f"from the corpus\n", flush=True)

    metrics: dict[str, dict] = {}
    preds_all: dict[str, dict] = {}

    t0 = time.time()
    preds_all["BM25"] = BM25Retriever(bench.sections).run(queries, k=3)
    metrics["BM25"] = evaluate(preds_all["BM25"], gold,
                               valid_docids=bench.valid_docids)
    print(f"BM25 done in {time.time() - t0:.0f}s", flush=True)

    for tag, with_toc in (("STAIR", True), ("no-ToC", False)):
        slug = args.model.split("/")[-1].replace(".", "-")
        variant = "stair" if with_toc else "no-toc"
        ckpt = Path(f"checkpoints/{args.book}-{variant}-{slug}")
        if not (ckpt / "adapter_config.json").exists():
            ckpt = Path(f"checkpoints/{args.book}-{variant}")   # legacy path
        if not (ckpt / "adapter_config.json").exists():
            print(f"{tag}: no adapter at {ckpt} — skipping", flush=True)
            continue
        model, tok = load_model(args.model, four_bit=args.four_bit,
                                lora_path=str(ckpt))
        r = StairRetriever(model, tok, bench.sections, with_toc=with_toc)
        t0 = time.time()
        preds_all[tag] = r.run(queries, k=3, progress=True)
        metrics[tag] = evaluate(preds_all[tag], gold,
                                valid_docids=bench.valid_docids)
        print(f"{tag} done in {time.time() - t0:.0f}s | {vram_report('')}",
              flush=True)
        del model
        import torch
        torch.cuda.empty_cache()

    # Recompute each metric as if the orphan queries were also asked and failed.
    for m in metrics.values():
        for key in ("R@1", "R@3", "nDCG@3"):
            m[f"{key}_all"] = round(m[key] * len(queries) / published, 2)

    print("\n" + "=" * 62)
    print("ANSWERABLE ONLY")
    print("=" * 62)
    print(compare(metrics))

    print("\n" + "=" * 62)
    print(f"SIGNIFICANCE — paired randomization test, R@1, {len(queries)} queries")
    print("=" * 62)
    comps = {"R@1": compare_systems(preds_all, gold, k=1),
             "R@3": compare_systems(preds_all, gold, k=3)}
    for k, cs in comps.items():
        print(f"\n@{k}:")
        print(format_comparisons(cs))

    print("\nNOT comparable to the paper's SearchTome numbers.")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{args.book}_test.json").write_text(json.dumps({
        "book": args.book, "model": args.model,
        "queries_evaluated": len(queries),
        "published_test_size": published,
        "orphans": orphans,
        "note": "not comparable to the paper; community subset",
        "metrics": metrics, "significance": comps,
    }, indent=2), encoding="utf-8")
    (outdir / f"{args.book}_predictions.json").write_text(
        json.dumps({"gold": gold, "predictions": preds_all}, indent=1),
        encoding="utf-8")
    print(f"\nsaved {outdir}/{args.book}_test.json "
          f"and {args.book}_predictions.json")


if __name__ == "__main__":
    main()
