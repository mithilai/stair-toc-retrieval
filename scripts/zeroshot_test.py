"""Does STAIR need the finetuning at all?

The paper justifies training by reporting out-of-the-box Mistral at 13.8%.
But that baseline was almost certainly unconstrained — free to invent section
names that do not exist. Our trie makes invention impossible, and it costs
nothing to apply.

So this measures the experiment the paper did not run: an **untrained** model,
shown the table of contents, constrained to emit only real section titles.
Zero GPU-hours, zero retraining when the corpus changes.

If it lands near the trained numbers, the entire finetuning apparatus — and
the retrain-on-update problem, and the per-query cost — is optional.

Usage:
    python scripts/zeroshot_test.py --book whole-child
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
from stair.data import load_subset  # noqa: E402
from stair.evaluate import compare, evaluate  # noqa: E402
from stair.model import StairRetriever, load_model, vram_report  # noqa: E402
from stair.significance import compare_systems, format_comparisons  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="whole-child")
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--limit", type=int, default=200,
                    help="test queries; 200 is enough to see the effect")
    ap.add_argument("--four-bit", action="store_true", default=True)
    args = ap.parse_args()

    bench = load_subset(book=args.book, split="test")[0]
    queries = dict(list(bench.queries.items())[: args.limit]) if args.limit \
        else bench.queries
    gold = {k: v for k, v in bench.gold.items() if k in queries}

    print(f"book {args.book} | {len(bench.sections)} leaves | "
          f"{len(queries)} queries | model {args.model}", flush=True)
    print("NO TRAINING — base weights only, constrained decoding on\n", flush=True)

    metrics, preds_all = {}, {}

    preds_all["BM25"] = BM25Retriever(bench.sections).run(queries, k=3)
    metrics["BM25"] = evaluate(preds_all["BM25"], gold,
                               valid_docids=bench.valid_docids)

    model, tok = load_model(args.model, four_bit=args.four_bit)
    print(vram_report("base model"), flush=True)

    for tag, with_toc in (("zeroshot+ToC", True), ("zeroshot-noToC", False)):
        r = StairRetriever(model, tok, bench.sections, with_toc=with_toc)
        if with_toc:
            print(f"ToC: {r.toc_info['tokens']} tokens, "
                  f"'{r.toc_info['strategy']}'", flush=True)
        t0 = time.time()
        preds_all[tag] = r.run(queries, k=3, progress=True)
        metrics[tag] = evaluate(preds_all[tag], gold,
                                valid_docids=bench.valid_docids)
        print(f"{tag}: {time.time() - t0:.0f}s | {vram_report('')}", flush=True)

    # Bring in the trained run's scores on the same queries, if we have them.
    trained = Path(f"results/{args.book}_predictions.json")
    if trained.exists():
        old = json.loads(trained.read_text(encoding="utf-8"))
        # Label with whichever model actually produced those predictions,
        # not a hardcoded guess — the file is overwritten per evaluation run.
        meta = Path(f"results/{args.book}_test.json")
        src_model = "unknown"
        if meta.exists():
            src_model = json.loads(meta.read_text(encoding="utf-8")).get(
                "model", "unknown").split("/")[-1]
        for name, p in old.get("predictions", {}).items():
            shared = {q: v for q, v in p.items() if q in queries}
            if len(shared) >= 0.8 * len(queries):
                metrics[f"trained-{name}({src_model})"] = evaluate(
                    shared, {k: v for k, v in gold.items() if k in shared},
                    valid_docids=bench.valid_docids)

    print("\n" + "=" * 62)
    print("ZERO-TRAINING vs EVERYTHING ELSE")
    print("=" * 62)
    print(compare(metrics))

    print("\n" + "=" * 62)
    print("SIGNIFICANCE (R@1)")
    print("=" * 62)
    print(format_comparisons(compare_systems(preds_all, gold, k=1)))

    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"{args.book}_zeroshot.json"
    out.write_text(json.dumps({"book": args.book, "model": args.model,
                               "queries": len(queries), "metrics": metrics},
                              indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
