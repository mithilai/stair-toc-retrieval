"""End-to-end smoke test: does constrained retrieval actually work on GPU?

Runs an untrained model zero-shot. The accuracy is expected to be poor — the
paper's out-of-the-box Mistral scores 13.8% — so accuracy is not what this
checks. It checks three things that must hold before training is worth
starting:

1. The model loads and generates on the GPU inside 8 GB.
2. Every prediction is a real section title. Constrained decoding guarantees
   this, so a hallucination rate above 0.0 is a bug in the trie, not a result.
3. The whole loop runs without shape, device, or tokenizer errors.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stair.data import load_subset  # noqa: E402
from stair.evaluate import evaluate  # noqa: E402
from stair.model import StairRetriever, load_model, vram_report  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
N_QUERIES = int(sys.argv[2]) if len(sys.argv) > 2 else 20
BOOK = "whole-child"


def main() -> None:
    print(f"model: {MODEL}")
    t0 = time.time()
    model, tok = load_model(MODEL)
    print(f"loaded in {time.time() - t0:.0f}s | {vram_report('after load')}")

    bench = load_subset(book=BOOK, split="test")[0]
    print(f"book: {bench.book_id} | {len(bench.sections)} leaves | "
          f"{len(bench.queries)} test queries")

    qs = dict(list(bench.queries.items())[:N_QUERIES])

    for with_toc in (True, False):
        r = StairRetriever(model, tok, bench.sections, with_toc=with_toc)
        label = "STAIR (ToC in context)" if with_toc else "no-ToC ablation"
        if with_toc:
            print(f"ToC: {r.toc_info['tokens']} tokens, "
                  f"strategy '{r.toc_info['strategy']}', "
                  f"lossless={r.toc_info['lossless']}")
        print(f"trie: {len(r.docids)} docids")

        t1 = time.time()
        preds = r.run(qs, k=3, progress=False)
        dt = time.time() - t1

        m = evaluate(preds, {k: v for k, v in bench.gold.items() if k in qs},
                     valid_docids=bench.valid_docids)
        print(f"\n--- {label} ---")
        print(f"  R@1 {m['R@1']}  R@3 {m['R@3']}  nDCG@3 {m['nDCG@3']}")
        print(f"  hallucination_rate {m['hallucination_rate']}%  "
              f"(MUST be 0.0 — anything else is a trie bug)")
        print(f"  {dt:.1f}s for {len(qs)} queries ({dt / len(qs):.2f}s/query)")
        print(f"  {vram_report('peak')}")

        sample = list(preds.items())[0]
        print(f"  example: {qs[sample[0]][:60]}")
        print(f"       ->  {sample[1][0][:70] if sample[1] else '(empty)'}")


if __name__ == "__main__":
    main()
