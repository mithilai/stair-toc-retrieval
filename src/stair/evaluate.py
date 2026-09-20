"""Retrieval metrics — Recall@1, Recall@3, nDCG@3.

These are the paper's three metrics. It computes them via BeIR/pytrec_eval;
we implement them directly so the eval harness has no heavyweight dependency,
and cross-check against pytrec_eval when it is installed.

Each query here has exactly one gold leaf section, which is how SearchTome is
built. That makes Recall@k equivalent to Hits@k and simplifies nDCG.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(ranked: Sequence[str], gold: str, k: int) -> float:
    return 1.0 if gold in list(ranked)[:k] else 0.0


def ndcg_at_k(ranked: Sequence[str], gold: str, k: int) -> float:
    """With a single relevant document, IDCG is 1 so DCG is already normalized."""
    for rank, docid in enumerate(list(ranked)[:k], start=1):
        if docid == gold:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def evaluate(
    predictions: dict[str, Sequence[str]],
    gold: dict[str, str],
    valid_docids: set[str] | None = None,
) -> dict[str, float]:
    """Score a run.

    ``hallucination_rate`` is the share of top-1 predictions that name a
    section which does not exist in the book. Under constrained decoding it
    must be exactly 0.0 — if it is not, the constraint is not being applied,
    and that is a bug rather than a result.
    """
    if not gold:
        return {}

    qids = list(gold)
    r1 = r3 = nd3 = 0.0
    hallucinated = 0
    answered = 0

    for qid in qids:
        ranked = list(predictions.get(qid, []))
        if ranked:
            answered += 1
            if valid_docids is not None and ranked[0] not in valid_docids:
                hallucinated += 1
        r1 += recall_at_k(ranked, gold[qid], 1)
        r3 += recall_at_k(ranked, gold[qid], 3)
        nd3 += ndcg_at_k(ranked, gold[qid], 3)

    n = len(qids)
    out = {
        "n_queries": n,
        "R@1": round(100 * r1 / n, 2),
        "R@3": round(100 * r3 / n, 2),
        "nDCG@3": round(100 * nd3 / n, 2),
        "answered": answered,
    }
    if valid_docids is not None:
        out["hallucination_rate"] = round(100 * hallucinated / max(1, answered), 4)
    return out


def compare(runs: dict[str, dict[str, float]]) -> str:
    """Render a table of runs. Never label any row a paper reproduction."""
    if not runs:
        return "(no runs)"
    cols = ["R@1", "R@3", "nDCG@3", "hallucination_rate"]
    width = max(len(name) for name in runs)
    head = f"{'system'.ljust(width)} | " + " | ".join(c.rjust(9) for c in cols)
    lines = [head, "-" * len(head)]
    for name, m in runs.items():
        cells = [
            f"{m[c]:9.2f}" if c in m and m[c] is not None else " " * 9
            for c in cols
        ]
        lines.append(f"{name.ljust(width)} | " + " | ".join(cells))
    return "\n".join(lines)
