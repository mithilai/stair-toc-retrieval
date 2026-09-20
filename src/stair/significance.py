"""Significance testing for retrieval comparisons.

The paper establishes its headline gap with "a randomization test tailored to
retrieval systems" at p=0.05. That test is what separates a real difference
from three lucky queries, and without it a 5-point gap on 60 dev queries
reads as a finding when it is noise.

Implemented as a two-sided paired permutation test: both systems answer the
same queries, so scores pair up per query. Under the null hypothesis the two
systems are interchangeable, meaning each query's pair of scores could just as
well be swapped. Shuffle those swaps many times, and see how often chance
produces a gap at least as large as the observed one.
"""

from __future__ import annotations

import random
from collections.abc import Sequence


def per_query_scores(predictions: dict[str, Sequence[str]],
                     gold: dict[str, str], k: int = 1) -> dict[str, float]:
    """1.0 if the gold section is in the top k, else 0.0."""
    return {
        qid: (1.0 if gold[qid] in list(predictions.get(qid, []))[:k] else 0.0)
        for qid in gold
    }


def randomization_test(scores_a: dict[str, float], scores_b: dict[str, float],
                       n_iter: int = 10000, seed: int = 42) -> dict:
    """Two-sided paired permutation test on per-query scores.

    Returns the observed difference, the p-value, and the number of queries
    where the systems actually disagree — that last number is the real sample
    size of the comparison, and it is usually far smaller than the query count.
    """
    qids = sorted(set(scores_a) & set(scores_b))
    if not qids:
        return {"error": "no shared queries"}

    diffs = [scores_a[q] - scores_b[q] for q in qids]
    n = len(qids)
    observed = sum(diffs) / n
    discordant = sum(1 for d in diffs if d != 0)

    if discordant == 0:
        return {
            "mean_a": round(100 * sum(scores_a[q] for q in qids) / n, 2),
            "mean_b": round(100 * sum(scores_b[q] for q in qids) / n, 2),
            "observed_diff": 0.0, "p_value": 1.0, "significant": False,
            "n_queries": n, "n_discordant": 0,
            "note": "systems are identical on every query",
        }

    rng = random.Random(seed)
    hits = 0
    target = abs(observed)
    for _ in range(n_iter):
        total = 0.0
        for d in diffs:
            total += d if rng.random() < 0.5 else -d
        if abs(total / n) >= target - 1e-12:
            hits += 1

    p = (hits + 1) / (n_iter + 1)   # add-one keeps p strictly positive
    return {
        "mean_a": round(100 * sum(scores_a[q] for q in qids) / n, 2),
        "mean_b": round(100 * sum(scores_b[q] for q in qids) / n, 2),
        "observed_diff": round(100 * observed, 2),
        "p_value": round(p, 4),
        "significant": p < 0.05,
        "n_queries": n,
        "n_discordant": discordant,
        "n_iter": n_iter,
    }


def compare_systems(runs: dict[str, dict[str, Sequence[str]]],
                    gold: dict[str, str], k: int = 1,
                    n_iter: int = 10000) -> list[dict]:
    """Every pairwise comparison, with the paper's significance threshold."""
    scores = {name: per_query_scores(preds, gold, k=k)
              for name, preds in runs.items()}
    names = list(scores)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = randomization_test(scores[a], scores[b], n_iter=n_iter)
            out.append({"a": a, "b": b, "k": k, **r})
    return out


def format_comparisons(comparisons: list[dict]) -> str:
    lines = [f"{'comparison':<26}{'diff':>9}{'p':>9}  verdict",
             "-" * 62]
    for c in comparisons:
        if "error" in c:
            continue
        verdict = ("SIGNIFICANT" if c["significant"]
                   else "not significant — consistent with noise")
        lines.append(
            f"{c['a'] + ' vs ' + c['b']:<26}"
            f"{c['observed_diff']:>+9.2f}{c['p_value']:>9.4f}  {verdict}"
        )
    return "\n".join(lines)
