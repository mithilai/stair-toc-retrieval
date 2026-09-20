"""Prompt construction and the ToC context budget.

STAIR trains on two correlated tasks:

1. **Ingestion** — section text -> its leaf title. Puts the corpus into the
   model's parameters.
2. **Retrieval** — query + the book's whole ToC -> the leaf title. The ToC in
   context is the "global structure" the paper is about.

The paper allows 14,000 input tokens, which is what makes task 2 possible:
one book's entire ToC fits. On a smaller context that stops being true, so
this module also implements *budgeted* ToC rendering — progressively lossier
strategies, each one honest about what it discards.

That budget is the project's central open problem (H3 in HURDLES.md). The
strategies here are what make it measurable rather than fatal.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

CHARS_PER_TOKEN = 3.8  # rough English-prose estimate for dependency-free checks

INGEST_INSTRUCTION = (
    "Read the passage from the book and name the section it belongs to."
)
RETRIEVE_INSTRUCTION = (
    "Using the book's table of contents, name the single section that answers "
    "the question."
)


def _est_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return int(len(text) / CHARS_PER_TOKEN)


# --- ToC rendering strategies, cheapest loss first ----------------------------

def render_full(sections: Sequence) -> str:
    """Every node, indented by depth. Lossless."""
    lines, seen = [], set()
    for s in sections:
        for depth in range(len(s.path)):
            key = tuple(s.path[: depth + 1])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"{'  ' * depth}{s.path[depth]}")
    return "\n".join(lines)


def render_leaves_only(sections: Sequence) -> str:
    """Drop interior nodes. Loses hierarchy, keeps every retrievable target."""
    seen, out = set(), []
    for s in sections:
        if s.docid not in seen:
            seen.add(s.docid)
            out.append(s.docid)
    return "\n".join(out)


def render_truncated(sections: Sequence, max_title_chars: int = 48) -> str:
    """Full tree with long titles clipped. Loses wording, keeps structure."""
    lines, seen = [], set()
    for s in sections:
        for depth in range(len(s.path)):
            key = tuple(s.path[: depth + 1])
            if key in seen:
                continue
            seen.add(key)
            title = s.path[depth]
            if len(title) > max_title_chars:
                title = title[: max_title_chars - 1].rstrip() + "…"
            lines.append(f"{'  ' * depth}{title}")
    return "\n".join(lines)


def render_depth_capped(sections: Sequence, max_depth: int = 3) -> str:
    """Collapse below ``max_depth``.

    Destructive in a way the others are not: leaves deeper than the cap stop
    being addressable at all, so recall is capped below 100% by construction.
    Only reach for this when nothing else fits, and report the ceiling.
    """
    lines, seen = [], set()
    for s in sections:
        for depth in range(min(len(s.path), max_depth)):
            key = tuple(s.path[: depth + 1])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"{'  ' * depth}{s.path[depth]}")
    return "\n".join(lines)


STRATEGIES: list[tuple[str, Callable[[Sequence], str]]] = [
    ("full", render_full),
    ("truncated", render_truncated),
    ("leaves_only", render_leaves_only),
    ("depth_capped_3", lambda s: render_depth_capped(s, 3)),
]


def fit_toc(sections: Sequence, budget_tokens: int, tokenizer=None) -> dict:
    """Pick the least-lossy ToC rendering that fits ``budget_tokens``.

    Returns the rendering plus what it cost, because a number produced under
    a lossy ToC is not comparable to one produced under a full ToC and the
    results table has to say so.
    """
    attempts = []
    for name, fn in STRATEGIES:
        text = fn(sections)
        n = _est_tokens(text, tokenizer)
        attempts.append((name, n))
        if n <= budget_tokens:
            addressable = _addressable(sections, name)
            return {
                "strategy": name,
                "toc": text,
                "tokens": n,
                "budget": budget_tokens,
                "lossless": name == "full",
                "addressable_leaves": addressable,
                "recall_ceiling": round(100 * addressable / max(1, len(sections)), 2),
                "attempts": attempts,
            }

    name, fn = STRATEGIES[-1]
    text = fn(sections)
    addressable = _addressable(sections, name)
    return {
        "strategy": name,
        "toc": text,
        "tokens": _est_tokens(text, tokenizer),
        "budget": budget_tokens,
        "lossless": False,
        "fits": False,
        "addressable_leaves": addressable,
        "recall_ceiling": round(100 * addressable / max(1, len(sections)), 2),
        "attempts": attempts,
    }


def _addressable(sections: Sequence, strategy: str) -> int:
    """How many leaves remain reachable under a strategy."""
    if strategy == "depth_capped_3":
        return sum(1 for s in sections if len(s.path) <= 3)
    return len(sections)


# --- training examples --------------------------------------------------------

def ingestion_example(section, max_chars: int = 3000) -> dict:
    text = section.text[:max_chars]
    return {
        "task": "ingest",
        "input": f"{INGEST_INSTRUCTION}\n\nPassage:\n{text}\n\nSection:",
        "target": section.docid,
    }


def retrieval_example(query: str, toc: str, target: str, with_toc: bool = True) -> dict:
    """``with_toc=False`` is the vanilla-DSI ablation — same everything, no ToC.

    That single flag is the paper's core claim. Any gap between the two runs
    is the contribution.
    """
    if with_toc:
        body = (
            f"{RETRIEVE_INSTRUCTION}\n\n"
            f"Table of contents:\n{toc}\n\n"
            f"Question: {query}\n\nSection:"
        )
    else:
        body = f"{RETRIEVE_INSTRUCTION}\n\nQuestion: {query}\n\nSection:"
    return {"task": "retrieve", "input": body, "target": target,
            "with_toc": with_toc}


def ingestion_examples(section, max_chars: int = 3000,
                       overlap: int = 300, max_windows: int = 8) -> list[dict]:
    """Cover a whole section with overlapping windows, not just its opening.

    Truncating at ``max_chars`` silently discards everything past the cut —
    56% of the text across our own corpus, 65% for the longest book. The model
    then cannot retrieve on any of it, because it never saw it. Worse, the
    discarding is invisible: training time looks reassuringly flat precisely
    because the extra content is being thrown away.

    Each window becomes its own example pointing at the same section id, so a
    long section contributes proportionally more signal and cost scales with
    content — which is what one would expect, and what truncation was hiding.

    ``overlap`` keeps a passage straddling a boundary intact in at least one
    window. ``max_windows`` stops one 200k-character appendix dominating an
    epoch.
    """
    text = section.text
    if len(text) <= max_chars:
        return [ingestion_example(section, max_chars)]

    step = max(1, max_chars - overlap)
    out: list[dict] = []
    for start in range(0, len(text), step):
        chunk = text[start:start + max_chars]
        if len(chunk) < 200:          # trailing scrap carries no signal
            break
        out.append({
            "task": "ingest",
            "input": (f"{INGEST_INSTRUCTION}\n\nPassage:\n{chunk}"
                      f"\n\nSection:"),
            "target": section.docid,
            "window": len(out),
        })
        if len(out) >= max_windows:
            break
    return out
