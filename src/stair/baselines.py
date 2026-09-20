"""Baselines.

Two, deliberately:

* **BM25** — lexical, no training, deterministic. The floor. The paper gets
  59.5% with it.
* **Vanilla DSI** — the same model and training as STAIR but *without* the
  ToC in the prompt. This is the one that matters: it isolates the paper's
  actual claim. Any gap between it and STAIR is the contribution; everything
  else is shared code.

DPR and the out-of-the-box Mistral baseline are skipped on purpose — they
cost training and inference without testing the ToC hypothesis.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    """BM25 over leaf-section text, returning leaf titles as identifiers.

    Indexes section *body text*, not titles — a title alone is too short for
    BM25 to score meaningfully, and the comparison to STAIR is only fair if
    both get the section content.
    """

    def __init__(self, sections: Sequence, use_path_docid: bool = False):
        from rank_bm25 import BM25Okapi

        self.sections = list(sections)
        self.docids = [
            s.path_docid if use_path_docid else s.docid for s in self.sections
        ]
        corpus = [
            tokenize(f"{s.docid}\n{' > '.join(s.path)}\n{s.text}")
            for s in self.sections
        ]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 3) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        # Deduplicate while preserving rank: sections can share a title.
        out, seen = [], set()
        for i in order:
            docid = self.docids[i]
            if docid not in seen:
                seen.add(docid)
                out.append(docid)
        return out

    def run(self, queries: dict[str, str], k: int = 3) -> dict[str, list[str]]:
        return {qid: self.search(q, k) for qid, q in queries.items()}
