"""Loading the benchmark.

The HF subset is ground truth for headline numbers — it is the only external
reference point that survives, since the paper's own benchmark is gone. Our
own book corpus is the second act: evidence the architecture generalises.

Both are normalized to the same shape so the eval harness cannot tell them
apart:  queries {qid: text}, gold {qid: docid}, valid_docids {docid}.
"""

from __future__ import annotations

from dataclasses import dataclass

from .toc import LeafSection

SUBSET_REPO = "SGK86/searchtome-subset"


@dataclass
class Benchmark:
    name: str
    book_id: str
    queries: dict[str, str]
    gold: dict[str, str]
    valid_docids: set[str]
    sections: list | None = None

    def split(self, keep: set[str]) -> Benchmark:
        q = {k: v for k, v in self.queries.items() if k in keep}
        return Benchmark(
            name=self.name,
            book_id=self.book_id,
            queries=q,
            gold={k: v for k, v in self.gold.items() if k in q},
            valid_docids=self.valid_docids,
            sections=self.sections,
        )

    def __repr__(self) -> str:
        return (
            f"<Benchmark {self.name}/{self.book_id} "
            f"{len(self.queries)} queries, {len(self.valid_docids)} docids>"
        )


SUBSET_BOOKS = [
    ("education", "digitalage"),
    ("education", "openmusictheory"),
    ("education", "whole-child"),
    ("social-sciences", "auralskills"),
    ("social-sciences", "behavioralecon"),
    ("social-sciences", "compgov"),
]


def _hf(path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(SUBSET_REPO, path, repo_type="dataset")


def _jsonl(path: str) -> list[dict]:
    import json

    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_subset(book: str | None = None, split: str | None = "test") -> list[Benchmark]:
    """Load the community SearchTome subset.

    ``datasets.load_dataset`` cannot read this repo — its per-book files have
    conflicting inferred schemas and it dies with "Couldn't cast array of type
    string to null" (H12). The files themselves are fine, so we read them
    directly and skip the loader entirely.

    Gold labels are ``leaf_id``, the dataset's stable key. ``leaf_title`` is
    carried alongside because the title is what STAIR actually generates.
    """
    out: list[Benchmark] = []
    for domain, name in SUBSET_BOOKS:
        if book and name != book:
            continue
        leaves = _jsonl(_hf(f"gen/{domain}/{name}/leaf_docs.jsonl"))
        questions = _jsonl(_hf(f"gen/{domain}/{name}/questions.jsonl"))

        # The identifier must be the SECTION TITLE, not the numeric leaf_id.
        # That is the entire point of STAIR: a semantically meaningful docid
        # the model can reason about from the ToC. Using leaf_id would rebuild
        # the opaque-id DSI baseline and mislabel it STAIR (H13).
        titles: dict[str, str] = {}
        used: dict[str, int] = {}
        for row in leaves:
            lid = str(row["leaf_id"])
            title = " ".join(str(row.get("title") or lid).split()) or lid
            # Titles repeat across chapters; disambiguate so the identifier
            # actually identifies, and so the trie has one path per section.
            if title in used:
                used[title] += 1
                title = f"{title} [{used[title]}]"
            else:
                used[title] = 1
            titles[lid] = title

        sections = [
            LeafSection(
                book_id=name,
                docid=titles[str(row["leaf_id"])],
                path=[titles[str(row["leaf_id"])]],
                page_start=int(row.get("page_start") or 0),
                page_end=int(row.get("page_end") or 0),
                text=str(row.get("text") or ""),
            )
            for row in leaves
        ]

        queries, gold = {}, {}
        skipped = 0
        for i, row in enumerate(questions):
            if split and str(row.get("split")) != split:
                continue
            lid = str(row["leaf_id"])
            if lid not in titles:      # query points at a section not in the corpus
                skipped += 1
                continue
            qid = f"{name}:{i}"
            queries[qid] = str(row["query"])
            gold[qid] = titles[lid]
        if skipped:
            print(f"  [{name}] {skipped} queries dropped: gold leaf_id absent "
                  f"from leaf_docs")

        out.append(
            Benchmark(
                name=f"searchtome-subset/{domain}",
                book_id=name,
                queries=queries,
                gold=gold,
                valid_docids={s.docid for s in sections},
                sections=sections,
            )
        )
    return out
