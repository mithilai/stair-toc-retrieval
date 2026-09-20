"""Table of Contents extraction.

STAIR's document identifier is the *leaf section title* from a book's ToC —
a real string like "2.2.3 Initiative vs. Guilt (Preschool Years)", not an
opaque id. Retrieval is then constrained generation over the set of valid
leaf titles for that book.

This module turns a PDF into that set. It is deliberately model-agnostic:
nothing here depends on which LLM ends up doing the retrieval.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass
class TocNode:
    """One entry in the ToC hierarchy."""

    level: int
    title: str
    page: int
    children: list[TocNode] = field(default_factory=list)
    parent: TocNode | None = field(default=None, repr=False)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def ancestors(self) -> list[TocNode]:
        """Root-first chain of ancestors, excluding self."""
        chain, node = [], self.parent
        while node is not None:
            chain.append(node)
            node = node.parent
        return list(reversed(chain))

    @property
    def path(self) -> list[str]:
        """Titles from root to self — the full hierarchical address."""
        return [n.title for n in self.ancestors] + [self.title]

    def walk(self) -> Iterator[TocNode]:
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass
class LeafSection:
    """A retrievable unit: the finest-granularity ToC entry plus its text."""

    book_id: str
    docid: str           # the leaf title — STAIR's identifier
    path: list[str]      # full ancestor chain, for the path-as-docid ablation
    page_start: int
    page_end: int        # exclusive
    text: str = ""

    @property
    def path_docid(self) -> str:
        """Alternative identifier: the full hierarchical path.

        The paper uses the bare leaf title. Whether the full path does better
        is a cheap ablation and a good blog result, so we carry both.
        """
        return " > ".join(self.path)

    @property
    def n_chars(self) -> int:
        return len(self.text)


def build_tree(entries: list[tuple[int, str, int]]) -> list[TocNode]:
    """Build a ToC forest from pymupdf's flat ``[level, title, page]`` list.

    pymupdf emits levels starting at 1. Levels can jump by more than one in
    malformed books, so we attach to the nearest shallower ancestor rather
    than assuming a well-formed increment.
    """
    roots: list[TocNode] = []
    stack: list[TocNode] = []

    for level, title, page in entries:
        title = " ".join(str(title).split())
        if not title:
            continue
        node = TocNode(level=int(level), title=title, page=int(page))

        while stack and stack[-1].level >= node.level:
            stack.pop()

        if stack:
            node.parent = stack[-1]
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)

    return roots


def leaves(roots: list[TocNode]) -> list[TocNode]:
    return [n for root in roots for n in root.walk() if n.is_leaf]


def extract_toc(pdf_path: str) -> list[TocNode]:
    """Read a PDF's embedded ToC. Returns an empty list if it has none."""
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        raw = doc.get_toc(simple=True)
    return build_tree([(lvl, title, page) for lvl, title, page in raw])


def section_spans(roots: list[TocNode], n_pages: int) -> list[tuple[TocNode, int, int]]:
    """Assign each leaf a page range.

    A leaf owns pages from its own start until the next ToC entry begins, at
    any level — the next entry is what actually ends it, not the next sibling.
    """
    ordered = sorted(
        (n for root in roots for n in root.walk()),
        key=lambda n: (n.page, n.level),
    )
    starts = [n.page for n in ordered]

    spans: list[tuple[TocNode, int, int]] = []
    for i, node in enumerate(ordered):
        if not node.is_leaf:
            continue
        end = next((p for p in starts[i + 1:] if p > node.page), n_pages + 1)
        spans.append((node, node.page, end))
    return spans
