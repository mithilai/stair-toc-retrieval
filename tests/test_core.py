"""Tests for the dependency-free logic.

Everything here runs without torch, a GPU, or a network. These are the pieces
where a silent bug produces plausible-looking numbers rather than a crash,
which is the failure mode that actually cost time building this.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stair.constrained import TokenTrie  # noqa: E402
from stair.corpus import disambiguate, is_front_matter  # noqa: E402
from stair.evaluate import evaluate, ndcg_at_k, recall_at_k  # noqa: E402
from stair.prompts import fit_toc, render_full, render_leaves_only  # noqa: E402
from stair.significance import (  # noqa: E402
    per_query_scores,
    randomization_test,
)
from stair.toc import build_tree, leaves, section_spans  # noqa: E402

# --------------------------------------------------------------------------
# ToC tree
# --------------------------------------------------------------------------

BOOK = [
    (1, "Chapter 1", 1),
    (2, "1.1 Intro", 2),
    (3, "1.1.1 Deep", 3),
    (2, "1.2 Next", 8),
    (1, "Chapter 2", 12),
]


def test_build_tree_nests_by_level():
    roots = build_tree(BOOK)
    assert [r.title for r in roots] == ["Chapter 1", "Chapter 2"]
    assert [c.title for c in roots[0].children] == ["1.1 Intro", "1.2 Next"]


def test_leaves_are_deepest_nodes_only():
    assert [n.title for n in leaves(build_tree(BOOK))] == [
        "1.1.1 Deep", "1.2 Next", "Chapter 2",
    ]


def test_path_is_root_to_leaf():
    deep = leaves(build_tree(BOOK))[0]
    assert deep.path == ["Chapter 1", "1.1 Intro", "1.1.1 Deep"]


def test_section_ends_at_next_entry_at_any_level():
    """A leaf ends where the NEXT ToC entry starts, not the next sibling.

    '1.1.1 Deep' (p3) must end at '1.2 Next' (p8) even though 1.2 is its
    parent's sibling. Getting this wrong silently swallows other sections'
    text into the wrong docid.
    """
    spans = {n.title: (a, b) for n, a, b in section_spans(build_tree(BOOK), 20)}
    assert spans["1.1.1 Deep"] == (3, 8)
    assert spans["1.2 Next"] == (8, 12)
    assert spans["Chapter 2"] == (12, 21)   # runs to end of book


def test_malformed_level_jumps_do_not_crash():
    roots = build_tree([(1, "A", 1), (4, "B deep", 2), (2, "C", 3)])
    assert [r.title for r in roots] == ["A"]
    assert len(leaves(roots)) >= 1


def test_blank_titles_are_dropped():
    assert [r.title for r in build_tree([(1, "  ", 1), (1, "Real", 2)])] == ["Real"]


# --------------------------------------------------------------------------
# Constrained decoding
# --------------------------------------------------------------------------

@pytest.fixture
def trie() -> TokenTrie:
    t = TokenTrie()
    for seq in ([5, 6, 7], [5, 6, 9], [1, 2]):
        t.add(seq)
    return t


def test_trie_branches_on_shared_prefix(trie):
    assert sorted(trie.next_tokens([5])) == [6]
    assert sorted(trie.next_tokens([5, 6])) == [7, 9]


def test_trie_marks_only_full_sequences_complete(trie):
    assert trie.is_complete([5, 6, 7])
    assert not trie.is_complete([5, 6])


def test_trie_returns_empty_on_invalid_prefix(trie):
    """An unreachable prefix must be a dead end, never a silent fallback."""
    assert trie.next_tokens([9]) == []
    assert trie.next_tokens([5, 6, 7, 8]) == []


def test_trie_admits_nothing_outside_its_vocabulary(trie):
    """This property is what makes the hallucination rate 0.00%."""
    for bogus in ([3], [0], [5, 8], [1, 3]):
        assert trie.next_tokens(bogus) == []


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_recall_at_k_respects_cutoff():
    ranked = ["A", "B", "C"]
    assert recall_at_k(ranked, "A", 1) == 1.0
    assert recall_at_k(ranked, "B", 1) == 0.0
    assert recall_at_k(ranked, "C", 3) == 1.0


def test_ndcg_decays_with_rank():
    assert ndcg_at_k(["A"], "A", 3) == 1.0
    assert ndcg_at_k(["X", "A"], "A", 3) == pytest.approx(0.6309, abs=1e-3)
    assert ndcg_at_k(["X", "Y", "Z"], "A", 3) == 0.0


def test_evaluate_computes_known_case():
    gold = {"q1": "A", "q2": "B", "q3": "C"}
    preds = {"q1": ["A", "X", "Y"], "q2": ["X", "B", "Y"], "q3": ["GHOST", "C"]}
    m = evaluate(preds, gold, valid_docids={"A", "B", "C", "X", "Y"})
    assert m["R@1"] == pytest.approx(33.33, abs=0.01)   # 1 of 3
    assert m["R@3"] == 100.0                            # all within top 3
    assert m["hallucination_rate"] == pytest.approx(33.33, abs=0.01)


def test_hallucination_is_zero_when_all_predictions_are_valid():
    gold = {"q1": "A", "q2": "B"}
    preds = {"q1": ["A"], "q2": ["A"]}
    m = evaluate(preds, gold, valid_docids={"A", "B"})
    assert m["hallucination_rate"] == 0.0


def test_evaluate_handles_empty_predictions():
    m = evaluate({}, {"q1": "A"}, valid_docids={"A"})
    assert m["R@1"] == 0.0
    assert m["answered"] == 0


# --------------------------------------------------------------------------
# Significance
# --------------------------------------------------------------------------

def test_small_gap_is_not_significant():
    """15/60 vs 12/60 is three questions. It must not read as a finding."""
    a = {f"q{i}": float(i < 15) for i in range(60)}
    b = {f"q{i}": float(i < 12) for i in range(60)}
    r = randomization_test(a, b, n_iter=4000)
    assert r["observed_diff"] == pytest.approx(5.0, abs=0.01)
    assert not r["significant"]
    assert r["n_discordant"] == 3


def test_large_consistent_gap_is_significant():
    a = {f"q{i}": 1.0 for i in range(200)}
    b = {f"q{i}": float(i < 40) for i in range(200)}
    r = randomization_test(a, b, n_iter=4000)
    assert r["significant"]
    assert r["p_value"] < 0.05


def test_identical_systems_report_no_difference():
    a = {f"q{i}": float(i % 2) for i in range(50)}
    r = randomization_test(a, dict(a), n_iter=500)
    assert r["observed_diff"] == 0.0
    assert r["p_value"] == 1.0
    assert r["n_discordant"] == 0


def test_p_value_is_never_zero():
    """Add-one smoothing: p=0 would claim impossible certainty."""
    a = {f"q{i}": 1.0 for i in range(100)}
    b = {f"q{i}": 0.0 for i in range(100)}
    assert randomization_test(a, b, n_iter=1000)["p_value"] > 0


def test_per_query_scores_match_k():
    preds = {"q1": ["X", "A"], "q2": ["B"]}
    gold = {"q1": "A", "q2": "B"}
    assert per_query_scores(preds, gold, k=1) == {"q1": 0.0, "q2": 1.0}
    assert per_query_scores(preds, gold, k=3) == {"q1": 1.0, "q2": 1.0}


# --------------------------------------------------------------------------
# Corpus hygiene
# --------------------------------------------------------------------------

@dataclass
class FakeSection:
    docid: str
    path: list
    text: str = "x" * 500

    @property
    def path_docid(self) -> str:
        return " > ".join(self.path)

    @property
    def n_chars(self) -> int:
        return len(self.text)


@pytest.mark.parametrize("title", [
    "Cover", "Copyright", "Table of Contents", "Index", "Acknowledgments",
    "About the Author", "Preface", "brief contents",
])
def test_front_matter_is_detected(title):
    assert is_front_matter(title)


@pytest.mark.parametrize("title", [
    "Brain Cells: Neurons", "2.2 Early Childhood", "Indexing Strategies",
])
def test_real_sections_are_not_front_matter(title):
    """'Indexing Strategies' must survive even though 'Index' is stoplisted."""
    assert not is_front_matter(title)


def test_duplicate_titles_become_unique():
    secs = [
        FakeSection("Summary", ["Ch 1", "Summary"]),
        FakeSection("Summary", ["Ch 2", "Summary"]),
        FakeSection("Unique", ["Ch 1", "Unique"]),
    ]
    out, n_fixed = disambiguate(secs)
    assert n_fixed == 2
    assert len({s.docid for s in out}) == 3
    assert [s for s in out if s.docid == "Unique"], "unique titles untouched"


def test_identical_paths_still_resolve():
    secs = [FakeSection("S", ["A", "S"]), FakeSection("S", ["A", "S"])]
    out, _ = disambiguate(secs)
    assert len({s.docid for s in out}) == 2


# --------------------------------------------------------------------------
# Prompt budgeting
# --------------------------------------------------------------------------

@pytest.fixture
def sections() -> list:
    return [
        FakeSection("1.1.1 Gradient methods",
                    ["Ch 1 Foundations", "1.1 Optimization", "1.1.1 Gradient methods"]),
        FakeSection("1.1.2 Second order",
                    ["Ch 1 Foundations", "1.1 Optimization", "1.1.2 Second order"]),
        FakeSection("2.1 Shallow", ["Ch 2 Systems", "2.1 Shallow"]),
    ]


def test_render_full_includes_every_node_once(sections):
    out = render_full(sections)
    assert out.count("Ch 1 Foundations") == 1
    assert out.count("1.1 Optimization") == 1
    assert "1.1.2 Second order" in out


def test_render_leaves_only_drops_interior_nodes(sections):
    out = render_leaves_only(sections)
    assert "Ch 1 Foundations" not in out
    assert "1.1.1 Gradient methods" in out


def test_generous_budget_stays_lossless(sections):
    r = fit_toc(sections, budget_tokens=500)
    assert r["strategy"] == "full"
    assert r["lossless"]
    assert r["recall_ceiling"] == 100.0


def test_tight_budget_degrades_and_says_so(sections):
    r = fit_toc(sections, budget_tokens=20)
    assert not r["lossless"]
    assert r["strategy"] != "full"


def test_impossible_budget_reports_failure_rather_than_lying(sections):
    r = fit_toc(sections, budget_tokens=1)
    assert r.get("fits") is False
