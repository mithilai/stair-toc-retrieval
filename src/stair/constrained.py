"""Constrained generation over the valid leaf-title set.

This is the mechanism behind STAIR's <0.05% hallucination rate, and it is
independent of the finetuning: the model can only emit token sequences that
spell out a real ToC leaf title for the book being searched. An unconstrained
model is free to invent a plausible-sounding section that does not exist;
this one is not.

Implemented as a token-level trie driving HuggingFace's
``prefix_allowed_tokens_fn``, so it works with any causal LM.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable


class TokenTrie:
    """Prefix trie over tokenized identifiers."""

    __slots__ = ("children", "is_terminal")

    def __init__(self) -> None:
        self.children: dict[int, TokenTrie] = {}
        self.is_terminal = False

    def add(self, tokens: Iterable[int]) -> None:
        node = self
        for tok in tokens:
            node = node.children.setdefault(tok, TokenTrie())
        node.is_terminal = True

    def next_tokens(self, prefix: list[int]) -> list[int]:
        """Tokens that may legally follow ``prefix``. Empty means dead end."""
        node = self
        for tok in prefix:
            node = node.children.get(tok)
            if node is None:
                return []
        return list(node.children)

    def is_complete(self, prefix: list[int]) -> bool:
        node = self
        for tok in prefix:
            node = node.children.get(tok)
            if node is None:
                return False
        return node.is_terminal

    def __len__(self) -> int:
        return sum(1 for _ in self._walk())

    def _walk(self):
        if self.is_terminal:
            yield ()
        for tok, child in self.children.items():
            for rest in child._walk():
                yield (tok,) + rest


def build_trie(docids: Iterable[str], tokenizer, eos_token_id: int | None = None) -> TokenTrie:
    """Trie over the book's leaf titles, each terminated by EOS.

    EOS is part of the trie so the model cannot stop halfway through a title
    and emit a valid-looking prefix of a real section.
    """
    eos = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    trie = TokenTrie()
    for docid in docids:
        toks = tokenizer(docid, add_special_tokens=False)["input_ids"]
        trie.add(list(toks) + [eos])
    return trie


def prefix_allowed_tokens_fn(
    trie: TokenTrie,
    prompt_lengths: list[int] | int,
) -> Callable[[int, object], list[int]]:
    """Adapt a trie to ``model.generate(prefix_allowed_tokens_fn=...)``.

    ``prompt_lengths`` is where the generated identifier starts for each item
    in the batch — everything before that is prompt and must be ignored when
    matching against the trie.
    """

    def fn(batch_id: int, input_ids) -> list[int]:
        start = (
            prompt_lengths
            if isinstance(prompt_lengths, int)
            else prompt_lengths[batch_id]
        )
        generated = input_ids[start:].tolist()
        allowed = trie.next_tokens(generated)
        if not allowed:
            # Dead end: the only legal move is to have finished already.
            # Returning the last token keeps generate() from crashing; the
            # sequence is discarded downstream.
            return [int(input_ids[-1])]
        return allowed

    return fn


def decode_to_docid(tokens: list[int], tokenizer) -> str:
    return tokenizer.decode(tokens, skip_special_tokens=True).strip()
