"""The STAIR retriever: a causal LM that generates a section title, and can
only generate a real one.

Retrieval here is generation. The model is shown the book's table of contents
and a question, and emits a leaf title. Constrained decoding over a trie of
the book's valid titles is what makes that safe — the model cannot invent a
section, because no path through the trie spells one that does not exist.

Model-agnostic on purpose. The paper uses Mistral-7B-Instruct-v0.2, which is
tight on 8 GB even in 4-bit; a smaller instruct model runs the same code and
is the sane place to establish that the pipeline is correct before scaling.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .constrained import build_trie, prefix_allowed_tokens_fn
from .prompts import fit_toc, retrieval_example


@dataclass
class GenConfig:
    max_new_tokens: int = 64        # the paper's output budget
    num_beams: int = 3              # need k=3 for R@3 and nDCG@3
    context_budget: int = 4000      # measured sufficient for our books (H3)


class StairRetriever:
    """Wraps a causal LM with a per-book trie and ToC prompt."""

    def __init__(self, model, tokenizer, sections: Sequence,
                 with_toc: bool = True, config: GenConfig | None = None):
        self.model = model
        self.tok = tokenizer
        self.sections = list(sections)
        self.with_toc = with_toc
        self.cfg = config or GenConfig()

        self.docids = [s.docid for s in self.sections]
        self.trie = build_trie(self.docids, tokenizer)

        budget = fit_toc(self.sections, self.cfg.context_budget, tokenizer)
        self.toc = budget["toc"] if with_toc else ""
        self.toc_info = budget

    def _prompt(self, query: str) -> str:
        return retrieval_example(query, self.toc, target="",
                                 with_toc=self.with_toc)["input"]

    def search(self, query: str, k: int = 3) -> list[str]:
        import torch

        enc = self.tok(self._prompt(query), return_tensors="pt",
                       truncation=True, max_length=self.cfg.context_budget + 512)
        enc = {k_: v.to(self.model.device) for k_, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.cfg.max_new_tokens,
                num_beams=max(k, self.cfg.num_beams),
                num_return_sequences=k,
                do_sample=False,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn(
                    self.trie, prompt_len
                ),
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            )

        seen, ranked = set(), []
        for seq in out:
            docid = self.tok.decode(seq[prompt_len:], skip_special_tokens=True).strip()
            if docid and docid not in seen:
                seen.add(docid)
                ranked.append(docid)
        return ranked[:k]

    def run(self, queries: dict[str, str], k: int = 3,
            progress: bool = True) -> dict[str, list[str]]:
        out = {}
        for i, (qid, q) in enumerate(queries.items()):
            out[qid] = self.search(q, k=k)
            if progress and (i + 1) % 50 == 0:
                # flush=True or this vanishes into a pipe buffer until exit
                print(f"  {i + 1}/{len(queries)}", flush=True)
        return out


def load_model(model_id: str, four_bit: bool = False, lora_path: str | None = None):
    """Load a causal LM for retrieval.

    ``four_bit`` needs bitsandbytes, which is the fragile part on Windows.
    bf16 on a small model avoids it entirely and is the recommended first run.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs: dict = {"dtype": torch.bfloat16, "device_map": "cuda:0"}
    if four_bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if lora_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tok


def vram_report(tag: str = "") -> str:
    import torch

    if not torch.cuda.is_available():
        return "no cuda"
    alloc = torch.cuda.memory_allocated() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{tag} VRAM {alloc:.2f}/{total:.1f} GB (peak {peak:.2f})"
