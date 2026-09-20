"""LoRA finetuning for STAIR.

The paper trains two correlated tasks together:

1. **Ingestion** — passage -> its section title. Writes the corpus into the
   model's parameters, which is what makes this a *generative* retriever
   rather than a prompt-stuffing trick.
2. **Retrieval** — ToC + question -> section title.

Both are plain causal-LM objectives with the loss masked to the target span:
the model is never scored on reproducing the prompt, only the identifier.

The ToC/no-ToC switch is threaded through from `prompts.py`, so the ablation
that carries the paper's entire claim is one flag, not a second codebase.

Paper hyperparameters: LoRA r=16 alpha=32, up to 200 epochs, early stopping
on dev R@1 with patience 20. Learning rate, batch size and optimizer are not
stated, so ours are marked as choices in HURDLES.md rather than passed off as
the paper's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    four_bit: bool = False
    with_toc: bool = True

    lora_r: int = 16            # paper
    lora_alpha: int = 32        # paper
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # Not specified in the paper — our choices.
    lr: float = 2e-4
    batch_size: int = 1
    grad_accum: int = 8
    epochs: int = 20            # paper allows 200; we cap for an 8 GB budget
    patience: int = 5           # paper 20, scaled to our epoch cap
    warmup_ratio: float = 0.03

    max_len: int = 4096         # measured sufficient (H3), vs the paper's 14k
    window_ingestion: bool = True   # cover whole sections, not just openings
    ingest_overlap: int = 300
    max_windows: int = 8
    # ~12k chars ~= 3.2k tokens, fitting max_len with room for the target.
    # The paper allows 14k tokens per input; the old 3,000-char cap was a
    # deviation of mine that fed the model ~18x less text per example.
    ingest_max_chars: int = 12000
    seed: int = 42
    out_dir: str = "checkpoints/stair"
    extra: dict = field(default_factory=dict)


def build_examples(sections: Sequence, queries: dict[str, str],
                   gold: dict[str, str], toc: str, with_toc: bool = True,
                   cfg: TrainConfig | None = None) -> list[dict]:
    """Interleave ingestion and retrieval examples."""
    from .prompts import ingestion_example, ingestion_examples, retrieval_example

    cfg = cfg or TrainConfig()
    examples: list[dict] = []
    for s in sections:
        if s.n_chars < 200:
            continue
        if cfg.window_ingestion:
            examples.extend(ingestion_examples(
                s, max_chars=cfg.ingest_max_chars,
                overlap=cfg.ingest_overlap, max_windows=cfg.max_windows))
        else:
            examples.append(ingestion_example(s, max_chars=cfg.ingest_max_chars))
    examples += [
        retrieval_example(q, toc, gold[qid], with_toc=with_toc)
        for qid, q in queries.items()
        if qid in gold
    ]
    return examples


def tokenize_example(ex: dict, tok, max_len: int) -> dict:
    """Mask the loss to the target span.

    Without this the model spends most of its gradient learning to echo the
    table of contents, which is both wasteful and not the task.
    """
    prompt_ids = tok(ex["input"], add_special_tokens=False)["input_ids"]
    target_ids = tok(" " + ex["target"], add_special_tokens=False)["input_ids"]
    target_ids = target_ids + [tok.eos_token_id]

    # Truncate the prompt from the left so the target always survives.
    budget = max_len - len(target_ids)
    if budget < 1:
        target_ids = target_ids[: max_len - 1]
        budget = 1
    prompt_ids = prompt_ids[-budget:]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def collate(batch: list[dict], pad_id: int) -> dict:
    import torch

    width = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        pad = width - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
        out["labels"].append(b["labels"] + [-100] * pad)
        out["attention_mask"].append(b["attention_mask"] + [0] * pad)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def attach_lora(model, cfg: TrainConfig):
    from peft import LoraConfig, get_peft_model

    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {trainable:,} trainable / {total:,} total "
          f"({100 * trainable / total:.3f}%)")
    return model
