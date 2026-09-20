"""Train STAIR (or its no-ToC ablation) on one book.

The ablation is the whole point: `--no-toc` changes exactly one thing, whether
the table of contents appears in the retrieval prompt. Same model, same data,
same schedule, same seed. Any gap between the two runs is the paper's claim,
isolated.

Usage:
    python scripts/train_stair.py --book whole-child
    python scripts/train_stair.py --book whole-child --no-toc
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class Tee:
    """Write to the console AND a log file, flushing every line.

    Without the flush, piping this script through anything buffers the whole
    run and you see nothing until it exits — which is exactly how the first
    training run became invisible for 15 minutes.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", encoding="utf-8", errors="replace")
        self.stdout = sys.stdout

    def write(self, text: str) -> int:
        self.stdout.write(text)
        self.stdout.flush()
        self.file.write(text)
        self.file.flush()
        return len(text)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stair.data import load_subset  # noqa: E402
from stair.evaluate import evaluate  # noqa: E402
from stair.model import StairRetriever, load_model, vram_report  # noqa: E402
from stair.prompts import fit_toc  # noqa: E402
from stair.train import (  # noqa: E402
    TrainConfig,
    attach_lora,
    build_examples,
    collate,
    tokenize_example,
)


def evaluate_dev(model, tok, sections, queries, gold, valid, with_toc, n=100):
    model.eval()
    sub = dict(list(queries.items())[:n])
    r = StairRetriever(model, tok, sections, with_toc=with_toc)
    preds = r.run(sub, k=3, progress=False)
    m = evaluate(preds, {k: v for k, v in gold.items() if k in sub},
                 valid_docids=valid)
    model.train()
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="whole-child")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--no-toc", action="store_true",
                    help="the vanilla-DSI ablation")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--dev-n", type=int, default=100)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the last saved epoch if present")
    args = ap.parse_args()

    import torch

    with_toc = not args.no_toc
    tag = "stair" if with_toc else "no-toc"
    cfg = TrainConfig(model_id=args.model, with_toc=with_toc, lr=args.lr,
                      grad_accum=args.grad_accum, epochs=args.epochs,
                      max_len=args.max_len, four_bit=args.four_bit)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Include the model in the path. Without it a 7B run silently overwrites
    # a 0.5B one of the same book+variant, and the two become impossible to
    # tell apart afterwards.
    slug = args.model.split("/")[-1].replace(".", "-")
    out = Path(f"checkpoints/{args.book}-{tag}-{slug}")
    legacy = Path(f"checkpoints/{args.book}-{tag}")
    if legacy.exists() and not out.exists():
        out.mkdir(parents=True, exist_ok=True)
        print(f"note: legacy checkpoint dir {legacy} exists from an earlier "
              f"run of a possibly different model; writing to {out}")
    out.mkdir(parents=True, exist_ok=True)

    log_path = Path(f"logs/{args.book}-{tag}-{slug}.log")
    tee = Tee(log_path)
    sys.stdout = tee
    print(f"\n{'=' * 70}")
    print(f"run started {time.strftime('%Y-%m-%d %H:%M:%S')}  ->  {log_path}")
    print(f"args: {vars(args)}")
    print("=" * 70)

    train_b = load_subset(book=args.book, split="train")[0]
    dev_b = load_subset(book=args.book, split="dev")[0]
    sections = train_b.sections

    print(f"book {args.book} | {len(sections)} leaves | "
          f"{len(train_b.queries)} train | {len(dev_b.queries)} dev")
    print(f"variant: {'STAIR (ToC in prompt)' if with_toc else 'no-ToC ablation'}")

    model, tok = load_model(cfg.model_id, four_bit=cfg.four_bit)
    model = attach_lora(model, cfg)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()

    budget = fit_toc(sections, cfg.max_len - 512, tok)
    toc = budget["toc"] if with_toc else ""
    if with_toc:
        print(f"ToC: {budget['tokens']} tokens, '{budget['strategy']}', "
              f"lossless={budget['lossless']}")

    examples = build_examples(sections, train_b.queries, train_b.gold, toc,
                              with_toc=with_toc, cfg=cfg)
    tokenized = [tokenize_example(e, tok, cfg.max_len) for e in examples]
    print(f"{len(examples)} training examples "
          f"({len(sections)} ingestion + {len(examples) - len(sections)} retrieval)")
    print(f"max example length: {max(len(t['input_ids']) for t in tokenized)} tokens")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr
    )
    steps = (len(tokenized) // cfg.grad_accum) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=max(1, steps), pct_start=0.06
    )
    pad = tok.pad_token_id or tok.eos_token_id

    history, best, bad, start_epoch = [], -1.0, 0, 1

    state_path = out / "train_state.pt"
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location=model.device,
                           weights_only=False)
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        # Restore the ADAPTER too. Loading only the optimizer put a decayed
        # learning-rate schedule on a freshly-initialised LoRA — strictly
        # worse than starting over, and invisible except that the loss jumps
        # back to its epoch-1 value.
        adapter = out / "adapter_model.safetensors"
        if adapter.exists():
            from safetensors.torch import load_file

            sd = load_file(str(adapter))
            missing = model.load_state_dict(
                {k.replace("base_model.model.", "base_model.model."): v
                 for k, v in sd.items()}, strict=False)
            n_loaded = len(sd) - len(getattr(missing, "unexpected_keys", []))
            print(f"restored {n_loaded}/{len(sd)} adapter tensors from {adapter.name}")
        else:
            print("WARNING: no adapter to restore — resuming weights from scratch")
        history = state.get("history", [])
        best = state.get("best", -1.0)
        bad = state.get("bad", 0)
        start_epoch = state.get("epoch", 0) + 1
        print(f"resumed from epoch {start_epoch - 1}: best dev R@1 {best}, "
              f"{len(history)} epochs of history")
    elif args.resume:
        print("--resume given but no saved state; starting fresh")

    for epoch in range(start_epoch, args.epochs + 1):
        random.shuffle(tokenized)
        t0, total, nb = time.time(), 0.0, 0
        opt.zero_grad()
        for i, ex in enumerate(tokenized):
            batch = collate([ex], pad)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(**batch).loss
            (loss / cfg.grad_accum).backward()
            total += loss.item()
            nb += 1
            if (i + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                opt.step()
                sched.step()
                opt.zero_grad()
            if (i + 1) % 100 == 0:
                print(f"     epoch {epoch} step {i + 1}/{len(tokenized)} "
                      f"loss {total / max(1, nb):.4f} "
                      f"({(time.time() - t0) / (i + 1):.2f}s/ex)")

        m = evaluate_dev(model, tok, sections, dev_b.queries, dev_b.gold,
                         dev_b.valid_docids, with_toc, n=args.dev_n)
        row = {"epoch": epoch, "loss": round(total / max(1, nb), 4),
               "dev_R@1": m["R@1"], "dev_R@3": m["R@3"],
               "dev_nDCG@3": m["nDCG@3"],
               "hallucination": m.get("hallucination_rate"),
               "secs": round(time.time() - t0)}
        history.append(row)
        print(f"  epoch {epoch:>2} loss {row['loss']:>7.4f} | "
              f"dev R@1 {m['R@1']:>5} R@3 {m['R@3']:>5} | "
              f"halluc {row['hallucination']}% | {row['secs']}s | "
              f"{vram_report('')}")

        (out / "history.json").write_text(
            json.dumps({"book": args.book, "variant": tag,
                        "model": cfg.model_id, "with_toc": with_toc,
                        "best_dev_R@1": max(best, m["R@1"]),
                        "status": "running", "history": history}, indent=2),
            encoding="utf-8")
        with (Path("logs") / f"{args.book}-{tag}-{slug}.jsonl").open(
                "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

        if m["R@1"] > best:
            best, bad = m["R@1"], 0
            model.save_pretrained(str(out))
            print(f"     new best -> saved {out}")
        else:
            bad += 1

        # Save resumable state every epoch, not only on improvement: an
        # interrupted run should restart from where it stopped, not from the
        # last epoch that happened to improve.
        torch.save({"optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "history": history, "best": best, "bad": bad,
                    "epoch": epoch}, state_path)

        if bad >= args.patience:
            print(f"  early stop (no dev R@1 gain in {args.patience} epochs)")
            break

    (out / "history.json").write_text(
        json.dumps({"book": args.book, "variant": tag, "model": cfg.model_id,
                    "with_toc": with_toc, "best_dev_R@1": best,
                    "history": history}, indent=2), encoding="utf-8")
    print(f"\nbest dev R@1 {best} | history -> {out}/history.json")


if __name__ == "__main__":
    main()
