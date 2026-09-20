"""Run the full ToC-vs-no-ToC sweep across books, unattended.

Each cell is a separate subprocess so one crash cannot take down the sweep,
and every cell resumes if it was interrupted. Progress is appended to
`logs/sweep_state.json` after each cell, so the sweep itself is restartable.

Usage:
    python scripts/sweep.py                       # all 6 books, both variants
    python scripts/sweep.py --books whole-child auralskills
    python scripts/sweep.py --dry-run             # show the plan and exit
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
STATE = ROOT / "logs" / "sweep_state.json"

# Smallest first: a broken sweep should fail fast and cheap, not after the
# most expensive book.
BOOKS = ["whole-child", "auralskills", "digitalage",
         "behavioralecon", "compgov", "openmusictheory"]


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"done": [], "failed": []}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def best_for(book: str, variant: str) -> float | None:
    h = ROOT / "checkpoints" / f"{book}-{variant}" / "history.json"
    if not h.exists():
        return None
    try:
        return json.loads(h.read_text(encoding="utf-8")).get("best_dev_R@1")
    except json.JSONDecodeError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="*", default=BOOKS)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--dev-n", type=int, default=100)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="rerun cells already marked done")
    args = ap.parse_args()

    cells = [(b, v) for b in args.books for v in ("stair", "no-toc")]
    state = load_state()

    print(f"sweep: {len(cells)} cells ({len(args.books)} books x 2 variants)")
    print(f"model: {args.model} | {args.epochs} epochs | dev-n {args.dev_n}")
    est = len(cells) * args.epochs * 340 / 3600
    print(f"rough estimate: {est:.1f} hours at ~340s/epoch\n")

    for book, variant in cells:
        key = f"{book}:{variant}"
        if key in state["done"] and not args.force:
            print(f"SKIP  {key} (done, best dev R@1 {best_for(book, variant)})")
            continue
        cmd = [str(PYTHON), str(ROOT / "scripts" / "train_stair.py"),
               "--book", book, "--epochs", str(args.epochs),
               "--dev-n", str(args.dev_n), "--model", args.model, "--resume"]
        if variant == "no-toc":
            cmd.append("--no-toc")

        if args.dry_run:
            print("PLAN  " + " ".join(cmd[1:]))
            continue

        print(f"RUN   {key}  ({time.strftime('%H:%M:%S')})", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(ROOT))
        mins = (time.time() - t0) / 60

        if proc.returncode == 0:
            state["done"].append(key)
            print(f"OK    {key} in {mins:.0f}m | "
                  f"best dev R@1 {best_for(book, variant)}\n", flush=True)
        else:
            state["failed"].append({"cell": key, "code": proc.returncode})
            print(f"FAIL  {key} exit {proc.returncode} after {mins:.0f}m — "
                  f"continuing\n", flush=True)
        save_state(state)

    if args.dry_run:
        return

    print("=" * 58)
    print(f"{'book':<18}{'STAIR':>9}{'no-ToC':>9}{'gap':>9}")
    print("=" * 58)
    for book in args.books:
        a, b = best_for(book, "stair"), best_for(book, "no-toc")
        if a is None or b is None:
            print(f"{book:<18}{'—' if a is None else a:>9}"
                  f"{'—' if b is None else b:>9}{'':>9}")
            continue
        print(f"{book:<18}{a:>9}{b:>9}{a - b:>+9.2f}")
    print("\ngap = STAIR minus no-ToC. Positive means the ToC helped.")
    print("Dev-set numbers on a subsample — run evaluate_run.py for test-set")
    print("results with significance testing before concluding anything.")
    if state["failed"]:
        print(f"\nfailed cells: {state['failed']}")


if __name__ == "__main__":
    main()
