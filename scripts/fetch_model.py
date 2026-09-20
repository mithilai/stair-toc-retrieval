"""Download a model with resume, retrying through stalls.

Unauthenticated HuggingFace downloads get throttled and can stall outright —
the connection stays open, bytes stop arriving, and nothing errors. A plain
`snapshot_download` sits there forever.

This retries in a loop. `.incomplete` blobs mean each attempt resumes rather
than restarting, so a stall costs seconds, not the whole file.

Usage:
    python scripts/fetch_model.py mistralai/Mistral-7B-Instruct-v0.2
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PATTERNS = ["*.safetensors", "*.json", "*.model", "tokenizer*"]


def cache_size_mb(repo: str) -> float:
    from huggingface_hub import constants

    d = Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}"
    if not d.exists():
        return 0.0
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1024**2


def main() -> None:
    from huggingface_hub import snapshot_download

    repo = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Mistral-7B-Instruct-v0.2"
    max_attempts = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    for attempt in range(1, max_attempts + 1):
        before = cache_size_mb(repo)
        print(f"[{time.strftime('%H:%M:%S')}] attempt {attempt}: "
              f"{before:,.0f} MB cached", flush=True)
        try:
            path = snapshot_download(
                repo,
                allow_patterns=PATTERNS,
                max_workers=2,          # fewer parallel streams stall less
                etag_timeout=30,
            )
            print(f"COMPLETE -> {path}", flush=True)
            print(f"final size: {cache_size_mb(repo):,.0f} MB", flush=True)
            return
        except Exception as exc:
            after = cache_size_mb(repo)
            gained = after - before
            print(f"  {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            print(f"  gained {gained:,.0f} MB this attempt; retrying", flush=True)
            time.sleep(min(30, 2 * attempt))

    print(f"gave up after {max_attempts} attempts", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
