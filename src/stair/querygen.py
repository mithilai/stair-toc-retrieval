"""Synthetic query generation.

The paper generates training and evaluation queries with Mixtral 8x7B —
"multiple questions covering all important topics in the paragraph". We keep
that shape and stay provider-agnostic: a local open model via Ollama is the
faithful substitute, and any callable can be swapped in.

Two properties matter more than the generator's identity:

1. **Answerability.** A query must be answerable from its source section, or
   the gold label is wrong and the benchmark measures noise.
2. **No title leakage.** If the question quotes the section title, retrieval
   collapses into string matching and every system scores high for the wrong
   reason. This is the single easiest way to build a benchmark that flatters
   itself, so we filter for it and report how much was caught.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence

OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT = """You are helping build a retrieval benchmark from a textbook.

Below is one section of the book. Write {n} distinct questions that this
section answers.

Rules:
- Each question must be answerable using ONLY this section.
- Do NOT use the section's title or any distinctive phrase from it.
- Do NOT refer to "this section", "the text", "the passage", or the book.
- Write questions a reader would actually type into a search box.
- Vary them: some factual, some conceptual, some about when or why.

Section text:
\"\"\"
{text}
\"\"\"

Return only a JSON array of {n} question strings, nothing else."""


def ollama_generate(prompt: str, model: str = "llama3.1:8b",
                    timeout: int = 180) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.8},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"]


def _parse_questions(raw: str, n: int) -> list[str]:
    """Models wrap JSON in prose and fences. Recover the array either way."""
    match = re.search(r"\[.*\]", raw, re.S)
    if match:
        try:
            items = json.loads(match.group(0))
            out = [str(q).strip() for q in items if str(q).strip()]
            if out:
                return out[:n]
        except json.JSONDecodeError:
            pass
    lines = [
        re.sub(r'^\s*(?:[-*\d.)\]]+\s*)|^\s*"|"\s*,?\s*$', "", ln).strip()
        for ln in raw.splitlines()
    ]
    return [ln for ln in lines if ln.endswith("?")][:n]


def _leaks_title(question: str, title: str, min_overlap: int = 4) -> bool:
    """True if the question copies a distinctive run of words from the title."""
    def words(s: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", s.lower())

    q, t = words(question), words(title)
    if len(t) < min_overlap:
        return False
    for i in range(len(t) - min_overlap + 1):
        gram = t[i:i + min_overlap]
        for j in range(len(q) - min_overlap + 1):
            if q[j:j + min_overlap] == gram:
                return True
    return False


def generate_for_section(section, n: int = 3, max_chars: int = 4000,
                         backend: Callable[[str], str] | None = None
                         ) -> tuple[list[str], int]:
    """Queries for one section, plus how many were dropped for title leakage."""
    backend = backend or ollama_generate
    if section.n_chars < 200:
        return [], 0
    raw = backend(PROMPT.format(n=n, text=section.text[:max_chars]))
    questions = _parse_questions(raw, n)
    kept = [q for q in questions if not _leaks_title(q, section.docid)]
    return kept, len(questions) - len(kept)


def generate_corpus_queries(sections: Sequence, n: int = 3,
                            backend: Callable[[str], str] | None = None,
                            progress: bool = True) -> dict:
    """Generate across a book. Returns benchmark-shaped queries + gold + stats."""
    queries: dict[str, str] = {}
    gold: dict[str, str] = {}
    leaked = failed = 0

    for i, section in enumerate(sections):
        try:
            qs, n_leaked = generate_for_section(section, n=n, backend=backend)
            leaked += n_leaked
        except Exception as exc:
            failed += 1
            code = getattr(exc, "code", "")
            if progress:
                print(f"  [{i}] {type(exc).__name__}{f' {code}' if code else ''}"
                      f" — skipped: {str(exc)[:90]}", flush=True)
            continue
        for j, q in enumerate(qs):
            qid = f"{section.book_id}:{i}:{j}"
            queries[qid] = q
            gold[qid] = section.docid
        if progress and (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(sections)} sections, "
                  f"{len(queries)} queries", flush=True)

    return {
        "queries": queries,
        "gold": gold,
        "stats": {
            "sections": len(sections),
            "queries": len(queries),
            "dropped_title_leakage": leaked,
            "failed_sections": failed,
        },
    }


# --- OpenRouter backend -------------------------------------------------------
#
# The paper generates queries with Mixtral 8x7B. Using that exact model removes
# a confound rather than adding one: if our benchmark differs from theirs, it
# should not be because a weaker generator wrote worse questions.

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MIXTRAL = "mistralai/mixtral-8x7b-instruct"

# Minimum seconds between API calls. Backing off after a 429 is reactive and
# loses the request; pacing avoids provoking the limiter at all.
_MIN_INTERVAL = 0.7
_last_call = [0.0]


def _pace() -> None:
    import time

    wait = _MIN_INTERVAL - (time.monotonic() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.monotonic()


def load_env(path: str = ".env") -> None:
    """Read KEY=VALUE lines into the environment.

    The key lives in .env (gitignored), never in source and never pasted into
    a chat transcript.
    """
    import os
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def openrouter_generate(prompt: str, model: str = MIXTRAL,
                        timeout: int = 120, max_retries: int = 9) -> str:
    """Call OpenRouter, retrying on rate limits and transient failures.

    Nine attempts with exponential backoff spans ~4 minutes, against the ~31
    seconds that five gave. A rate-limit window outlasting the retry budget is
    indistinguishable from a hard failure, and that is what silently lost 107
    of one book's 142 sections.
    """
    import os
    import random
    import time

    # Accept both spellings — the underscored one is easy to write by hand.
    key = (os.environ.get("OPENROUTER_API_KEY")
           or os.environ.get("OPEN_ROUTER_API_KEY"))
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Put it in .env as "
            "OPENROUTER_API_KEY=sk-or-... (the file is gitignored)."
        )

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 512,
    }).encode()

    last: Exception | None = None
    for attempt in range(max_retries):
        _pace()
        req = urllib.request.Request(
            OPENROUTER_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read())
            return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (408, 429, 500, 502, 503, 520, 524, 529):
                time.sleep(min(90, 2 ** attempt) + random.random())
                continue
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:200]
            except Exception:
                pass
            raise RuntimeError(f"OpenRouter {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"OpenRouter failed after {max_retries} attempts: {last}")


class CachedBackend:
    """Wrap a backend with an append-only disk cache.

    Generation costs money and takes hours. Every response is written to a
    JSONL file keyed by a hash of the prompt, so a crash, a rate limit or a
    changed mind never means paying for the same section twice.
    """

    def __init__(self, backend: Callable[[str], str], cache_path: str):
        import hashlib
        from pathlib import Path

        self._hash = hashlib.sha256
        self.backend = backend
        self.path = Path(cache_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, str] = {}
        self.hits = self.misses = 0

        if self.path.exists():
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        self.cache[row["key"]] = row["response"]
                    except (json.JSONDecodeError, KeyError):
                        continue

    def _key(self, prompt: str) -> str:
        return self._hash(prompt.encode("utf-8")).hexdigest()[:32]

    def __call__(self, prompt: str) -> str:
        key = self._key(prompt)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        response = self.backend(prompt)
        self.misses += 1
        self.cache[key] = response
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "response": response},
                                ensure_ascii=False) + "\n")
        return response


# --- Mistral La Plateforme (direct) -------------------------------------------
#
# The paper's generator, called directly rather than through a broker. Mistral
# exposes Mixtral 8x7B as "open-mixtral-8x7b" on an OpenAI-compatible endpoint.

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MIXTRAL_8X7B = "open-mixtral-8x7b"


def mistral_generate(prompt: str, model: str | None = None,
                     timeout: int = 120, max_retries: int = 5) -> str:
    """Call Mistral's API directly. Retries on rate limits and transient errors."""
    import os
    import random
    import time

    key = os.environ.get("MISTRAL_API_KEY") or os.environ.get("MIXTRAL_API_KEY")
    if not key:
        raise RuntimeError(
            "MISTRAL_API_KEY not set. Put it in .env (which is gitignored)."
        )
    model = (model or os.environ.get("MISTRAL_MODEL")
             or os.environ.get("MIXTRAL_MODEL") or MIXTRAL_8X7B)
    url = (os.environ.get("MISTRAL_API") or os.environ.get("MIXTRAL_API")
           or MISTRAL_URL)

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 512,
    }).encode()

    last: Exception | None = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read())
            return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            last = exc
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            if exc.code == 429:
                limit = exc.headers.get("x-ratelimit-limit-req-minute")
                if limit == "0":
                    raise RuntimeError(
                        "Mistral rate limit is 0 requests/minute: the account "
                        "has no quota at all, so this is not throttling and "
                        "retrying cannot help. Activate the account "
                        "(console.mistral.ai — free tier needs phone "
                        "verification) or add billing."
                    ) from exc
            if exc.code in (429, 500, 502, 503, 529):
                time.sleep(min(60, 2 ** attempt) + random.random())
                continue
            raise RuntimeError(f"Mistral API {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"Mistral API failed after {max_retries} attempts: {last}")


# --- Local generation (free, no API) ------------------------------------------
#
# Paid APIs are out. We already download Mistral-7B-Instruct-v0.2 for the
# retrieval experiment, and it is from the same family as the paper's Mixtral
# generator — so generating queries with it locally costs nothing and is
# arguably closer in spirit than a hosted substitute.
#
# Slower than an API, but it runs unattended and cannot run out of credit.

class LocalBackend:
    """Generate with a local HF model. Loads once, reused across sections."""

    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 four_bit: bool = True, max_new_tokens: int = 400):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        kwargs: dict = {"dtype": torch.bfloat16, "device_map": "cuda:0"}
        if four_bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=4096).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=self.tok.pad_token_id,
            )
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:],
                               skip_special_tokens=True)
