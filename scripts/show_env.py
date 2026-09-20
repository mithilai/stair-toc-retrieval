"""Print .env variable NAMES with all values redacted.

Never grep .env by hand: a redaction pattern tied to one variable name silently
stops redacting the moment a variable is renamed, and the secret goes to
stdout. This redacts by position — everything after the first '=' — so it
cannot miss.
"""

from __future__ import annotations

import sys
from pathlib import Path

NON_SECRET = {"MISTRAL_MODEL", "MISTRAL_API", "MIXTRAL_MODEL", "MIXTRAL_API"}


def main(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        print(f"{path} not found")
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name in NON_SECRET:
            print(f"{name}={value}")
        else:
            print(f"{name}=<redacted len={len(value.strip())}>")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".env")
