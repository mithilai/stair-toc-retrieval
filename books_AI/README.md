# books/

Drop source books here. **This folder is gitignored** — nothing in it is
committed, on purpose.

## What to put here

Format decides whether a book is usable, not its length:

| Format | Verdict |
|---|---|
| EPUB / HTML | Ideal — structural ToC, clean section boundaries |
| PDF with embedded bookmarks | Workable — ToC from bookmarks, text extraction is fiddly |
| Scanned / image PDF | Unusable — would need OCR + layout reconstruction |

The real budget is **ToC leaf sections**, not megabytes. A DSI memorises one
identifier per leaf section in its weights, so the corpus size that matters is
the section count: aim for roughly 5–10k total on a T5-base at 8 GB VRAM,
which works out to about 6–20 typical textbooks.

Domain spread matters too — the paper used 6 domains. Variety beats volume.

## Licensing

Anything here may be used for **local experiments**. Only **open-licence**
books (OpenStax, BCcampus, Open Music Theory and similar) may be named in the
blog post or included in a released dataset. See `../PROJECT.md`.

## Next step

Once books are in here, ask for the corpus audit: formats, ToC depth, leaf
counts per book, licence sort, and a shortlist of what to actually use.
