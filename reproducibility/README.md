# Saved experiment records

This directory documents the saved measurements from the community
`SGK86/searchtome-subset` benchmark. The results characterize this independent
implementation under a small training budget. They do not reproduce the
original STAIR benchmark or its reported scores.

The related manuscript, its PDF, and `video/` are kept local and excluded
from the code-and-evidence archive.

## Verify the saved results

From the repository root, with Python 3.11 or newer:

```bash
python scripts/verify_results.py
```

This uses the Python standard library and the repository's metric and
randomization-test implementations. It does not load model weights, use a
GPU, download data, or overwrite predictions. It checks the artifact hashes,
recomputes the 7B test metrics and paired tests, checks the six-book audit
calculations, and verifies the development histories against the saved logs.
It also recomputes the trained scores and ToC comparison on the first 200
answerable test queries.

## Archive contents

The code-and-evidence package contains the implementation, scripts, tests, public README illustrations,
result JSON files, selected experiment logs, checkpoint JSON metadata, and
these reproducibility records. `manifest.json` records SHA-256 checksums and
the base Git commit. The checksums describe the actual packaged files,
including changes made after that commit. UTF-8 text is stored with LF line
endings. Verification normalizes CRLF to LF before checking text hashes, so
Git checkouts on Windows and Linux verify against the same manifest. Binary
files are checked byte for byte.

Only the selected research tools and evidence files are included. Model weights,
optimizer states, source books, raw corpus passages, credentials, private
project notes, the PDF, LaTeX source, and video files are excluded. The
package also excludes local packaging helpers and machine-specific launch
scripts. The included verification command works on an extracted archive.

## Evidence inventory

| Measurement | Saved evidence |
|---|---|
| Six-book orphan audit and BM25 | `results/bm25_subset.json` |
| 7B test metrics and paired tests, 610 queries | `results/whole-child_test.json` |
| Gold titles and 7B/BM25 ranked predictions | `results/whole-child_predictions.json` |
| Untrained Mistral, first 200 answerable test queries | `results/whole-child_zeroshot.json`, `logs/chain.log` |
| Historical Qwen-0.5B test metrics and paired tests | `logs/eval-whole-child.log` |
| Mistral ToC training and four development epochs | `logs/whole-child-stair.log`, `logs/whole-child-stair.jsonl` |
| Qwen and Mistral no-ToC training | `logs/whole-child-no-toc.log`, `logs/whole-child-no-toc.jsonl` |
| Failed Mistral no-ToC continuation | `logs/whole-child-no-toc-Mistral-7B-Instruct-v0-2.log` |
| Mistral ToC checkpoint history and LoRA settings | `checkpoints/whole-child-stair-Mistral-7B-Instruct-v0-2/history.json`, `adapter_config.json` in that directory |
| Mistral no-ToC checkpoint history and LoRA settings | `checkpoints/whole-child-no-toc-Mistral-7B-Instruct-v0-2/history.json`, `adapter_config.json` in that directory |
| Benchmark revision, input hashes, ToC census, and valid titles | `reproducibility/benchmark_snapshot.json` |

Paths in the table are relative to the repository/archive root.

## Interpretation and gaps in the record

- The raw zero-shot summary labels imported trained scores as `0.5B`.
  Recomputing them from the saved Mistral predictions gives exactly those
  scores: ToC R@1 43.50 and R@3 63.50; no-ToC R@1 42.50 and R@3 60.50.
  The archived original is preserved, and the verification output identifies
  the trained model as Mistral-7B.
- The selected Mistral adapters are from epoch 3. The ToC history has four
  completed epochs. The no-ToC history has three, despite retaining a
  historical `status: running` field. Its attempted continuation did not
  restore adapter weights and is not a valid fourth training epoch.
- `logs/whole-child-no-toc.log` and its JSONL file contain Qwen records
  followed by Mistral records. Their order and original model labels are
  retained.
- Earlier Qwen test predictions and zero-shot per-query predictions were not
  retained. Their printed metrics and tests can be inspected in the logs,
  but cannot all be independently recomputed from this package. The original
  Qwen ToC training history is also absent from the surviving checkpoint
  directories; those legacy directories were later overwritten by Mistral.
- The training logs print 475 examples, but their category counts and
  eligible-query totals are not fully reconciled with the later corpus
  loader. The current ingestion implementation also differs from the early
  3,000-character setting. This is an evidence snapshot, not a claim that
  rerunning the current defaults exactly recreates historical training.
- The invalid-title metric in `stair.evaluate` checks the top-ranked title
  among answered queries. A valid title need not be a relevant answer.
- The benchmark snapshot contains derived identifiers and counts, not book
  passages or raw question files. Its revision and file hashes identify the
  upstream data used for the offline census. The character-based ToC token
  estimates are not exact Mistral tokenizer counts.

## Attribution and licensing

STAIR is the method of its original authors. The benchmark rebuild is
maintained by [SGK86 on Hugging Face](https://huggingface.co/datasets/SGK86/searchtome-subset).
The implementation is covered by the repository's MIT `LICENSE`. Upstream
models, benchmark material, and source textbooks retain their own terms.
