#!/bin/sh
set -e
PY=./.venv/Scripts/python.exe
echo "=== [1/2] 7B test-set evaluation (610 queries, STAIR vs no-ToC vs BM25) ==="
$PY scripts/evaluate_run.py --book whole-child --model mistralai/Mistral-7B-Instruct-v0.2
echo "=== [2/2] zero-training test ==="
$PY scripts/zeroshot_test.py --book whole-child --limit 200
echo "=== chain complete ==="
