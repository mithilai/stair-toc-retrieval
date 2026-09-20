#!/bin/sh
# Wait for the running ablation to finish, then run the zero-training test.
while ! grep -q "best dev R@1" logs/whole-child-no-toc.log 2>/dev/null; do sleep 60; done
sleep 30
echo "=== ablation done, starting zero-shot test ==="
./.venv/Scripts/python.exe scripts/zeroshot_test.py --book whole-child --limit 200
