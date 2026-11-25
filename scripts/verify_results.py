"""Verify the packaged evidence offline, without models or GPU dependencies."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stair.evaluate import evaluate  # noqa: E402
from stair.significance import compare_systems  # noqa: E402


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_manifest() -> None:
    manifest = load("reproducibility/manifest.json")
    require(manifest["schema_version"] == 2, "Unsupported artifact manifest version")
    require(not manifest["manuscript_included"] and not manifest["video_included"],
            "The release must exclude the local manuscript and video")
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        require(path.resolve().is_relative_to(ROOT), "Manifest path escapes the archive")
        data = path.read_bytes()
        require(entry["normalization"] in ("none", "lf"),
                f"Unsupported normalization: {entry['path']}")
        if entry["normalization"] == "lf":
            data.decode("utf-8")
            data = data.replace(b"\r\n", b"\n")
        require(len(data) == entry["bytes"], f"Size mismatch: {entry['path']}")
        require(hashlib.sha256(data).hexdigest() == entry["sha256"],
                f"Checksum mismatch: {entry['path']}")
    print(f"Verified {len(manifest['files'])} file hashes")


def main() -> None:
    verify_manifest()
    snapshot = load("reproducibility/benchmark_snapshot.json")
    valid_titles = set(snapshot["whole_child_valid_titles"])
    predictions = load("results/whole-child_predictions.json")
    summary = load("results/whole-child_test.json")
    gold, runs = predictions["gold"], predictions["predictions"]
    require(len(gold) == summary["queries_evaluated"] == 610, "Unexpected test size")
    require(summary["model"] == "mistralai/Mistral-7B-Instruct-v0.2", "Wrong base model")
    require(set(gold.values()) <= valid_titles, "Gold labels absent from the title set")
    for name, ranked in runs.items():
        require(set(ranked) == set(gold), f"Incomplete query coverage: {name}")
        actual = evaluate(ranked, gold, valid_docids=valid_titles)
        for metric, value in actual.items():
            require(summary["metrics"][name][metric] == value,
                    f"Metric mismatch: {name}/{metric}")
        for metric in ("R@1", "R@3", "nDCG@3"):
            expected = round(actual[metric] * len(gold) / summary["published_test_size"], 2)
            require(summary["metrics"][name][f"{metric}_all"] == expected,
                    f"All-query metric mismatch: {name}/{metric}")
    for k in (1, 3):
        actual = compare_systems(runs, gold, k=k)
        require(actual == summary["significance"][f"R@{k}"], f"Paired-test mismatch at k={k}")
    print("Verified all 7B/BM25 test metrics, invalid-title rates, and six paired tests")

    audit = load("results/bm25_subset.json")
    for book, record in audit["books"].items():
        original = snapshot["books"][book]
        require(record["test_total"] == original["test_queries"], f"Test count mismatch: {book}")
        require(record["orphans"] == original["test_orphans"], f"Orphan count mismatch: {book}")
        answerable = record["test_total"] - record["orphans"]
        require(answerable == record["answerable"]["n_queries"], f"Answerable count mismatch: {book}")
        expected = round(record["answerable"]["R@1"] * answerable / record["test_total"], 2)
        require(expected == record["R@1_all"], f"BM25 denominator mismatch: {book}")
    for key, field in (("macro_R@1_answerable", "answerable"), ("macro_R@1_all", "R@1_all")):
        values = [b[field]["R@1"] if field == "answerable" else b[field]
                  for b in audit["books"].values()]
        require(round(sum(values) / len(values), 2) == audit[key], f"Macro average mismatch: {key}")
    print("Verified six-book audit counts and BM25 denominator/macro calculations")

    for variant, with_toc in (("stair", True), ("no-toc", False)):
        directory = f"checkpoints/whole-child-{variant}-Mistral-7B-Instruct-v0-2"
        history = load(f"{directory}/history.json")
        adapter = load(f"{directory}/adapter_config.json")
        rows = [json.loads(line) for line in (ROOT / f"logs/whole-child-{variant}.jsonl")
                .read_text(encoding="utf-8").splitlines() if line.strip()]
        require(history["model"] == adapter["base_model_name_or_path"] == summary["model"],
                f"Checkpoint model mismatch: {variant}")
        require(history["with_toc"] == with_toc, f"Checkpoint variant mismatch: {variant}")
        require(rows[-len(history["history"]):] == history["history"],
                f"Development log/history mismatch: {variant}")
        selected = max(history["history"], key=lambda row: row["dev_R@1"])
        require(selected["epoch"] == 3 and selected["dev_R@1"] == history["best_dev_R@1"],
                f"Selected checkpoint mismatch: {variant}")
        print(f"Verified {variant}: {len(history['history'])} completed epochs, selected epoch 3")

    subset = dict(list(gold.items())[:200])
    zero = load("results/whole-child_zeroshot.json")
    require(zero["queries"] == len(subset) == 200, "Unexpected zero-shot sample size")
    for name in runs:
        actual = evaluate(runs[name], subset, valid_docids=valid_titles)
        archived = zero["metrics"][f"trained-{name}(0.5B)"]
        require(actual == archived, f"Imported trained subset mismatch: {name}")
    trained = {name: runs[name] for name in ("STAIR", "no-ToC")}
    paired = compare_systems(trained, subset, k=1)[0]
    require(paired["observed_diff"] == 1.0 and paired["p_value"] == 0.8809,
            "Trained 200-query comparison mismatch")
    print("Verified trained 200-query scores and R@1 comparison: +1.00 pp, p=0.8809")
    print("Historical '(0.5B)' subset labels refer to the saved Mistral-7B predictions.")
    print("Qwen and zero-shot per-query outputs are absent; their recorded tests remain log-backed.")
    print("All available offline checks passed. Original-benchmark comparability is not claimed.")


if __name__ == "__main__":
    main()
