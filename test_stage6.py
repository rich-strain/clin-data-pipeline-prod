"""Stage 6 verification (registry + lineage half).

Free, deterministic, CI-safe: exercises the pure-Python model registry / data↔
model lineage. The training itself (`train_runner.py`, torch/MPS) is Lane 1 and
not covered here; this locks down the lifecycle bookkeeping that ties a committed
adapter to its exact data snapshot + git SHA.

Run: python -m pytest test_stage6.py -v
"""

import json
from pathlib import Path

import registry

ROOT = Path(__file__).parent


def test_data_lineage_hashes_the_committed_split_snapshot() -> None:
    lineage = registry.data_lineage()
    # The splits exist (Stage 5), so their content hashes must be recorded.
    assert {"train_sha256", "val_sha256", "gold_sha256"} <= set(lineage)
    assert all(len(lineage[k]) == 64 for k in ("train_sha256", "val_sha256", "gold_sha256"))
    # Lineage is exact: the gold hash matches the committed gold file.
    assert lineage["gold_sha256"] == registry.file_sha256(registry.LINEAGE_FILES["gold"])
    assert lineage["gold_version"] == json.loads(registry.GOLD_MANIFEST.read_text())["version"]


def test_git_sha_is_a_sha_or_unknown() -> None:
    sha = registry.git_sha()
    assert sha == "unknown" or (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha))


def _metrics() -> dict:
    return {
        "best_epoch": 3,
        "best_val_loss": 0.42,
        "final_train_loss": 0.31,
        "final_val_loss": 0.45,
        "train_examples": 98,
        "val_examples": 12,
    }


def test_register_run_appends_versioned_entry_with_lineage(tmp_path) -> None:
    path = tmp_path / "model_registry.json"
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"x" * 128)

    entry = registry.register_run(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        lora_config={"r": 8},
        metrics=_metrics(),
        adapter_dir=adapter,
        mlflow_run_id="abc123",
        path=path,
    )
    assert entry["version"] == "v1"
    assert entry["best_epoch"] == 3 and entry["mlflow_run_id"] == "abc123"
    assert entry["adapter_bytes"] == 128
    assert "train_sha256" in entry["data_lineage"]  # ties model -> data snapshot

    # A second run appends v2, and latest() returns it.
    second = registry.register_run(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        lora_config={"r": 8},
        metrics=_metrics(),
        adapter_dir=adapter,
        path=path,
    )
    assert second["version"] == "v2"
    assert len(registry.load_registry(path)) == 2
    newest = registry.latest(path)
    assert newest is not None and newest["version"] == "v2"
