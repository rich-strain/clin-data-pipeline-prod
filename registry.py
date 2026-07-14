"""Stage 6 — model registry + data↔model lineage.

Pure-Python (no ML deps), so CI type-checks and tests it while the heavy
`train_runner.py` stays a local-only Lane 1 script. After a fine-tune,
`register_run()` appends an immutable registry entry that ties the model back to:

  - the exact **data snapshot** it was trained on — content hashes of the
    train/val/gold split files plus the frozen gold-set version, and
  - the exact **code** — the current git commit SHA,

so any committed adapter can be traced to precisely the data and code that
produced it (the lineage the build_spec's Stage 6 calls for), and the app/Stage 7
can read the selected version + best-epoch rationale off a committed file.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "training_results" / "model_registry.json"
DATA_DIR = ROOT / "data"
LINEAGE_FILES = {
    "train": DATA_DIR / "splits" / "train.jsonl",
    "val": DATA_DIR / "splits" / "val.jsonl",
    "gold": DATA_DIR / "gold" / "gold.jsonl",
}
GOLD_MANIFEST = DATA_DIR / "gold" / "gold_manifest.json"


def git_sha() -> str:
    """Current commit SHA, or 'unknown' outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapter_bytes(adapter_dir: Path) -> int:
    return sum(f.stat().st_size for f in Path(adapter_dir).rglob("*") if f.is_file())


def data_lineage() -> dict:
    """Content hashes of the data snapshot a model is derived from."""
    lineage: dict = {
        f"{name}_sha256": file_sha256(p) for name, p in LINEAGE_FILES.items() if p.exists()
    }
    if GOLD_MANIFEST.exists():
        lineage["gold_version"] = json.loads(GOLD_MANIFEST.read_text()).get("version")
    return lineage


def load_registry(path: Path = REGISTRY_PATH) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def register_run(
    *,
    base_model: str,
    lora_config: dict,
    metrics: dict,
    adapter_dir: Path,
    mlflow_run_id: str | None = None,
    path: Path = REGISTRY_PATH,
) -> dict:
    """Append a registry entry for a completed run and return it. `metrics` carries
    best_epoch + best/final losses + example counts (the best-epoch rationale)."""
    registry = load_registry(path)
    entry = {
        "version": f"v{len(registry) + 1}",
        "git_sha": git_sha(),
        "base_model": base_model,
        "lora_config": lora_config,
        "best_epoch": metrics["best_epoch"],
        "best_val_loss": metrics["best_val_loss"],
        "final_train_loss": metrics["final_train_loss"],
        "final_val_loss": metrics["final_val_loss"],
        "train_examples": metrics["train_examples"],
        "val_examples": metrics["val_examples"],
        "adapter_bytes": adapter_bytes(adapter_dir) if Path(adapter_dir).exists() else None,
        "mlflow_run_id": mlflow_run_id,
        "data_lineage": data_lineage(),
    }
    registry.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2) + "\n")
    return entry


def latest(path: Path = REGISTRY_PATH) -> dict | None:
    registry = load_registry(path)
    return registry[-1] if registry else None
