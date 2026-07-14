"""Stage 9 — pipeline run/lineage log (Tier 1, real, displayed).

Builds a provenance table over the committed pipeline artifacts: for each stage,
the output it produced, that output's content hash, and its record count. This is
a *real* lineage record (content-addressed, reproducible), not a fabricated run
log — the honest counterpart to the Tier-2 access/audit log that can only be
documented on synthetic data.

Timestamps are deliberately omitted from the committed artifact (they'd make it
non-reproducible); the app pairs this with the live git SHA + the model
registry's version to anchor "which code + data" produced the run. Emits
data/reports/provenance.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PROVENANCE_PATH = DATA / "reports" / "provenance.json"

# (stage, output artifact) in pipeline order. Each is a committed derivation.
STAGE_OUTPUTS: list[tuple[str, str]] = [
    ("0/1 Ingestion", "data/landing/Patient.ndjson"),
    ("0/1 Canonical (FHIR)", "data/canonical/fhir_bundles.json"),
    ("0/1 OMOP CDM", "data/omop/condition_occurrence.csv"),
    ("2 De-identification", "data/deid/notes_deid.jsonl"),
    ("3 Extraction", "data/extracted/extractions.jsonl"),
    ("4 Curation (gated)", "data/curated/synthesized.jsonl"),
    ("5 Split — train", "data/splits/train.jsonl"),
    ("5 Split — val", "data/splits/val.jsonl"),
    ("5 Gold (frozen)", "data/gold/gold.jsonl"),
    ("6 Training (adapter)", "training_results/adapter/adapter_model.safetensors"),
    ("7 Evaluation", "training_results/eval_report.json"),
]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record_count(path: Path) -> int | None:
    """Best-effort record count by artifact type (lines / list length / rows)."""
    suffix = path.suffix
    if suffix in {".jsonl", ".ndjson"}:
        return sum(1 for line in path.read_text().splitlines() if line.strip())
    if suffix == ".csv":
        return max(sum(1 for _ in path.read_text().splitlines()) - 1, 0)  # minus header
    if suffix == ".json":
        obj = json.loads(path.read_text())
        return len(obj) if isinstance(obj, list) else None
    return None  # binary (adapter) etc.


def build_provenance() -> list[dict]:
    rows = []
    for stage, rel in STAGE_OUTPUTS:
        path = ROOT / rel
        if not path.exists():
            continue
        rows.append(
            {
                "stage": stage,
                "artifact": rel,
                "sha256": file_sha256(path),
                "record_count": _record_count(path),
            }
        )
    return rows


def main() -> None:
    rows = build_provenance()
    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE_PATH.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote provenance log ({len(rows)} stages) to {PROVENANCE_PATH}")
    for r in rows:
        print(f"  {r['stage']:24} {str(r['record_count']):>5}  {r['sha256'][:12]}  {r['artifact']}")


if __name__ == "__main__":
    main()
