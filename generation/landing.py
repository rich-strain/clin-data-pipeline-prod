"""Immutable raw landing layer (Stage 0).

Writes the generated resources as NDJSON — one resource per line, partitioned
by resource type — mirroring what a real FHIR Bulk Data `$export` emits. This
is the authoritative record: "exactly what the source sent at time T." Every
downstream artifact (OMOP CDM, feature views, notes) is a reproducible
derivation FROM this layer, never a second source of truth.

A real landing layer is write-once/immutable (append-only object storage). Here
that discipline is represented by regenerating deterministically from a fixed
seed — the committed NDJSON is the fixed point everything else rebuilds against.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def write_landing(dataset: list[list[dict]], landing_dir: Path) -> dict[str, int]:
    """Write per-resource-type NDJSON. Returns {resourceType: line_count}."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for patient_resources in dataset:
        for resource in patient_resources:
            by_type[resource["resourceType"]].append(resource)

    landing_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for rtype, resources in sorted(by_type.items()):
        path = landing_dir / f"{rtype}.ndjson"
        with path.open("w") as f:
            for r in resources:
                f.write(json.dumps(r) + "\n")
        counts[rtype] = len(resources)
    return counts


def read_landing(landing_dir: Path) -> list[dict]:
    """Read all resources back from the landing NDJSON files."""
    resources: list[dict] = []
    for path in sorted(landing_dir.glob("*.ndjson")):
        with path.open() as f:
            resources.extend(json.loads(line) for line in f if line.strip())
    return resources


def group_by_patient(resources: list[dict]) -> dict[str, dict]:
    """Regroup a flat resource list into per-patient records keyed by patient id.

    Each record: {"patient", "conditions", "observations", "medications"}.
    """
    records: dict[str, dict] = {}

    def _pid(resource: dict) -> str | None:
        if resource["resourceType"] == "Patient":
            return resource["id"]
        ref = resource.get("subject", {}).get("reference", "")
        return ref.split("/", 1)[1] if "/" in ref else None

    # Patients first so every referenced record exists.
    for r in resources:
        if r["resourceType"] == "Patient":
            records[r["id"]] = {
                "patient": r,
                "conditions": [],
                "observations": [],
                "medications": [],
            }

    bucket = {
        "Condition": "conditions",
        "Observation": "observations",
        "MedicationRequest": "medications",
    }
    for r in resources:
        if r["resourceType"] == "Patient":
            continue
        pid = _pid(r)
        key = bucket.get(r["resourceType"])
        if pid in records and key:
            records[pid][key].append(r)
    return records
