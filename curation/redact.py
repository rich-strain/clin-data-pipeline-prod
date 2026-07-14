"""Stage 4 — PHI leakage assertion over normalized records (defense in depth).

Unlike the sibling repos, this pipeline **de-identifies upstream in Stage 2**:
extraction (Stage 3) runs on already-de-identified note text, so the extracted
records carry no PHI fields to strip here. Rather than fake a redaction that has
already happened, this sub-step is an honest *assertion*: it scans the curated
free-text fields for any residual PHI pattern (a raw date, an MRN, a leftover
`[PLACEHOLDER]` token, a "Dr. Name" provider) and FAILS the pipeline if one
appears — then passes the records through unchanged.

The synthetic `patient_id`/`note_id` are kept: they are resource ids (not real
PHI) and Stage 5's group-aware split needs `patient_id` to keep a patient's
records — and their rebalance duplicates — on the same side of the split.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NORMALIZED_PATH = DATA_DIR / "curated" / "normalized.jsonl"
REDACTED_PATH = DATA_DIR / "curated" / "redacted.jsonl"

# Patterns that must NOT appear in any curated free-text field. (Vital *values*
# like "120/80" are exempt from the date check — see _free_text below.)
_PHI_PATTERNS = {
    "date": re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    "mrn": re.compile(r"\bMRN\d+\b", re.IGNORECASE),
    "placeholder": re.compile(r"\[[A-Z_]+\]"),
    "provider": re.compile(r"\bDr\.\s+[A-Z][a-z]+"),
}


def _free_text_fields(record: dict) -> list[str]:
    """The human-readable strings a PHI leak could hide in (not numeric vitals)."""
    fields = [d["name"] for d in record["diagnoses"]]
    for m in record["medications"]:
        fields.append(m["name"])
        if m.get("dosage"):
            fields.append(m["dosage"])
    fields += [v["name"] for v in record["vitals"]]
    return fields


def find_leaks(record: dict) -> list[dict]:
    """PHI-pattern hits in a record's free-text fields, empty if clean."""
    leaks = []
    for text in _free_text_fields(record):
        for kind, pattern in _PHI_PATTERNS.items():
            if pattern.search(text):
                leaks.append({"kind": kind, "text": text, "note_id": record.get("note_id")})
    return leaks


def read_records(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4 PHI leakage assertion (de-id is upstream)."
    )
    parser.add_argument("--in", dest="in_path", type=Path, default=NORMALIZED_PATH)
    parser.add_argument("--out", type=Path, default=REDACTED_PATH)
    args = parser.parse_args()

    records = list(read_records(args.in_path))
    all_leaks = [leak for r in records for leak in find_leaks(r)]
    if all_leaks:
        raise SystemExit(
            f"PHI LEAK in curated records ({len(all_leaks)}): {all_leaks[:5]} — "
            f"upstream de-id (Stage 2) or normalization regressed."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"Leakage assertion passed: 0 PHI patterns across {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
