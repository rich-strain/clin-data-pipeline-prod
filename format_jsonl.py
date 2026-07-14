"""Stage 5 — format split curated records into instruction/response JSONL.

Takes `split.py`'s `data/curated/split_{train,val,test}.jsonl` and emits the
fine-tuning-ready `data/splits/{train,val,test}.jsonl`: one
`{"instruction": ..., "response": ...}` object per line.

**Instruction = the already-de-identified note text.** Unlike the sibling repos —
which extracted from RAW notes and had to redact at format time (find-and-replace
known PHI strings + shift dates) — this pipeline de-identifies UPSTREAM in Stage 2.
The notes in `data/deid/notes_deid.jsonl` are already redacted ([PATIENT_NAME],
[MRN], shifted dates), and Stage 3 extracted from exactly those. So formatting
here is a straight join on `note_id`, not a second redaction pass. A rebalance
duplicate shares its original's `note_id` (rebalance copies it verbatim), so its
instruction is the same note as its original — consistent with them being
near-identical examples by design.

**Response = the curated clinical fields as a JSON string** (diagnoses /
medications / vitals). A string, not a nested object, because fine-tuning trains
the model to generate the serialized JSON it must emit at inference time. Pipeline
bookkeeping (patient_id, confidence, provenance, rebalance tags) is dropped — a
note-extraction model should learn to predict the clinical fields, not metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from split import SPLIT_PATHS, read_records

DATA_DIR = Path(__file__).resolve().parent / "data"
DEID_NOTES_PATH = DATA_DIR / "deid" / "notes_deid.jsonl"
FORMATTED_PATHS = {name: DATA_DIR / "splits" / f"{name}.jsonl" for name in SPLIT_PATHS}

RESPONSE_FIELDS = ("diagnoses", "medications", "vitals")


def load_notes(path: Path) -> dict[str, str]:
    """note_id -> de-identified note text."""
    return {n["note_id"]: n["text"] for n in read_records(path)}


def build_response(record: dict) -> str:
    return json.dumps({field: record[field] for field in RESPONSE_FIELDS})


def format_record(record: dict, notes_by_id: dict[str, str]) -> dict:
    note_text = notes_by_id.get(record["note_id"])
    if note_text is None:
        raise KeyError(f"no de-identified note for note_id {record['note_id']!r}")
    return {"instruction": note_text, "response": build_response(record)}


def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def format_split(name: str, notes_by_id: dict[str, str]) -> list[dict]:
    records = list(read_records(SPLIT_PATHS[name]))
    return [format_record(r, notes_by_id) for r in records]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5 format splits into instruction/response.")
    parser.add_argument("--notes", type=Path, default=DEID_NOTES_PATH)
    args = parser.parse_args()

    notes_by_id = load_notes(args.notes)
    for name in SPLIT_PATHS:
        examples = format_split(name, notes_by_id)
        write_jsonl(examples, FORMATTED_PATHS[name])
        print(f"  {name:5} {len(examples):>4} examples -> {FORMATTED_PATHS[name]}")


if __name__ == "__main__":
    main()
