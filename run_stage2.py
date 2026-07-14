"""Stage 2 build (local; needs presidio-analyzer + the spaCy model).

Reads the committed FHIR landing, generates PHI-laden clinical notes, then
de-identifies both modalities: structured redaction + interval-preserving date
shift for the FHIR, and layered NLP de-id (Presidio + known-identifier removal)
for the notes. Measures Presidio's recall against ground-truth PHI and runs the
per-patient leakage check. Writes every committed artifact the De-id page reads.

Run:  python run_stage2.py     (outputs are committed; CI never re-runs this)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from deid.dateshift import shift_resources
from deid.freetext import build_analyzer, deidentify_note, measure_recall
from deid.leakage import leakage_check
from deid.redact import redact_resources
from generation.generate_notes import generate_notes
from generation.landing import read_landing

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def _aggregate_recall(per_note: list[dict]) -> dict:
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"caught": 0, "total": 0})
    missed: list[dict] = []
    for r in per_note:
        for t, v in r["by_type"].items():
            by_type[t]["caught"] += v["caught"]
            by_type[t]["total"] += v["total"]
        missed.extend(r["missed"])
    caught = sum(v["caught"] for v in by_type.values())
    total = sum(v["total"] for v in by_type.values())
    # keep a bounded, representative sample of misses for the app
    sample = missed[:25]
    return {
        "caught": caught,
        "total": total,
        "recall": round(caught / total, 4) if total else 0.0,
        "by_type": dict(by_type),
        "missed_sample": sample,
        "engine": "Microsoft Presidio (spaCy en_core_web_lg) + custom MRN recognizer",
    }


def main() -> None:
    resources = read_landing(DATA / "landing")
    notes = generate_notes(resources, seed=42)

    # 1) Raw notes + ground-truth PHI labels (the labeled sample).
    notes_dir = DATA / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    with (
        (notes_dir / "raw_notes.jsonl").open("w") as f,
        (notes_dir / "phi_labels.jsonl").open("w") as lf,
    ):
        for n in notes:
            f.write(
                json.dumps({k: n[k] for k in ("patient_id", "note_id", "visit_date", "text")})
                + "\n"
            )
            lf.write(json.dumps({"note_id": n["note_id"], "phi_spans": n["phi_spans"]}) + "\n")

    # 2) Structured de-id: interval-preserving date shift, then identifier redaction.
    deid_resources = redact_resources(shift_resources(resources))
    deid_dir = DATA / "deid"
    deid_dir.mkdir(parents=True, exist_ok=True)
    with (deid_dir / "resources_deid.ndjson").open("w") as f:
        for r in deid_resources:
            f.write(json.dumps(r) + "\n")

    # 3) Free-text de-id (Presidio, layered) + per-note recall.
    analyzer = build_analyzer()
    per_note_recall: list[dict] = []
    deid_notes_by_patient: dict[str, str] = {}
    with (deid_dir / "notes_deid.jsonl").open("w") as f:
        for n in notes:
            known = [(s["start"], s["end"], s["type"]) for s in n["phi_spans"]]
            redacted, detected = deidentify_note(n["text"], known, n["patient_id"], analyzer)
            deid_notes_by_patient[n["patient_id"]] = redacted
            per_note_recall.append(measure_recall(n["phi_spans"], detected))
            f.write(
                json.dumps(
                    {"patient_id": n["patient_id"], "note_id": n["note_id"], "text": redacted}
                )
                + "\n"
            )

    # 4) Reports: recall + leakage.
    reports = DATA / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    recall = _aggregate_recall(per_note_recall)
    (reports / "deid_recall.json").write_text(json.dumps(recall, indent=2))
    leakage = leakage_check(resources, deid_resources, deid_notes_by_patient)
    (reports / "deid_leakage.json").write_text(json.dumps(leakage, indent=2))

    print(f"Notes: {len(notes)} | labeled PHI spans: {sum(len(n['phi_spans']) for n in notes)}")
    print(f"Presidio recall: {recall['recall'] * 100:.1f}% ({recall['caught']}/{recall['total']})")
    print("  by type:", {t: f"{v['caught']}/{v['total']}" for t, v in recall["by_type"].items()})
    print(f"Leakage check: {leakage['total_leaks']} leaks across {leakage['patients']} patients")


if __name__ == "__main__":
    main()
