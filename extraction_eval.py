"""Stage 3 — extraction eval: Haiku's extractions vs the source FHIR facts.

The teacher-side check that complements Stage 7's student-side eval. The notes
were generated FROM committed FHIR resources, so those resources are the ground
truth for what each note should contain — nothing had to be labeled. This scores
the curated extraction targets (Haiku output, run through the same Stage 4
normalization) against them, reusing the exact TP/FP/FN + hallucination scorer
Stage 7 uses (`eval_metrics.score_field`).

Free — no API, no model: a pure join over committed artifacts (like provenance.py
/ analytics.py). Scope: diagnoses + medications by canonical name. Vitals are out
of scope (value-closeness judgment, same deferral as Stage 7). Raw formatting
quality (date-contaminated names, odd dosages) is measured separately by Stage 4's
normalize_metrics.json — here the prediction is the normalized target that
actually trains the model.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import eval_metrics
from curation.normalize import normalize_dosage, normalize_record

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
EXTRACTED_PATH = DATA_DIR / "extracted" / "extractions.jsonl"
CONDITION_PATH = DATA_DIR / "landing" / "Condition.ndjson"
MEDICATION_PATH = DATA_DIR / "landing" / "MedicationRequest.ndjson"
REPORT_PATH = DATA_DIR / "reports" / "extraction_eval.json"

MAX_MISMATCH_EXAMPLES = 10


def _read_ndjson(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _patient_id(resource: dict) -> str:
    return resource["subject"]["reference"].split("/", 1)[1]


def _condition_name(cond: dict) -> str:
    return cond["code"].get("text") or cond["code"]["coding"][0].get("display", "")


def ground_truth(conditions: list[dict], medications: list[dict]) -> tuple[dict, dict]:
    """Per-patient ground truth from the FHIR facts that generated the notes:
    diagnosis names (set), and medications as {name: canonical_dosage_or_None}
    (the FHIR sig, canonicalized the same way the extraction is)."""
    gt_dx: dict[str, set] = defaultdict(set)
    gt_med: dict[str, dict] = defaultdict(dict)
    for c in conditions:
        gt_dx[_patient_id(c)].add(_condition_name(c))
    for m in medications:
        name = m["medicationCodeableConcept"].get("text", "")
        sig = (m.get("dosageInstruction") or [{}])[0].get("text")
        gt_med[_patient_id(m)][name] = normalize_dosage(name, sig)[0]
    return gt_dx, gt_med


def _score_dosage(extractions: list[dict], gt_med: dict) -> dict:
    """Exact-match on the value the name check can't catch — this is where the
    extraction genuinely slips. For each medication matched by name, compare the
    normalized dosage to the source sig (both canonicalized). Three failure kinds:
    `wrong` (different value), `missing` (source had a sig, model produced none),
    and `hallucinated` (source had NO sig, model invented one — e.g. echoing the
    strength from the drug name). A source-None ↔ predicted-None pair agrees."""
    matched = total = hallucinated = 0
    mismatches: list[dict] = []
    for e in extractions:
        normalized, _ = normalize_record(e)
        gt = gt_med.get(e["patient_id"], {})
        for m in normalized["medications"]:
            name, dose = m["name"], m.get("dosage")
            if name not in gt:
                continue  # spurious drug — already an FP in the name metric
            total += 1
            expected = gt[name]  # canonical dosage, or None if the source had no sig
            if dose == expected:
                matched += 1
                continue
            if expected is None:
                hallucinated += 1
                kind = "hallucinated"
            elif dose is None:
                kind = "missing"
            else:
                kind = "wrong"
            if len(mismatches) < MAX_MISMATCH_EXAMPLES:
                mismatches.append(
                    {"medication": name, "predicted": dose, "expected": expected, "type": kind}
                )
    return {
        "exact_match": matched,
        "total": total,
        "accuracy": (matched / total) if total else 0.0,
        "hallucinated": hallucinated,
        "mismatch_examples": mismatches,
    }


def evaluate_extractions(extractions: list[dict], gt_dx: dict, gt_med: dict) -> dict:
    """Score the normalized extraction against the source FHIR facts. Pure."""
    dx_pairs, med_pairs = [], []
    for e in extractions:
        normalized, _ = normalize_record(e)
        pid = e["patient_id"]
        dx_pairs.append(({"diagnoses": [{"name": n} for n in gt_dx.get(pid, set())]}, normalized))
        med_pairs.append(({"medications": [{"name": n} for n in gt_med.get(pid, {})]}, normalized))

    return {
        "n_records": len(extractions),
        "ground_truth": "source FHIR Condition / MedicationRequest resources (generated each note)",
        "prediction": "Haiku extraction, normalized (curation.normalize) — the curated targets",
        "diagnosis": eval_metrics.score_field(
            dx_pairs, eval_metrics.CANONICAL_DIAGNOSES, "diagnoses"
        ),
        "medication": eval_metrics.score_field(
            med_pairs, eval_metrics.CANONICAL_MEDS, "medications"
        ),
        "dosage": _score_dosage(extractions, gt_med),
        "scope_note": (
            "diagnoses + medications by canonical name, plus medication dosage "
            "exact-match. Vitals out of scope (value-closeness). Raw formatting "
            "quality is tracked in normalize_metrics.json."
        ),
    }


def build_report() -> dict:
    extractions = _read_ndjson(EXTRACTED_PATH)
    gt_dx, gt_med = ground_truth(_read_ndjson(CONDITION_PATH), _read_ndjson(MEDICATION_PATH))
    return evaluate_extractions(extractions, gt_dx, gt_med)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 extraction eval (Haiku vs source FHIR).")
    parser.add_argument("--report-out", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    report = build_report()
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")

    for field in ("diagnosis", "medication"):
        s = report[field]
        prf = f"{s['micro_precision']:.3f}/{s['micro_recall']:.3f}/{s['micro_f1']:.3f}"
        print(
            f"  {field:10} P/R/F1 {prf}  TP={s['tp']} FP={s['fp']} "
            f"FN={s['fn']} hallucinations={s['non_canonical_count']}"
        )
    d = report["dosage"]
    print(
        f"  dosage     exact-match {d['exact_match']}/{d['total']} "
        f"({d['accuracy']:.1%}), {d['hallucinated']} hallucinated, "
        f"{len(d['mismatch_examples'])} mismatch example(s)"
    )
    print(f"Wrote extraction eval ({report['n_records']} records) to {args.report_out}")


if __name__ == "__main__":
    main()
