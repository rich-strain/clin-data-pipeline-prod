"""Stage 4 verification (Working Plan step 5 verify gate).

Free, deterministic, CI-safe: no Anthropic SDK, no network. Exercises the
curation logic directly and validates the committed curated artifacts + DQ gate.

Run: python -m pytest test_stage4.py -v
"""

import json
from pathlib import Path

from curation import dq, normalize, rebalance, redact

ROOT = Path(__file__).parent
CURATED = ROOT / "data" / "curated"


# --- normalize -------------------------------------------------------------


def test_normalize_strips_date_contamination_from_diagnosis() -> None:
    name, matched = normalize.normalize_diagnosis_name("Migraine, diagnosed 2017-07-17")
    assert (name, matched) == ("Migraine", True)
    name, matched = normalize.normalize_diagnosis_name(
        "Major depression, single episode, first diagnosed 2019-01-30"
    )
    assert (name, matched) == ("Major depression, single episode", True)


def test_normalize_maps_dropped_qualifier_and_take_prefix() -> None:
    name, matched = normalize.normalize_diagnosis_name("Type 2 diabetes mellitus")
    assert (name, matched) == ("Type 2 diabetes mellitus without complication", True)
    # Haiku drops the imperative "Take " — still the canonical dosage.
    dosage, matched = normalize.normalize_dosage(
        "Amlodipine 5 MG Oral Tablet", "5 mg by mouth once daily"
    )
    assert (dosage, matched) == ("Take 5 mg by mouth once daily", True)


def test_normalize_converts_alt_units_and_keeps_bp_compound() -> None:
    # 154.3 lb -> kg, canonicalized unit.
    name, value, unit, matched = normalize.normalize_vital("Body weight", "154.3", "lb")
    assert name == "Body weight" and unit == "kg" and matched
    assert isinstance(value, float) and abs(value - 70.0) < 0.2
    # Blood pressure stays a "sys/dia" string.
    assert normalize.normalize_vital("BP", "120/80", "mmHg") == (
        "Blood pressure",
        "120/80",
        "mm[Hg]",
        True,
    )


def test_committed_normalized_diagnoses_are_all_canonical() -> None:
    canon = set(normalize.CANONICAL_CONDITIONS)
    recs = [
        json.loads(line)
        for line in (CURATED / "normalized.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert recs, "committed normalized.jsonl missing — run `python run_stage4.py`"
    names = {d["name"] for r in recs for d in r["diagnoses"]}
    assert names <= canon, f"non-canonical diagnosis names survived: {names - canon}"


# --- rebalance -------------------------------------------------------------


def test_rebalance_oversamples_to_the_max_category_and_marks_duplicates() -> None:
    records = [
        json.loads(line)
        for line in (CURATED / "normalized.jsonl").read_text().splitlines()
        if line.strip()
    ]
    augmented, duplicates = rebalance.rebalance_records(records)
    before = rebalance.category_counts(records)
    after = rebalance.category_counts(augmented)
    target = max(before.values())
    # Every category that started >0 reaches the target; duplicates are flagged.
    for cat, count in before.items():
        if count > 0:
            assert after[cat] >= target, f"{cat} not brought up to target"
    assert all("rebalance_duplicate_of" in d for d in duplicates)


# --- redact (leakage assertion) --------------------------------------------


def test_redact_flags_planted_phi_and_passes_clean() -> None:
    clean = {"note_id": "n1", "diagnoses": [{"name": "Migraine"}], "medications": [], "vitals": []}
    assert redact.find_leaks(clean) == []
    leaky = {
        "note_id": "n2",
        "diagnoses": [{"name": "Migraine, diagnosed 2017-07-17"}],
        "medications": [],
        "vitals": [],
    }
    kinds = {leak["kind"] for leak in redact.find_leaks(leaky)}
    assert "date" in kinds


# --- DQ gate ---------------------------------------------------------------


def _committed_curated() -> list[dict]:
    path = CURATED / "synthesized.jsonl"
    assert path.exists(), "committed synthesized.jsonl missing — run `python run_stage4.py`"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_dq_gate_passes_on_committed_curated_data() -> None:
    report = dq.validate_records(_committed_curated())
    assert report["passed"], f"DQ gate failed on committed data: {report['tables']}"


def test_dq_gate_actually_fails_on_violations() -> None:
    """A gate that can't fail is worthless — plant each violation family."""
    bad = json.loads(json.dumps(_committed_curated()[0]))
    bad["confidence"] = 1.5  # out of [0,1]
    bad["diagnoses"] = [{"name": "Made-up disease"}]  # not in vocabulary
    bad["vitals"] = [{"name": "Heart rate", "value": 999.0, "unit": "/min"}]  # out of bounds
    report = dq.validate_records([bad])
    assert not report["passed"]
    failed_tables = {name for name, e in report["tables"].items() if not e["passed"]}
    assert {"records", "diagnoses", "numeric_vitals"} <= failed_tables


def test_committed_dq_report_is_green() -> None:
    report = json.loads((ROOT / "data" / "reports" / "dq_gate.json").read_text())
    assert report["passed"] and all(e["passed"] for e in report["tables"].values())
