"""Stage 2 verification (Working Plan step 3 verify gate).

Free, deterministic, CI-safe: does NOT import Presidio or re-run the NLP de-id
(that's a local step whose outputs are committed). Instead it re-checks the
committed de-id artifacts and the deterministic de-id logic:
  - per-patient leakage == 0 (the actual gate),
  - the date shift preserves intra-patient intervals and is never zero,
  - structured redaction removes direct identifiers,
  - the committed recall report is well-formed and honest.

Run: python -m pytest test_stage2.py -v
"""

import json
from datetime import date
from pathlib import Path

from deid.dateshift import DATE_FIELDS, patient_offset, shift_resources
from deid.leakage import leakage_check
from deid.redact import RESEARCH_ID_SYSTEM, pseudonym, redact_resources
from generation.landing import group_by_patient, read_landing

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def _raw() -> list[dict]:
    resources = read_landing(DATA / "landing")
    assert resources, "committed landing missing — run `python run_stage01.py`"
    return resources


def _committed_deid_resources() -> list[dict]:
    path = DATA / "deid" / "resources_deid.ndjson"
    assert path.exists(), "committed de-id artifacts missing — run `python run_stage2.py`"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _committed_deid_notes() -> dict[str, str]:
    path = DATA / "deid" / "notes_deid.jsonl"
    return {
        json.loads(line)["patient_id"]: json.loads(line)["text"]
        for line in path.read_text().splitlines()
        if line.strip()
    }


def test_committed_leakage_is_zero() -> None:
    report = json.loads((DATA / "reports" / "deid_leakage.json").read_text())
    assert report["total_leaks"] == 0, (
        f"committed leakage report shows leaks: {report['leaked_examples'][:3]}"
    )


def test_leakage_reproduces_zero_from_committed_artifacts() -> None:
    result = leakage_check(_raw(), _committed_deid_resources(), _committed_deid_notes())
    assert result["total_leaks"] == 0, result["leaked_examples"][:3]


def test_date_shift_preserves_intervals_and_is_nonzero() -> None:
    raw = _raw()
    shifted = shift_resources(raw)
    per_patient_deltas: dict[str, set[int]] = {}
    for r_raw, r_shift in zip(raw, shifted, strict=True):
        pid = (
            r_raw["id"]
            if r_raw["resourceType"] == "Patient"
            else r_raw.get("subject", {}).get("reference", "").split("/")[-1]
        )
        for field in DATE_FIELDS:
            if isinstance(r_raw.get(field), str):
                delta = (date.fromisoformat(r_shift[field]) - date.fromisoformat(r_raw[field])).days
                per_patient_deltas.setdefault(pid, set()).add(delta)
    for pid, deltas in per_patient_deltas.items():
        assert len(deltas) == 1, f"patient {pid} dates shifted inconsistently: {deltas}"
        assert 0 not in deltas, f"patient {pid} has a zero shift (would leak raw dates)"


def test_all_patient_offsets_nonzero() -> None:
    pids = list(group_by_patient(_raw()).keys())
    assert all(patient_offset(p) != 0 for p in pids)


def test_structured_redaction_removes_direct_identifiers() -> None:
    deid = redact_resources(_raw())
    for r in deid:
        if r["resourceType"] != "Patient":
            continue
        assert "name" not in r, "patient name must be removed"
        assert r["identifier"][0]["system"] == RESEARCH_ID_SYSTEM
        assert r["identifier"][0]["value"] == pseudonym(r["id"])
        for addr in r.get("address", []):
            assert "line" not in addr and "city" not in addr, "street/city must be dropped"


def test_recall_report_is_well_formed() -> None:
    recall = json.loads((DATA / "reports" / "deid_recall.json").read_text())
    assert recall["total"] > 0 and 0.0 <= recall["recall"] <= 1.0
    assert recall["caught"] <= recall["total"]
    assert recall["by_type"], "recall should break down by PHI type"
