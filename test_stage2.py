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

from deid.dateshift import FIELD_CATEGORY, patient_offset, shift_resources
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


def test_date_shift_preserves_intervals_within_category_and_is_nonzero() -> None:
    raw = _raw()
    shifted = shift_resources(raw)
    # Deltas keyed by (patient, category): every date in a category moves by the
    # same amount (interval-preserving) and never by zero.
    deltas: dict[tuple[str, str], set[int]] = {}
    for r_raw, r_shift in zip(raw, shifted, strict=True):
        pid = (
            r_raw["id"]
            if r_raw["resourceType"] == "Patient"
            else r_raw.get("subject", {}).get("reference", "").split("/")[-1]
        )
        for field, category in FIELD_CATEGORY.items():
            if isinstance(r_raw.get(field), str):
                d = (date.fromisoformat(r_shift[field]) - date.fromisoformat(r_raw[field])).days
                deltas.setdefault((pid, category), set()).add(d)
    for (pid, category), ds in deltas.items():
        assert len(ds) == 1, f"{pid}/{category} shifted inconsistently: {ds}"
        assert 0 not in ds, f"{pid}/{category} has a zero shift (would leak raw dates)"


def test_dob_and_visit_offsets_are_independent() -> None:
    # DOB shifts independently of visit dates: for essentially every patient the
    # two category offsets differ (a shared offset would let one recovered date
    # unshift the rest).
    pids = list(group_by_patient(_raw()).keys())
    differ = sum(patient_offset(p, "dob") != patient_offset(p, "visit") for p in pids)
    assert differ >= len(pids) - 1, "DOB and visit offsets should be independent for ~all patients"


def test_no_shifted_date_is_in_the_future() -> None:
    today = date.today()
    for r in shift_resources(_raw()):
        for field in FIELD_CATEGORY:
            if isinstance(r.get(field), str):
                assert date.fromisoformat(r[field]) <= today, f"{field} shifted into the future"


def test_all_patient_offsets_nonzero() -> None:
    pids = list(group_by_patient(_raw()).keys())
    assert all(patient_offset(p, c) != 0 for p in pids for c in ("dob", "visit"))


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
