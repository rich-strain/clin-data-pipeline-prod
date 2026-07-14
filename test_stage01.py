"""Stage 0/1 verification (Working Plan step 2 verify gate).

Free, deterministic, CI-safe: exercises the derivation logic against the
committed immutable landing layer and asserts the committed reports/OMOP tables
reproduce exactly. This is the "free pipeline stage against committed inputs"
Lane 2 check for Stage 0/1 — no API, no network.

Run: python -m pytest test_stage01.py -v
"""

import json
from pathlib import Path

import pandas as pd

from generation.generate_fhir import generate_dataset
from generation.landing import group_by_patient, read_landing
from omop.etl import CDM_TABLES, fhir_to_omop
from terminology.bind import binding_report
from terminology.validate import validate_resources

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def _committed_landing() -> list[dict]:
    resources = read_landing(DATA / "landing")
    assert resources, "committed landing layer is missing — run `python run_stage01.py`"
    return resources


def test_rebuild_is_deterministic() -> None:
    a = generate_dataset(15, messy=True, seed=42)
    b = generate_dataset(15, messy=True, seed=42)
    assert json.dumps(a) == json.dumps(b), "seeded rebuild is not reproducible"


def test_committed_landing_all_base_fhir_valid() -> None:
    report = validate_resources(_committed_landing())
    assert report["invalid"] == 0, f"structural validation failures: {report['failures'][:3]}"
    assert report["us_core_profiled"] == report["total"], (
        "every resource should carry a US Core profile"
    )


def test_derivations_reproduce_committed_reports() -> None:
    resources = _committed_landing()
    committed_val = json.loads((DATA / "reports" / "validation.json").read_text())
    committed_bind = json.loads((DATA / "reports" / "terminology_binding.json").read_text())

    val = validate_resources(resources)
    assert (val["valid"], val["total"], val["us_core_profiled"]) == (
        committed_val["valid"],
        committed_val["total"],
        committed_val["us_core_profiled"],
    )

    bind = binding_report(resources)
    assert bind["overall"] == committed_bind["overall"]
    assert bind["by_system"] == committed_bind["by_system"]


def test_omop_row_counts_match_committed() -> None:
    tables = fhir_to_omop(_committed_landing())
    for name in CDM_TABLES:
        committed = pd.read_csv(DATA / "omop" / f"{name}.csv")
        assert len(tables[name]) == len(committed), f"{name} row count drifted from committed"


def test_conditions_are_dual_coded_and_bp_is_a_panel() -> None:
    records = group_by_patient(_committed_landing())
    for rec in records.values():
        for cond in rec["conditions"]:
            systems = {c["system"] for c in cond["code"]["coding"]}
            assert (
                "http://snomed.info/sct" in systems
                and "http://hl7.org/fhir/sid/icd-10-cm" in systems
            )
        bp = [
            o
            for o in rec["observations"]
            if o.get("code", {}).get("coding", [{}])[0].get("code") == "85354-9"
        ]
        for panel in bp:
            comp_codes = {c["code"]["coding"][0]["code"] for c in panel["component"]}
            assert comp_codes == {"8480-6", "8462-4"}, (
                "BP panel must have systolic + diastolic components"
            )


def test_measurements_all_have_verified_standard_concepts() -> None:
    m = pd.read_csv(DATA / "omop" / "measurement.csv")
    assert (m["measurement_concept_id"] == 0).sum() == 0, (
        "every measurement should map to a real LOINC concept"
    )
