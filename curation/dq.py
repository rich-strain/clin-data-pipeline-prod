"""Stage 4 — declarative data-quality gate (Pandera) that FAILS the pipeline.

The production addition over the demo's print-a-metric curation: schema-as-code
checks that raise on violation instead of logging. Runs on the final curated
output (`data/curated/synthesized.jsonl`), which is committed — so CI re-runs it
for free (no API), and a regression anywhere upstream (extraction, normalize,
rebalance, synthesize) surfaces here as a hard failure with the exact offending
rows, not a silently-degraded dataset.

Three families of check, per build_spec Stage 4:
  - completeness    — required ids present, every record has >=1 diagnosis
  - referential     — every diagnosis/medication/vital name is in the closed
                      terminology vocabulary; units are canonical
  - value ranges    — confidence in [0,1]; vitals within plausible clinical
                      bounds; blood pressure well-formed "sys/dia"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

import terminology as t
from generation.generate_fhir import MED_CONTENT, SIMPLE_OBS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SYNTHESIZED_PATH = DATA_DIR / "curated" / "synthesized.jsonl"
REPORT_PATH = DATA_DIR / "reports" / "dq_gate.json"

CANONICAL_DX = [
    t.get_condition(i).standard_name or t.get_condition(i).source_name for i in t.list_conditions()
]
CANONICAL_MEDS = [c["label"] for c in MED_CONTENT.values()]
_NUMERIC_VITALS = {t.get_observation(loinc).source_name: spec for loinc, spec in SIMPLE_OBS.items()}
NUMERIC_VITAL_NAMES = list(_NUMERIC_VITALS)
_EXPECTED_UNIT = {name: spec["unit"] for name, spec in _NUMERIC_VITALS.items()}

# Plausible clinical bounds in canonical units (wider than generation ranges —
# a gate, not a tight fit).
VITAL_BOUNDS = {
    "Heart rate": (30, 220),
    "Body temperature": (30, 45),
    "Body weight": (20, 350),
    "Body height": (100, 250),
    "Glucose [Mass/volume] in Blood": (20, 600),
}


def _frames(records: list[dict]) -> dict[str, pd.DataFrame]:
    rec_rows, dx_rows, med_rows, numvit_rows, bp_rows = [], [], [], [], []
    for r in records:
        rec_rows.append(
            {
                "patient_id": r.get("patient_id"),
                "note_id": r.get("note_id"),
                "confidence": r.get("confidence"),
                "n_diagnoses": len(r.get("diagnoses", [])),
            }
        )
        for d in r.get("diagnoses", []):
            dx_rows.append({"note_id": r.get("note_id"), "name": d.get("name")})
        for m in r.get("medications", []):
            med_rows.append(
                {"note_id": r.get("note_id"), "name": m.get("name"), "dosage": m.get("dosage")}
            )
        for v in r.get("vitals", []):
            if v.get("name") == "Blood pressure":
                bp_rows.append({"note_id": r.get("note_id"), "value": v.get("value")})
            else:
                numvit_rows.append(
                    {
                        "note_id": r.get("note_id"),
                        "name": v.get("name"),
                        "value": v.get("value"),
                        "unit": v.get("unit"),
                    }
                )
    return {
        "records": pd.DataFrame(rec_rows),
        "diagnoses": pd.DataFrame(dx_rows),
        "medications": pd.DataFrame(med_rows),
        "numeric_vitals": pd.DataFrame(numvit_rows),
        "blood_pressure": pd.DataFrame(bp_rows),
    }


def _vital_in_bounds(df: pd.DataFrame) -> pd.Series:
    lo = df["name"].map(lambda n: VITAL_BOUNDS.get(n, (float("-inf"), float("inf")))[0])
    hi = df["name"].map(lambda n: VITAL_BOUNDS.get(n, (float("-inf"), float("inf")))[1])
    return (df["value"] >= lo) & (df["value"] <= hi)


def _vital_unit_ok(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda r: r["unit"] == _EXPECTED_UNIT.get(r["name"]), axis=1)


def _schemas() -> dict[str, pa.DataFrameSchema]:
    return {
        "records": pa.DataFrameSchema(
            {
                "patient_id": pa.Column(str, nullable=False, unique=True),
                "note_id": pa.Column(str, nullable=False),
                "confidence": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=False),
                "n_diagnoses": pa.Column(int, pa.Check.ge(1)),  # completeness
            }
        ),
        "diagnoses": pa.DataFrameSchema(
            {"note_id": pa.Column(str), "name": pa.Column(str, pa.Check.isin(CANONICAL_DX))}
        ),
        "medications": pa.DataFrameSchema(
            {
                "note_id": pa.Column(str),
                "name": pa.Column(str, pa.Check.isin(CANONICAL_MEDS)),
                "dosage": pa.Column(str, nullable=True),
            }
        ),
        "numeric_vitals": pa.DataFrameSchema(
            {
                "note_id": pa.Column(str),
                "name": pa.Column(str, pa.Check.isin(NUMERIC_VITAL_NAMES)),
                "value": pa.Column(float, nullable=False),
                "unit": pa.Column(str, nullable=False),
            },
            checks=[
                pa.Check(_vital_in_bounds, error="vital value out of clinical bounds"),
                pa.Check(_vital_unit_ok, error="vital unit not canonical"),
            ],
        ),
        "blood_pressure": pa.DataFrameSchema(
            {
                "note_id": pa.Column(str),
                "value": pa.Column(str, pa.Check.str_matches(r"^\d{2,3}/\d{2,3}$")),
            }
        ),
    }


def validate_records(records: list[dict]) -> dict:
    """Validate all curated records against the DQ schemas. Returns a JSON-able
    report {passed, n_records, tables:{name:{rows,passed,failures:[...]}}}."""
    frames = _frames(records)
    schemas = _schemas()
    tables: dict[str, dict] = {}
    all_passed = True

    for name, schema in schemas.items():
        df = frames[name]
        entry: dict = {"rows": int(len(df)), "passed": True, "failures": []}
        if df.empty:
            tables[name] = entry
            continue
        try:
            schema.validate(df, lazy=True)
        except SchemaErrors as exc:
            all_passed = False
            entry["passed"] = False
            fc = exc.failure_cases[["check", "column", "failure_case", "index"]]
            entry["failures"] = fc.head(25).to_dict(orient="records")
            entry["n_failures"] = int(len(exc.failure_cases))
        tables[name] = entry

    return {"passed": all_passed, "n_records": len(records), "tables": tables}


def read_records(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def gate_and_promote(
    records: list[dict], staging_path: Path, final_path: Path, report_path: Path
) -> dict:
    """Stage-then-promote: write `records` to `staging_path`, run the DQ gate, and
    promote staging -> final ONLY if the gate passes — so the final (committed)
    artifact existing is equivalent to it having passed. On failure `final_path`
    is left untouched (last-good), and the staging file + report remain for
    triage. Returns the report; the caller decides how to fail the pipeline."""
    _write_jsonl(staging_path, records)
    report = validate_records(records)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    if report["passed"]:
        staging_path.replace(final_path)  # atomic on the same filesystem
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 Pandera DQ gate (fails the pipeline).")
    parser.add_argument("--in", dest="in_path", type=Path, default=SYNTHESIZED_PATH)
    parser.add_argument("--report-out", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    report = validate_records(read_records(args.in_path))
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2))

    for name, entry in report["tables"].items():
        status = "PASS" if entry["passed"] else f"FAIL ({entry.get('n_failures', 0)})"
        print(f"  {name:16} {entry['rows']:>5} rows  {status}")
    if not report["passed"]:
        raise SystemExit(f"DQ GATE FAILED — see {args.report_out}")
    print(f"DQ gate PASSED across {report['n_records']} records -> {args.report_out}")


if __name__ == "__main__":
    main()
