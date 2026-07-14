"""Stage 0/1 build (free, deterministic): generate synthetic US Core FHIR,
land it immutably as NDJSON, validate shapes, bind terminology, and derive the
OMOP CDM. Writes every committed artifact the Stage 0/1 app page reads.

Run:  python run_stage01.py   (defaults: 100 patients, messy, seed 42)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from generation.generate_fhir import as_bundle, generate_dataset
from generation.landing import read_landing, write_landing
from omop.etl import CDM_TABLES, fhir_to_omop
from terminology.bind import binding_report
from terminology.validate import validate_resources

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 0/1 artifacts.")
    parser.add_argument("--n-patients", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Generate without messiness")
    args = parser.parse_args()

    messy = not args.clean
    dataset = generate_dataset(args.n_patients, messy=messy, seed=args.seed)

    # 1) Immutable raw landing layer (source of truth).
    landing_dir = DATA / "landing"
    counts = write_landing(dataset, landing_dir)

    # 2) Convenience derivation: per-patient bundles for the app's sample view.
    canonical = DATA / "canonical"
    canonical.mkdir(parents=True, exist_ok=True)
    (canonical / "fhir_bundles.json").write_text(
        json.dumps([as_bundle(r) for r in dataset], indent=2)
    )

    # Everything below regenerates FROM the landing layer, not from `dataset`.
    resources = read_landing(landing_dir)

    # 3) Structural validation + 4) terminology binding reports.
    reports = DATA / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    validation = validate_resources(resources)
    (reports / "validation.json").write_text(json.dumps(validation, indent=2))
    binding = binding_report(resources)
    (reports / "terminology_binding.json").write_text(json.dumps(binding, indent=2))

    # 5) OMOP CDM tables.
    omop_dir = DATA / "omop"
    omop_dir.mkdir(parents=True, exist_ok=True)
    tables = fhir_to_omop(resources)
    for name in CDM_TABLES:
        pd.DataFrame(tables[name]).to_csv(omop_dir / f"{name}.csv", index=False)

    print(f"Landing NDJSON ({messy and 'messy' or 'clean'}, seed {args.seed}):")
    for rtype, n in counts.items():
        print(f"  {rtype:18s} {n}")
    print(
        f"Validation: {validation['valid']}/{validation['total']} base-FHIR valid, "
        f"{validation['us_core_profiled']} US Core-profiled"
    )
    print(
        f"Terminology binding: {binding['overall']['pct']}% of coded fields matched "
        f"({binding['overall']['matched']}/{binding['overall']['total']})"
    )
    print("OMOP CDM rows:", {name: len(tables[name]) for name in CDM_TABLES})


if __name__ == "__main__":
    main()
