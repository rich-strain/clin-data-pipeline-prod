"""Stage 4 build — curation + data-quality gate.

Chains the curation sub-steps and ends on the Pandera DQ gate that fails the
pipeline on any violation:

    extractions.jsonl
      -> normalize   (canonical dx/vital/dosage)        -> normalized.jsonl
      -> redact      (PHI leakage assertion; de-id upstream) -> redacted.jsonl
      -> rebalance   (oversample under-represented dx)   -> rebalanced.jsonl
      -> synthesize  (fill zero-represented dx; $0 no-op at 100 scale)
      -> dq gate     (completeness / referential / value ranges)
                     -> promote staging -> synthesized.jsonl (only if it passes)

The DQ gate uses **stage-then-promote**: the curated output is written to a
staging file, validated, and only renamed to `synthesized.jsonl` if the gate
passes — so the committed artifact existing is equivalent to it having passed.
A failing run leaves `synthesized.jsonl` untouched (last-good) and keeps the
staging file + report for triage.

The synthesize sub-step is the only paid one (Lane 1) and is a no-op whenever
every diagnosis is already represented — which it is at the committed 100-record
scale, so `python run_stage4.py` runs free. All outputs are committed; CI
re-runs normalize + the DQ gate against them without the API.

Run:  python run_stage4.py
"""

from __future__ import annotations

import json
from pathlib import Path

from curation import dq, normalize, rebalance, redact, synthesize

ROOT = Path(__file__).resolve().parent
CURATED = ROOT / "data" / "curated"
STAGING_PATH = CURATED / "synthesized.staging.jsonl"


def main() -> None:
    CURATED.mkdir(parents=True, exist_ok=True)

    # 1) normalize
    raw = list(normalize.read_extractions(normalize.EXTRACTED_PATH))
    normalized, _, norm_metrics = normalize.compute_metrics(raw)
    _write(normalize.NORMALIZED_PATH, normalized)
    normalize.METRICS_PATH.write_text(json.dumps(norm_metrics, indent=2))
    print(
        f"normalize:  {norm_metrics['overall']['matched']}/{norm_metrics['overall']['total']} "
        f"values canonical ({norm_metrics['overall']['unmatched']} unmatched)"
    )

    # 2) redact (leakage assertion — raises on any residual PHI)
    leaks = [leak for r in normalized for leak in redact.find_leaks(r)]
    if leaks:
        raise SystemExit(f"PHI LEAK in curated records: {leaks[:5]}")
    _write(redact.REDACTED_PATH, normalized)
    print(f"redact:     0 PHI patterns across {len(normalized)} records (de-id is upstream)")

    # 3) rebalance
    augmented, duplicates = rebalance.rebalance_records(normalized)
    _write(rebalance.REBALANCED_PATH, augmented)
    print(f"rebalance:  +{len(duplicates)} duplicates -> {len(augmented)} records")

    # 4) synthesize (paid, cache-first; no-op when no category is short). The
    # final synthesized.jsonl is NOT written here — it's staged and only promoted
    # by the gate below, so the committed artifact always passed.
    needed = synthesize.deficits(augmented, redact.REDACTED_PATH)
    if needed:
        raise SystemExit(
            f"synthesize needs the API for {sorted(needed)} — run "
            f"`python -m curation.synthesize` (Lane 1, paid) then re-run the gate."
        )
    print(f"synthesize: nothing under-represented — $0 no-op ({len(augmented)} records)")

    # 5) DQ gate — stage-then-promote: writes staging, validates, and promotes to
    # synthesized.jsonl only on pass. On failure the committed artifact is left
    # untouched and staging + report remain for triage.
    report = dq.gate_and_promote(
        augmented, STAGING_PATH, synthesize.SYNTHESIZED_PATH, dq.REPORT_PATH
    )
    for name, entry in report["tables"].items():
        status = "PASS" if entry["passed"] else f"FAIL ({entry.get('n_failures', 0)})"
        print(f"  dq/{name:16} {entry['rows']:>5} rows  {status}")
    if not report["passed"]:
        raise SystemExit(
            f"DQ GATE FAILED — synthesized.jsonl NOT updated; staged data at "
            f"{STAGING_PATH}, report at {dq.REPORT_PATH}"
        )
    print(
        f"DQ gate PASSED across {report['n_records']} records — "
        f"promoted to {synthesize.SYNTHESIZED_PATH.name}."
    )


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
