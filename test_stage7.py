"""Stage 7 verification (metrics + release gate half).

Free, deterministic, CI-safe: exercises the pure evaluation scoring, release gate,
and model card against hand-built prediction pairs (independent of the adapter,
which is Lane 1). The gold-set inference itself lives in evaluate.py (torch).

Run: python -m pytest test_stage7.py -v
"""

import json
from pathlib import Path

import eval_metrics

EVAL_REPORT = Path(__file__).parent / "training_results" / "eval_report.json"

DX = eval_metrics.CANONICAL_DIAGNOSES[0]  # a real canonical diagnosis
DX2 = eval_metrics.CANONICAL_DIAGNOSES[1]
MED = eval_metrics.CANONICAL_MEDS[0]  # a real canonical medication


def _pred(diagnoses, medications=()):
    return json.dumps(
        {
            "diagnoses": [{"name": n} for n in diagnoses],
            "medications": [{"name": n} for n in medications],
            "vitals": [],
        }
    )


def _gt(diagnoses, medications=()):
    return {
        "diagnoses": [{"name": n} for n in diagnoses],
        "medications": [{"name": n} for n in medications],
        "vitals": [],
    }


def test_perfect_predictions_score_1_and_pass_the_gate() -> None:
    pairs = [(_gt([DX], [MED]), _pred([DX], [MED])) for _ in range(4)]
    m = eval_metrics.evaluate_predictions(pairs)
    assert m["json_validity"]["rate"] == 1.0
    assert m["diagnosis"]["micro_f1"] == 1.0 and m["medication"]["micro_f1"] == 1.0
    assert eval_metrics.release_gate(m)["passed"]


def test_invalid_json_counts_against_validity_only() -> None:
    pairs = [(_gt([DX]), _pred([DX])), (_gt([DX]), "not json at all")]
    m = eval_metrics.evaluate_predictions(pairs)
    assert m["json_validity"] == {"valid": 1, "total": 2, "rate": 0.5}
    # The parseable one is still perfect on diagnoses; the bad one is a failure example.
    assert m["diagnosis"]["tp"] == 1 and m["diagnosis"]["fn"] == 0
    assert m["failure_examples"] and m["failure_examples"][0]["reason"] == "unparseable JSON"


def test_missing_and_spurious_diagnoses_are_fn_and_fp() -> None:
    pairs = [
        (_gt([DX, DX2]), _pred([DX])),  # DX2 missed -> fn
        (_gt([DX]), _pred([DX, DX2])),  # DX2 spurious -> fp
    ]
    m = eval_metrics.evaluate_predictions(pairs)
    assert m["diagnosis"]["tp"] == 2 and m["diagnosis"]["fn"] == 1 and m["diagnosis"]["fp"] == 1


def test_non_canonical_names_are_tracked_not_scored_as_fp() -> None:
    pairs = [(_gt([DX]), _pred([DX, "Made-up disease"]))]
    m = eval_metrics.evaluate_predictions(pairs)
    assert m["diagnosis"]["fp"] == 0  # not fuzzy-matched, not a canonical FP
    assert m["diagnosis"]["non_canonical_count"] == 1
    assert "Made-up disease" in m["diagnosis"]["non_canonical_examples"]


def test_release_gate_fails_below_thresholds() -> None:
    # All predictions empty -> zero recall/F1 -> gate fails.
    pairs = [(_gt([DX], [MED]), _pred([], [])) for _ in range(3)]
    m = eval_metrics.evaluate_predictions(pairs)
    gate = eval_metrics.release_gate(m)
    assert not gate["passed"]
    failed = {c["name"] for c in gate["checks"] if not c["passed"]}
    assert {"diagnosis_micro_f1", "medication_micro_f1"} <= failed


def test_model_card_reports_verdict_and_scope() -> None:
    m = eval_metrics.evaluate_predictions([(_gt([DX], [MED]), _pred([DX], [MED]))])
    gate = eval_metrics.release_gate(m)
    entry = {
        "version": "v1",
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "lora_config": {"r": 8},
        "adapter_bytes": 4_356_653,
        "best_epoch": 5,
        "train_examples": 98,
        "git_sha": "d4a2b3d538005b41",
        "data_lineage": {"gold_version": "v1"},
    }
    card = eval_metrics.build_model_card(entry, m, gate)
    assert "Model card" in card and "Release gate" in card
    assert "not a claim of clinical accuracy" in card  # honest-limits section present


def test_committed_eval_report_is_consistent_and_recomputes_its_gate() -> None:
    """The committed eval report (from the real gold-set run) must be internally
    consistent: re-running the pure gate on its metrics reproduces its verdict."""
    if not EVAL_REPORT.exists():  # eval is Lane 1 / local — skip cleanly if not run
        return
    report = json.loads(EVAL_REPORT.read_text())
    assert report["json_validity"]["total"] == report["n_examples"]
    recomputed = eval_metrics.release_gate(report, report["thresholds"])
    assert recomputed["passed"] == report["release_gate"]["passed"]
