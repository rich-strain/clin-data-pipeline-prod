"""Stage 7 — evaluation metrics + release gate (pure, no ML deps).

Pure Python so CI type-checks and tests it while the heavy `evaluate.py` (which
runs the committed adapter on the gold set) stays a local-only Lane 1 script. The
scoring is a lookup against the same closed terminology vocabulary the rest of the
pipeline uses: a predicted diagnosis/medication name that isn't a canonical value
is counted **separately as non-canonical** (a tracked hallucination signal), never
fuzzy-matched to the nearest label or silently dropped.

Scope: diagnoses + medications (both have canonical name sets here, so exact-name
micro P/R/F1 is well-defined). Vitals are out of scope — scoring them needs
value-closeness judgment, a genuinely different metric-design problem — stated,
not ignored.
"""

from __future__ import annotations

import json

import terminology as t
from generation.generate_fhir import MED_CONTENT

CANONICAL_DIAGNOSES = [
    t.get_condition(i).standard_name or t.get_condition(i).source_name for i in t.list_conditions()
]
CANONICAL_MEDS = [c["label"] for c in MED_CONTENT.values()]

MAX_NON_CANONICAL_EXAMPLES = 20
MAX_FAILURE_EXAMPLES = 5

# Release gate — a model is "releasable" only if it clears all of these on the
# frozen gold set. Deliberately modest but non-trivial for a 0.5B/local fine-tune;
# the point is a real pass/fail gate, not a rubber stamp.
RELEASE_THRESHOLDS = {
    "json_validity_rate": 0.90,
    "diagnosis_micro_f1": 0.80,
    "medication_micro_f1": 0.70,
    "max_non_canonical": 2,  # total hallucinated (out-of-vocab) names allowed across the gold set
}


def _names(obj: dict, field: str) -> list[str]:
    items = obj.get(field)
    if not isinstance(items, list):
        return []
    return [i["name"] for i in items if isinstance(i, dict) and isinstance(i.get("name"), str)]


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def score_field(parsed_pairs: list[tuple[dict, dict]], canonical: list[str], field: str) -> dict:
    """Micro P/R/F1 for one field over parseable outputs. `parsed_pairs` is
    (ground_truth, parsed_output). Non-canonical predicted names are tracked, not
    scored as false positives against the canonical categories."""
    canon = set(canonical)
    tp = fp = fn = 0
    per_cat = {c: {"tp": 0, "fp": 0, "fn": 0} for c in canonical}
    non_canonical: list[str] = []

    for gt, parsed in parsed_pairs:
        gt_names = set(_names(gt, field)) & canon
        out_names = _names(parsed, field)
        out_canon = {n for n in out_names if n in canon}
        non_canonical += [n for n in out_names if n not in canon]
        for c in canonical:
            in_gt, in_out = c in gt_names, c in out_canon
            if in_gt and in_out:
                tp += 1
                per_cat[c]["tp"] += 1
            elif in_gt:
                fn += 1
                per_cat[c]["fn"] += 1
            elif in_out:
                fp += 1
                per_cat[c]["fp"] += 1

    precision, recall, f1 = _prf(tp, fp, fn)
    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "non_canonical_count": len(non_canonical),
        "non_canonical_examples": sorted(set(non_canonical))[:MAX_NON_CANONICAL_EXAMPLES],
        "per_category": [{"category": c, **per_cat[c]} for c in canonical],
    }


def evaluate_predictions(pairs: list[tuple[dict, str]]) -> dict:
    """Compute metrics over (ground_truth_obj, raw_model_output_text) pairs. Pure:
    no model, no I/O. JSON validity is measured over ALL pairs; field metrics over
    the parseable ones."""
    n_total = len(pairs)
    parsed_pairs: list[tuple[dict, dict]] = []
    failures: list[dict] = []
    for idx, (gt, output_text) in enumerate(pairs):
        try:
            parsed = json.loads(output_text)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if not isinstance(parsed, dict):
            if len(failures) < MAX_FAILURE_EXAMPLES:
                failures.append(
                    {
                        "example_index": idx,
                        "reason": "unparseable JSON",
                        "output": output_text[:300],
                    }
                )
            continue
        parsed_pairs.append((gt, parsed))

    n_valid = len(parsed_pairs)
    return {
        "n_examples": n_total,
        "json_validity": {
            "valid": n_valid,
            "total": n_total,
            "rate": (n_valid / n_total) if n_total else 0.0,
        },
        "diagnosis": score_field(parsed_pairs, CANONICAL_DIAGNOSES, "diagnoses"),
        "medication": score_field(parsed_pairs, CANONICAL_MEDS, "medications"),
        "failure_examples": failures,
        "scope_note": (
            "diagnoses + medications (exact canonical-name micro P/R/F1). Vitals "
            "out of scope (value-closeness judgment). Non-canonical names tracked "
            "as hallucinations, not fuzzy-matched."
        ),
    }


def release_gate(metrics: dict, thresholds: dict = RELEASE_THRESHOLDS) -> dict:
    """Pass/fail against the release thresholds. Returns {passed, checks:[...]}."""
    non_canonical = (
        metrics["diagnosis"]["non_canonical_count"] + metrics["medication"]["non_canonical_count"]
    )
    checks = [
        {
            "name": "json_validity_rate",
            "value": round(metrics["json_validity"]["rate"], 4),
            "threshold": thresholds["json_validity_rate"],
            "op": ">=",
            "passed": metrics["json_validity"]["rate"] >= thresholds["json_validity_rate"],
        },
        {
            "name": "diagnosis_micro_f1",
            "value": round(metrics["diagnosis"]["micro_f1"], 4),
            "threshold": thresholds["diagnosis_micro_f1"],
            "op": ">=",
            "passed": metrics["diagnosis"]["micro_f1"] >= thresholds["diagnosis_micro_f1"],
        },
        {
            "name": "medication_micro_f1",
            "value": round(metrics["medication"]["micro_f1"], 4),
            "threshold": thresholds["medication_micro_f1"],
            "op": ">=",
            "passed": metrics["medication"]["micro_f1"] >= thresholds["medication_micro_f1"],
        },
        {
            "name": "non_canonical_total",
            "value": non_canonical,
            "threshold": thresholds["max_non_canonical"],
            "op": "<=",
            "passed": non_canonical <= thresholds["max_non_canonical"],
        },
    ]
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def build_model_card(registry_entry: dict, metrics: dict, gate: dict) -> str:
    """A model card documenting scope, data, metrics, gate, and honest limits."""
    dx, med = metrics["diagnosis"], metrics["medication"]

    def field_line(label: str, s: dict) -> str:
        prf = f"{s['micro_precision']:.3f} / {s['micro_recall']:.3f} / {s['micro_f1']:.3f}"
        counts = (
            f"TP {s['tp']}, FP {s['fp']}, FN {s['fn']}, non-canonical {s['non_canonical_count']}"
        )
        return f"- **{label} micro P/R/F1:** {prf} ({counts})"

    jv = metrics["json_validity"]
    lineage = registry_entry["data_lineage"]
    verdict = "✅ PASS" if gate["passed"] else "❌ FAIL"
    lines = [
        f"# Model card — clinical extraction LoRA ({registry_entry['version']})",
        "",
        "## Overview",
        f"- **Base model:** {registry_entry['base_model']} (~0.5B params)",
        f"- **Adapter:** LoRA r={registry_entry['lora_config']['r']}, "
        f"{registry_entry['adapter_bytes'] / 1e6:.1f} MB, best epoch "
        f"{registry_entry['best_epoch']}",
        "- **Task:** extract diagnoses / medications / vitals from a de-identified note as JSON.",
        f"- **Trained on:** {registry_entry['train_examples']} examples; git "
        f"`{registry_entry['git_sha'][:12]}`, gold set `{lineage.get('gold_version')}`.",
        "",
        "## Evaluation (frozen gold set)",
        f"- **JSON validity:** {jv['valid']}/{jv['total']} ({jv['rate']:.0%})",
        field_line("Diagnosis", dx),
        field_line("Medication", med),
        "",
        f"## Release gate: {verdict}",
    ]
    for c in gate["checks"]:
        mark = "✅" if c["passed"] else "❌"
        lines.append(f"- {mark} `{c['name']}` = {c['value']} (needs {c['op']} {c['threshold']})")
    lines += [
        "",
        "## Intended use & limits (honest framing)",
        "- **Synthetic-data demonstration**, not a production medical device. A 0.5B "
        "model fine-tuned on ~100 synthetic notes is not a claim of clinical accuracy.",
        "- Vitals are not scored (value-closeness is out of scope). Metrics are over a "
        f"small ({metrics['n_examples']}-example) frozen gold set — treat as directional.",
        "- Closed vocabulary only; behavior on out-of-vocabulary clinical language is untested.",
        "- Not for clinical decisions. Inputs are synthetic, de-identified upstream (no PHI).",
    ]
    return "\n".join(lines) + "\n"
