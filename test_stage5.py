"""Stage 5 verification (Working Plan step 6 verify gate).

Free, deterministic, CI-safe: no API, no network. Exercises the split/format/gold
logic and validates the committed dataset artifacts.

Run: python -m pytest test_stage5.py -v
"""

import json
from pathlib import Path

import format_jsonl
import gold
import split

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- split: group-aware, leakage-safe --------------------------------------


def test_split_is_group_aware_and_leakage_free() -> None:
    records = list(split.read_records(split.SYNTHESIZED_PATH))
    splits = split.build_splits(records)
    # No original patient group appears in more than one split...
    assert split.leakage(splits) == set()
    # ...i.e. every group maps to exactly one split (which is where its rebalance
    # duplicates land too, since they share the original's patient id).
    where: dict[str, set[str]] = {}
    for name, split_records in splits.items():
        for r in split_records:
            where.setdefault(split.original_patient_id(r), set()).add(name)
    assert where and all(len(s) == 1 for s in where.values())


def test_split_ratios_are_near_target_and_partition_all_records() -> None:
    records = list(split.read_records(split.SYNTHESIZED_PATH))
    eligible = [r for r in records if not r.get("synthesized")]
    splits = split.build_splits(records)
    total = sum(len(v) for v in splits.values())
    assert total == len(eligible), "splits must partition every eligible record exactly once"
    for name, frac in split.SPLIT_FRACTIONS.items():
        actual = len(splits[name]) / total
        assert abs(actual - frac) < 0.08, f"{name} ratio {actual:.2f} far from target {frac}"


def test_split_is_deterministic() -> None:
    records = list(split.read_records(split.SYNTHESIZED_PATH))
    a = {k: [r["patient_id"] for r in v] for k, v in split.build_splits(records).items()}
    b = {k: [r["patient_id"] for r in v] for k, v in split.build_splits(records).items()}
    assert a == b


# --- format: instruction/response ------------------------------------------


def test_format_joins_deid_note_and_builds_json_response() -> None:
    notes = format_jsonl.load_notes(format_jsonl.DEID_NOTES_PATH)
    record = list(split.read_records(split.SPLIT_PATHS["train"]))[0]
    ex = format_jsonl.format_record(record, notes)
    assert ex["instruction"] == notes[record["note_id"]]
    payload = json.loads(ex["response"])  # response is a JSON *string*
    assert set(payload) == set(format_jsonl.RESPONSE_FIELDS)
    # No pipeline bookkeeping leaks into the response.
    assert "patient_id" not in payload and "confidence" not in payload


def test_committed_formatted_splits_align_with_curated_splits() -> None:
    for name, path in format_jsonl.FORMATTED_PATHS.items():
        formatted = _read(path)
        curated = _read(split.SPLIT_PATHS[name])
        assert len(formatted) == len(curated), f"{name}: formatted/curated count mismatch"
        assert all("instruction" in e and "response" in e for e in formatted)


# --- gold: frozen + versioned ----------------------------------------------


def test_committed_gold_matches_its_manifest_hash() -> None:
    assert gold.GOLD_PATH.exists() and gold.GOLD_MANIFEST_PATH.exists(), (
        "gold set missing — run `python run_stage5.py`"
    )
    assert gold.verify_gold(), "gold.jsonl content does not match its manifest sha256"


def test_gold_is_the_frozen_test_split() -> None:
    gold_examples = _read(gold.GOLD_PATH)
    test_examples = _read(format_jsonl.FORMATTED_PATHS["test"])
    assert gold_examples == test_examples, "gold set must equal the committed test split"
    manifest = json.loads(gold.GOLD_MANIFEST_PATH.read_text())
    assert manifest["n_examples"] == len(gold_examples)
