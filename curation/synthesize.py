"""Stage 4 — synthesize new records for zero/under-represented diagnosis categories.

The final curation sub-step. `rebalance.py` can only amplify a category that
already appears at least once (duplication can't manufacture a category from
nothing); this fills the rest by generating genuinely new, differently-worded
records via the Anthropic API, up to the same target rebalance used (the max
per-category count in the pre-rebalance data).

**Paid, Lane 1, cache-first** — same discipline as Stage 3 extraction: the model
picks category-appropriate meds/vitals from the closed vocabulary (strict tool
enums), code fills canonical dosage/unit and seeded dates/ids, and one response
per category is cached so a rerun is free. At the 100-record scale every
diagnosis is already represented, so this runs as a $0 no-op; it exists for the
edge/scale cases where a category is genuinely absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import date
from pathlib import Path
from typing import Any

import terminology as t
from common.resilient_client import IncrementalCache, call_with_retry
from curation.rebalance import CANONICAL_CONDITION_ORDER, category_counts
from generation.generate_fhir import MED_CONTENT, SIMPLE_OBS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REDACTED_PATH = DATA_DIR / "curated" / "redacted.jsonl"
REBALANCED_PATH = DATA_DIR / "curated" / "rebalanced.jsonl"
SYNTHESIZED_PATH = DATA_DIR / "curated" / "synthesized.jsonl"
CACHE_PATH = Path(__file__).resolve().parent / "cache" / "synthesize_cache.json"

MODEL = "claude-haiku-4-5"
PROMPT_VERSION = "v1"

MEDICATION_NAMES = [c["label"] for c in MED_CONTENT.values()]
_MED_DOSAGE = {c["label"]: c["full"] for c in MED_CONTENT.values()}
_OBS = {t.get_observation(loinc).source_name: spec for loinc, spec in SIMPLE_OBS.items()}
VITAL_NAMES = list(_OBS)

BIRTH_RANGE = (date(1940, 1, 1), date(2005, 12, 31))

SYNTHESIS_TOOL: dict[str, Any] = {
    "name": "record_synthetic_patient",
    "description": (
        "Record one SYNTHETIC (non-real) patient's clinical fields for the given "
        "primary diagnosis: category-appropriate medications and 2-4 vitals with "
        "plausible numeric values. Choose only from the provided enums."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "medications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "enum": MEDICATION_NAMES}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
            "vitals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": VITAL_NAMES},
                        "value": {"type": "number"},
                    },
                    "required": ["name", "value"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["medications", "vitals"],
        "additionalProperties": False,
    },
    "strict": True,
}


def _new_client():
    import anthropic  # noqa: PLC0415 — lazy so the no-op path needs no SDK/key

    return anthropic.Anthropic()


def generate_fields(client, category: str) -> dict:
    prompt = (
        f"Generate one SYNTHETIC (non-real) patient record whose primary diagnosis "
        f"is '{category}'. Pick clinically appropriate medications and 2-4 vitals."
    )
    response = call_with_retry(
        client.messages.create,
        model=MODEL,
        max_tokens=1024,
        tools=[SYNTHESIS_TOOL],
        tool_choice={"type": "tool", "name": "record_synthetic_patient"},
        messages=[{"role": "user", "content": prompt}],
    )
    tool_use = next(b for b in response.content if b.type == "tool_use")
    return dict(tool_use.input)


def _seeded_rng(category: str, index: int, salt: str) -> random.Random:
    seed = int(hashlib.sha256(f"synth:{category}:{index}:{salt}".encode()).hexdigest(), 16)
    return random.Random(seed)


def build_record(category: str, index: int, fields: dict) -> dict:
    """Assemble a full curated-schema record from the model's med/vital choices
    plus code-filled canonical dosage/unit and seeded id."""
    medications = [
        {"name": m["name"], "dosage": _MED_DOSAGE[m["name"]]} for m in fields["medications"]
    ]
    vitals = [
        {"name": v["name"], "value": round(float(v["value"]), 1), "unit": _OBS[v["name"]]["unit"]}
        for v in fields["vitals"]
    ]
    pid = f"synth-{_seeded_rng(category, index, 'id').getrandbits(48):012x}"
    return {
        "patient_id": pid,
        "note_id": f"note-{pid}",
        "diagnoses": [{"name": category}],
        "medications": medications,
        "vitals": vitals,
        "confidence": 1.0,
        "low_confidence": False,
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "synthesized": True,
        "synthesized_category": category,
    }


def read_records(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def deficits(records: list[dict], redacted_path: Path) -> dict[str, int]:
    """Per-category shortfall against the pre-rebalance target (max category count)."""
    target = max(category_counts(list(read_records(redacted_path))).values(), default=0)
    current = category_counts(records)
    return {
        c: target - current.get(c, 0)
        for c in CANONICAL_CONDITION_ORDER
        if target - current.get(c, 0) > 0
    }


def synthesize_records(client, records, cache, needed: dict[str, int], *, refresh=False):
    """Return (augmented_records, new_records). `client` may be None when needed is empty."""
    augmented, new_records = list(records), []
    for category, deficit in needed.items():
        for i in range(1, deficit + 1):
            key = category if i == 1 else f"{category}#{i}"
            if refresh or key not in cache:
                cache[key] = generate_fields(client, category)
            record = build_record(category, i, cache[key])
            augmented.append(record)
            new_records.append(record)
    return augmented, new_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4 synthesize (paid, cache-first no-op when full)."
    )
    parser.add_argument("--in", dest="in_path", type=Path, default=REBALANCED_PATH)
    parser.add_argument("--out", type=Path, default=SYNTHESIZED_PATH)
    parser.add_argument("--redacted", type=Path, default=REDACTED_PATH)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    records = list(read_records(args.in_path))
    needed = deficits(records, args.redacted)
    cache = IncrementalCache(args.cache)

    # Only build a paid client when there's actually a category to fill.
    client = None
    if needed and any(k not in cache for k in _wanted_keys(needed)) and not args.refresh:
        client = _new_client()
    elif needed and args.refresh:
        client = _new_client()

    augmented, new_records = synthesize_records(
        client, records, cache, needed, refresh=args.refresh
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for record in augmented:
            f.write(json.dumps(record) + "\n")

    if new_records:
        print(f"Synthesized {len(new_records)} records for {sorted(needed)} -> {args.out}")
    else:
        print(
            f"No under-represented category — nothing to synthesize ($0). "
            f"Wrote {len(augmented)} records -> {args.out}"
        )


def _wanted_keys(needed: dict[str, int]) -> list[str]:
    keys = []
    for category, deficit in needed.items():
        keys += [category if i == 1 else f"{category}#{i}" for i in range(1, deficit + 1)]
    return keys


if __name__ == "__main__":
    main()
