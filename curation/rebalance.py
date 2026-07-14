"""Stage 4 — rebalance diagnosis-category representation by oversampling.

Takes `data/curated/redacted.jsonl` and evens out the under-represented diagnosis
categories by DUPLICATING existing records (never downsampling — with a
data-starved fine-tune, shrinking to the rarest category is actively worse).
Generating genuinely new records for zero-represented categories is deliberately
left to `synthesize.py`, the next sub-step; duplication can only amplify a
category that already appears at least once.

Category is counted once per record (not per raw mention — `normalize.py` already
deduped mentions), because a record is one training example regardless of how
many times it names a diagnosis.

**Duplicates are marked, not silently blended in.** Each carries
`rebalance_duplicate_of: <original patient_id>` and a suffixed patient_id
(`<original>-dup1`, …). This matters downstream: Stage 5's group-aware split must
keep a duplicate in the SAME split as its original (it groups on the original id),
or a near-identical example leaking across the split boundary would inflate
validation scores.

**Known limitation — multi-diagnosis overshoot.** Duplicating a record to boost
one deficient category also boosts the other diagnoses it carries, so some
categories overshoot the target. A set-cover selection would minimize this;
not worth it at this scale — documented, not silently accepted.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import terminology as t

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REDACTED_PATH = DATA_DIR / "curated" / "redacted.jsonl"
REBALANCED_PATH = DATA_DIR / "curated" / "rebalanced.jsonl"

CANONICAL_CONDITION_ORDER = [
    t.get_condition(icd).standard_name or t.get_condition(icd).source_name
    for icd in t.list_conditions()
]


def _diagnosis_names(record: dict) -> set[str]:
    return {d["name"] for d in record["diagnoses"]}


def category_counts(records: list[dict]) -> collections.Counter:
    """Number of records containing each diagnosis category (deduped per record)."""
    counts: collections.Counter = collections.Counter()
    for r in records:
        counts.update(_diagnosis_names(r))
    return counts


def _pick_templates(records: list[dict]) -> dict[str, dict]:
    """First record (in original order) containing each category, if any."""
    templates: dict[str, dict] = {}
    for category in CANONICAL_CONDITION_ORDER:
        for r in records:
            if category in _diagnosis_names(r):
                templates[category] = r
                break
    return templates


def rebalance_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (augmented_records, duplicates_added)."""
    before = category_counts(records)
    target = max(before.values()) if before else 0
    templates = _pick_templates(records)

    augmented = list(records)
    duplicates: list[dict] = []
    dup_index: collections.Counter = collections.Counter()

    for category in CANONICAL_CONDITION_ORDER:
        template = templates.get(category)
        if template is None:
            continue  # zero-represented: nothing to duplicate — synthesize.py's job
        current = sum(1 for r in augmented if category in _diagnosis_names(r))
        while current < target:
            dup_index[template["patient_id"]] += 1
            dup = dict(template)
            dup["patient_id"] = f"{template['patient_id']}-dup{dup_index[template['patient_id']]}"
            dup["rebalance_duplicate_of"] = template["patient_id"]
            augmented.append(dup)
            duplicates.append(dup)
            current += 1

    return augmented, duplicates


def read_records(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 rebalance diagnosis categories.")
    parser.add_argument("--in", dest="in_path", type=Path, default=REDACTED_PATH)
    parser.add_argument("--out", type=Path, default=REBALANCED_PATH)
    args = parser.parse_args()

    records = list(read_records(args.in_path))
    before = category_counts(records)
    augmented, duplicates = rebalance_records(records)
    after = category_counts(augmented)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for record in augmented:
            f.write(json.dumps(record) + "\n")

    print(
        f"Wrote {len(augmented)} records "
        f"({len(records)} original + {len(duplicates)} dup) to {args.out}"
    )
    print(f"\n{'diagnosis category':55} {'before':>7} {'after':>7}")
    for category in CANONICAL_CONDITION_ORDER:
        flag = "  <- still 0, needs synthesize.py" if after.get(category, 0) == 0 else ""
        print(f"{category:55} {before.get(category, 0):>7} {after.get(category, 0):>7}{flag}")


if __name__ == "__main__":
    main()
