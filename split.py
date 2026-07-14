"""Stage 5 — group-aware, leakage-safe train/val/test split of curated records.

Takes `data/curated/synthesized.jsonl` (Stage 4's gate-passed output) and
partitions it into `data/curated/split_{train,val,test}.jsonl` — still in the
curated record shape (patient_id/diagnoses/medications/vitals/...), not yet the
instruction/response training format (`format_jsonl.py` does that next).
Splitting and formatting are separate steps so "which record goes in which split"
and "how a record becomes a training example" can each be verified independently.

**Split by ORIGINAL patient group, not by raw record.** `rebalance.py` produces
near-identical duplicate records (`rebalance_duplicate_of`, patient_id
`<orig>-dupN`) to correct diagnosis-category representation — they are not
independent patients. If a duplicate landed in val/test while its original sat in
train, the eval set would contain content the model saw almost verbatim in
training, silently inflating the metric. So the unit of splitting is the original
patient group (a patient's record plus every `-dupN` copy of it): every record in
a group always goes to the same split.

**Assignment: largest-group-first, balanced against the record-level target.**
Group sizes vary (1 record for most, up to ~8 after rebalancing), so assigning in
first-seen order can cluster large groups into one split and drift the record
ratio off target. Instead each group (largest first) goes to whichever split is
currently furthest below its target *record* count. Fully deterministic (a stable
sort by size, first-seen order as tiebreak) and it never splits a group across
sets, so the record-level ratio tracks the target without breaking anti-leakage.

**Synthesized records are excluded.** `synthesize.py` records (`synthesized:
true`) are fabricated structured fields with no de-identified note to form an
instruction from, so they can't become instruction/response examples; excluded
here rather than force-fit. A no-op at 100 scale (zero synthesized), but the logic
stays since that isn't guaranteed on a future regeneration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
SYNTHESIZED_PATH = DATA_DIR / "curated" / "synthesized.jsonl"
SPLIT_PATHS = {
    "train": DATA_DIR / "curated" / "split_train.jsonl",
    "val": DATA_DIR / "curated" / "split_val.jsonl",
    "test": DATA_DIR / "curated" / "split_test.jsonl",
}

# Record-level target fractions. test doubles as the frozen gold set (Stage 7).
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def original_patient_id(record: dict) -> str:
    """The original patient a record belongs to — itself, unless it's a
    rebalance duplicate, in which case whoever it was copied from."""
    return record.get("rebalance_duplicate_of", record["patient_id"])


def read_records(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def group_by_original_patient(records: list[dict]) -> dict[str, list[dict]]:
    """original_patient_id -> [records], in first-seen order."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(original_patient_id(r), []).append(r)
    return groups


def split_groups(
    groups: dict[str, list[dict]], fractions: dict[str, float]
) -> dict[str, list[str]]:
    """Assign whole groups to splits (largest-first, each to the split furthest
    below its target record count). Returns {split_name: [group_key, ...]}."""
    total = sum(len(v) for v in groups.values())
    targets = {name: total * frac for name, frac in fractions.items()}
    assigned: dict[str, list[str]] = {name: [] for name in fractions}
    counts = {name: 0 for name in fractions}

    for key in sorted(groups, key=lambda k: len(groups[k]), reverse=True):
        # Deficit = how far each split is below its target; ties broken by the
        # fraction order (dict insertion order), keeping this deterministic.
        pick = max(fractions, key=lambda name: targets[name] - counts[name])
        assigned[pick].append(key)
        counts[pick] += len(groups[key])
    return assigned


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def leakage(splits: dict[str, list[dict]]) -> set[str]:
    """Original patient groups that appear in more than one split (should be empty)."""
    seen: dict[str, str] = {}
    crossing: set[str] = set()
    for name, records in splits.items():
        for r in records:
            pid = original_patient_id(r)
            if pid in seen and seen[pid] != name:
                crossing.add(pid)
            seen[pid] = name
    return crossing


def build_splits(records: list[dict]) -> dict[str, list[dict]]:
    eligible = [r for r in records if not r.get("synthesized")]
    groups = group_by_original_patient(eligible)
    assigned = split_groups(groups, SPLIT_FRACTIONS)
    return {name: [r for k in keys for r in groups[k]] for name, keys in assigned.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5 group-aware train/val/test split.")
    parser.add_argument("--in", dest="in_path", type=Path, default=SYNTHESIZED_PATH)
    args = parser.parse_args()

    records = list(read_records(args.in_path))
    splits = build_splits(records)
    for name, split_records in splits.items():
        write_jsonl(split_records, SPLIT_PATHS[name])

    total = sum(len(v) for v in splits.values())
    for name, split_records in splits.items():
        n = len(split_records)
        print(f"  {name:5} {n:>4} records ({n / total:.0%})  -> {SPLIT_PATHS[name].name}")
    crossing = leakage(splits)
    if crossing:
        raise SystemExit(
            f"LEAKAGE: {len(crossing)} patient group(s) cross splits: {sorted(crossing)}"
        )
    print("Verified: no original patient group appears in more than one split.")


if __name__ == "__main__":
    main()
