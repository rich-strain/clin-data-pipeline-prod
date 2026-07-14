"""Stage 5 build — dataset assembly (split, format, freeze gold).

    curated/synthesized.jsonl
      -> split      group-aware, leakage-safe train/val/test (split.py)
      -> format     instruction/response JSONL via de-id note join (format_jsonl)
      -> gold       freeze the test split as the versioned held-out gold set (gold.py)

All free (no API). Verifies zero patient-group leakage across splits and that the
frozen gold set matches its manifest hash. Outputs are committed; CI re-runs the
whole thing against committed inputs.

Run:  python run_stage5.py
"""

from __future__ import annotations

import format_jsonl
import gold
import split


def main() -> None:
    # 1) split (group-aware, leakage-safe)
    records = list(split.read_records(split.SYNTHESIZED_PATH))
    splits = split.build_splits(records)
    for name, split_records in splits.items():
        split.write_jsonl(split_records, split.SPLIT_PATHS[name])

    total = sum(len(v) for v in splits.values())
    print(f"split: {total} eligible records ({len(records)} curated)")
    for name, split_records in splits.items():
        n = len(split_records)
        print(f"  {name:5} {n:>4} records ({n / total:.0%})")

    crossing = split.leakage(splits)
    if crossing:
        raise SystemExit(
            f"LEAKAGE: {len(crossing)} patient group(s) cross splits: {sorted(crossing)}"
        )
    print("  leakage check: no original patient group crosses splits ✓")

    # 2) format (instruction/response via de-identified note join)
    notes_by_id = format_jsonl.load_notes(format_jsonl.DEID_NOTES_PATH)
    formatted: dict[str, list[dict]] = {}
    for name in split.SPLIT_PATHS:
        examples = format_jsonl.format_split(name, notes_by_id)
        format_jsonl.write_jsonl(examples, format_jsonl.FORMATTED_PATHS[name])
        formatted[name] = examples
        print(f"format: {name:5} {len(examples):>4} instruction/response examples")

    # 3) freeze the test split as the versioned gold set
    manifest = gold.freeze_gold(formatted["test"])
    if not gold.verify_gold():
        raise SystemExit("gold set does not match its manifest hash after freezing")
    print(
        f"gold:  frozen {manifest['n_examples']} examples as {manifest['version']} "
        f"(sha256 {manifest['sha256'][:12]}…) -> {gold.GOLD_PATH}"
    )


if __name__ == "__main__":
    main()
