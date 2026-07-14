"""Stage 9 — build the committed Parquet analytics table.

Flattens the curated split records into a denormalized, analytics-shaped fact
table (one row per patient×diagnosis) and writes it as **Parquet** — the real
columnar analytics store the Analytics page queries live via in-process duckdb
(no server, no Spark). This is the Tier-1 "real Parquet + duckdb" layer neither
prior repo had; Delta/Iceberg (ACID, time-travel) stays a documented upgrade.

Emits data/analytics/pipeline_analytics.parquet from data/curated/split_*.jsonl,
so `split` (train/val/test) is a real column to slice on.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from split import SPLIT_PATHS, read_records

ROOT = Path(__file__).resolve().parent
ANALYTICS_PATH = ROOT / "data" / "analytics" / "pipeline_analytics.parquet"


def build_rows() -> list[dict]:
    """One row per (patient, diagnosis) across all splits."""
    rows: list[dict] = []
    for split_name, path in SPLIT_PATHS.items():
        if not path.exists():
            continue
        for rec in read_records(path):
            for dx in rec["diagnoses"]:
                rows.append(
                    {
                        "patient_id": rec["patient_id"],
                        "split": split_name,
                        "diagnosis": dx["name"],
                        "n_diagnoses": len(rec["diagnoses"]),
                        "n_medications": len(rec["medications"]),
                        "n_vitals": len(rec["vitals"]),
                        "confidence": rec.get("confidence"),
                        "is_rebalance_duplicate": "rebalance_duplicate_of" in rec,
                    }
                )
    return rows


def build_table() -> pd.DataFrame:
    return pd.DataFrame(build_rows())


def main() -> None:
    df = build_table()
    ANALYTICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ANALYTICS_PATH, index=False)
    print(f"Wrote {len(df)} rows × {len(df.columns)} cols to {ANALYTICS_PATH}")
    print(f"  columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
