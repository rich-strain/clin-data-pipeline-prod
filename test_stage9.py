"""Stage 9 verification (provenance + analytics builders).

Free, deterministic, CI-safe: validates the pipeline run/lineage log and the
Parquet analytics table against the committed pipeline artifacts.

Run: python -m pytest test_stage9.py -v
"""

import json
from pathlib import Path

import analytics
import provenance

ROOT = Path(__file__).parent


def test_provenance_log_is_content_addressed_and_matches_committed_artifacts() -> None:
    rows = provenance.build_provenance()
    assert rows, "provenance log should not be empty"
    for r in rows:
        assert len(r["sha256"]) == 64
        # The recorded hash matches the artifact's actual bytes right now.
        assert r["sha256"] == provenance.file_sha256(ROOT / r["artifact"])
    stages = {r["stage"] for r in rows}
    assert any("Gold" in s for s in stages) and any("Evaluation" in s for s in stages)


def test_committed_provenance_json_matches_a_fresh_build() -> None:
    committed = json.loads((ROOT / "data" / "reports" / "provenance.json").read_text())
    assert committed == provenance.build_provenance(), "committed provenance.json is stale"


def test_analytics_table_flattens_splits_one_row_per_diagnosis() -> None:
    df = analytics.build_table()
    assert not df.empty
    assert set(df["split"].unique()) <= {"train", "val", "test"}
    assert {"patient_id", "diagnosis", "confidence", "is_rebalance_duplicate"} <= set(df.columns)
    # Every row's diagnosis is a real string; confidence is in [0, 1] where present.
    assert df["diagnosis"].map(lambda s: isinstance(s, str) and bool(s)).all()
    conf = df["confidence"].dropna()
    assert ((conf >= 0) & (conf <= 1)).all()


def test_analytics_parquet_is_queryable_with_duckdb() -> None:
    import duckdb

    path = ROOT / "data" / "analytics" / "pipeline_analytics.parquet"
    assert path.exists(), "analytics parquet missing — run `python analytics.py`"
    n = duckdb.sql(f"SELECT COUNT(*) AS n FROM read_parquet('{path}')").df()["n"][0]
    assert n == len(analytics.build_table())
