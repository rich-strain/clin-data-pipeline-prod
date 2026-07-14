"""Headless verification for app.py (verify gate for each Working Plan step:
"app launches locally, navigation works, and each built page renders its real
committed data").

Loads the app and clicks each sidebar stage, asserting zero uncaught
exceptions on the initial load and on every page. Also checks that a built
page (Stage 0/1) renders real content while an unbuilt page stays a stub.

Reuses ONE AppTest session across all clicks rather than re-instantiating
AppTest.from_file() per page: the sibling clin-data-pipeline-scale repo
documented that this environment segfaults after repeatedly re-instantiating
fresh AppTest sessions in one process, while re-running .run() on a single
already-constructed session is stable. Re-running .run() after each sidebar
click is exactly what a real user session does anyway.

Run: python -m pytest test_app.py -v   (or: python test_app.py)
"""

from streamlit.testing.v1 import AppTest

from app import STAGES

STAGE_LABELS = [s[0] for s in STAGES]


def _radio(at: AppTest):
    """The single sidebar stage-selector radio."""
    assert len(at.radio) == 1, f"expected exactly one radio, found {len(at.radio)}"
    return at.radio[0]


def test_app_loads_and_every_stage_renders() -> None:
    at = AppTest.from_file("app.py")
    at.run()
    assert not at.exception, f"initial load raised: {at.exception}"

    # The radio should expose exactly the stages we defined, in order.
    assert _radio(at).options == STAGE_LABELS

    for label in STAGE_LABELS:
        _radio(at).set_value(label).run()
        assert not at.exception, f"stage {label!r} raised: {at.exception}"


def _all_text(at: AppTest) -> str:
    return " ".join(
        el.value for el in (*at.markdown, *at.info, *at.warning, *at.subheader) if el.value
    )


def test_stage01_page_renders_real_content_and_stub_pages_stay_stubs() -> None:
    at = AppTest.from_file("app.py")
    at.run()

    _radio(at).set_value("0/1 — Ingestion + Canonical Storage").run()
    assert not at.exception
    text = _all_text(at)
    assert "OMOP CDM" in text, "Stage 0/1 page should render real OMOP content"
    assert len(at.dataframe) >= 1, "Stage 0/1 page should render at least one real table"
    assert not any("Stub — not built yet" in (m.value or "") for m in at.info)

    # Stage 4 curation page renders real DQ-gate content (same session — a fresh
    # AppTest per page segfaults in this env, see module docstring).
    _radio(at).set_value("4 — Curation + DQ").run()
    assert not at.exception
    curation_text = _all_text(at)
    assert "Pandera" in curation_text, "Curation page should describe the Pandera gate"
    assert len(at.dataframe) >= 1, "Curation page should render real before/after tables"
    assert not any("Stub — not built yet" in (m.value or "") for m in at.info)

    # Stage 5 dataset-assembly page renders the leakage check + gold set.
    _radio(at).set_value("5 — Dataset Assembly").run()
    assert not at.exception
    dataset_text = _all_text(at)
    assert "leakage" in dataset_text.lower(), "Dataset page should show the leakage check"
    assert any("No original patient group" in (s.value or "") for s in at.success), (
        "Dataset page should report zero patient-group leakage"
    )
    assert not any("Stub — not built yet" in (m.value or "") for m in at.info)

    # Stage 6 training page renders the real run (loss history + lineage).
    _radio(at).set_value("6 — Training").run()
    assert not at.exception
    assert len(at.dataframe) >= 1, "Training page should render the real loss-history table"
    assert not any("Stub — not built yet" in (m.value or "") for m in at.info)

    # Stage 7 evaluation page renders the release gate + per-field metrics.
    _radio(at).set_value("7 — Evaluation + Release").run()
    assert not at.exception
    eval_text = _all_text(at)
    assert "gate" in eval_text.lower(), "Evaluation page should show the release gate"
    assert len(at.dataframe) >= 2, "Evaluation page should render per-field + gate tables"
    assert not any("Stub — not built yet" in (m.value or "") for m in at.info)

    # Provenance page renders the real run/lineage log.
    _radio(at).set_value("Provenance").run()
    assert not at.exception
    assert "lineage" in _all_text(at).lower(), "Provenance page should show the lineage log"
    assert len(at.dataframe) >= 1, "Provenance page should render the lineage table"
    assert not any("Stub — not built yet" in (m.value or "") for m in at.info)

    # Analytics page runs a real in-process duckdb query over the Parquet.
    _radio(at).set_value("Analytics").run()
    assert not at.exception
    assert any("duckdb" in (m.value or "").lower() for m in at.info), (
        "Analytics page describes duckdb"
    )
    assert len(at.dataframe) >= 1, "Analytics page should render a live query result"
    assert not any("Stub — not built yet" in (m.value or "") for m in at.info)

    # Live Inference renders its input UI (the model loads only on button click,
    # not at page render, so this stays light and torch-free in CI).
    _radio(at).set_value("Live Inference").run()
    assert not at.exception
    assert len(at.button) >= 1 and len(at.text_area) >= 1, "Live Inference should render its UI"
    assert not any("Stub — not built yet" in (m.value or "") for m in at.info)


if __name__ == "__main__":
    test_app_loads_and_every_stage_renders()
    test_stage01_page_renders_real_content_and_stub_pages_stay_stubs()
    print(f"OK — app loaded, all {len(STAGE_LABELS)} stages rendered, Stage 0/1 shows real data.")
