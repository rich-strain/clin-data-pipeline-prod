"""Streamlit app shell for the production-true clinical data pipeline.

This is the Working Plan step 1 scaffold: a sidebar-stepper shell with every
stage page stubbed empty. No page shows real data yet — each stage's page is
wired in during the same Working Plan step that produces its data (see
build_spec.md, "Showcase content per stage" and the Working Plan), so the
running app is always a truthful, up-to-date picture of what's actually built.

Like the two prior repos, this app is a read-only showcase of *committed*
artifacts (Lane 3). It never calls the Anthropic API, runs extraction, or
trains — those are deliberate Lane 1 (local/manual) steps.
"""

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent

# (label, one-line description, Working Plan step that fills this page).
# Order mirrors the pipeline; the two trailing "cross-cutting" pages and the
# optional pages land in the step 9 showcase-polish pass.
STAGES: list[tuple[str, str, str]] = [
    (
        "Intro",
        "What this is, why it exists, and how to read it — narrative, architecture "
        "diagram, and the honest-framing note.",
        "populated across the build; polished in step 9",
    ),
    (
        "0/1 — Ingestion + Canonical Storage",
        "Synthetic US Core FHIR (BP as panel + components), terminology binding, "
        "and the flattened feature table.",
        "step 2",
    ),
    (
        "2 — De-identification",
        "Per-entity interval-preserving date shift, structured-field redaction, "
        "free-text de-id with measured recall.",
        "step 3",
    ),
    (
        "3 — Extraction",
        "LLM structured extraction: notes/cache-hit counts, confidence "
        "distribution, and per-record provenance.",
        "step 4",
    ),
    (
        "4 — Curation + DQ",
        "Normalize / redact / rebalance / synthesize, gated by Great "
        "Expectations / Pandera data-quality checks.",
        "step 5",
    ),
    (
        "5 — Dataset Assembly",
        "Group-aware, leakage-safe train/val/test JSONL plus a frozen, versioned gold set.",
        "step 6",
    ),
    (
        "6 — Training",
        "LoRA fine-tune: loss curve, adapter-vs-base size, experiment-run link, "
        "and registry version + best-epoch rationale.",
        "step 7",
    ),
    (
        "7 — Evaluation + Release",
        "Per-field precision/recall/F1, hallucination tracking, raw failing "
        "examples, model card, and the release-gate result.",
        "step 8",
    ),
    (
        "Provenance",
        "The pipeline run/lineage log as a real table: stage, timestamp, in→out "
        "hash, record count, model version + git SHA (Tier 1).",
        "step 9",
    ),
    (
        "Scale & Production Readiness",
        "The honest-gap page: the access/audit-log explainer (Tier 2), the "
        "Delta/Iceberg-vs-Parquet trade-off, and other deferred items.",
        "step 9",
    ),
    (
        "Analytics",
        "The committed Parquet table queried live via duckdb in-process — a real "
        "query, not a screenshot.",
        "step 9 (optional)",
    ),
    (
        "Live Inference",
        "Paste-a-note → structured extraction against the committed LoRA adapter, "
        "in real time (Space only).",
        "step 9",
    ),
]


def render_stub(label: str, description: str, planned_step: str) -> None:
    """Render a not-yet-built stage page honestly, with no fabricated data."""
    st.title(label)
    st.caption(description)
    st.info(
        f"**Stub — not built yet.** This page is wired with real, committed "
        f"data in **Working Plan {planned_step}** (see `build_spec.md`). "
        f"Per this repo's discipline, nothing here is a placeholder number or "
        f"a claimed-but-unrun figure — the page stays empty until its stage "
        f"actually produces a committed artifact."
    )


def main() -> None:
    st.set_page_config(page_title="Clinical Data Pipeline — Production-True", layout="wide")

    labels = [s[0] for s in STAGES]
    with st.sidebar:
        st.header("Clinical Data Pipeline")
        st.caption("Production-true patterns · fully synthetic data")
        choice = st.radio("Stage", labels, label_visibility="collapsed")

    label, description, planned_step = next(s for s in STAGES if s[0] == choice)
    render_stub(label, description, planned_step)


if __name__ == "__main__":
    main()
