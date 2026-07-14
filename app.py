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

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
DATA = ROOT / "data"


@st.cache_data
def load_json(relpath: str):
    path = ROOT / relpath
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_csv(relpath: str):
    path = ROOT / relpath
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_jsonl(relpath: str) -> list[dict]:
    path = ROOT / relpath
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


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


def render_stage01(label: str, description: str) -> None:
    """Stage 0/1 — real data read off the committed Stage 0/1 artifacts."""
    st.title(label)
    st.caption(description)

    bundles = load_json("data/canonical/fhir_bundles.json")
    binding = load_json("data/reports/terminology_binding.json")
    validation = load_json("data/reports/validation.json")
    if not (bundles and binding and validation):
        st.warning("Stage 0/1 artifacts missing — run `python run_stage01.py` to generate them.")
        return

    # --- Immutable raw landing layer ---
    st.subheader("Immutable raw landing layer (NDJSON `$export`)")
    st.caption(
        "The authoritative record — exactly what the source sent. Everything "
        "below is a reproducible derivation FROM this layer, never a second "
        "source of truth."
    )
    counts: dict[str, int] = {}
    for bundle in bundles:
        for entry in bundle["entry"]:
            rt = entry["resource"]["resourceType"]
            counts[rt] = counts.get(rt, 0) + 1
    cols = st.columns(len(counts))
    for col, (rt, n) in zip(cols, sorted(counts.items()), strict=False):
        col.metric(rt, n)

    # --- Sample US Core bundle ---
    st.subheader("Sample US Core FHIR bundle")
    st.caption(
        "Note the two production-true upgrades over the demo: blood pressure is "
        "one Observation (LOINC 85354-9) with **systolic + diastolic "
        "components**, and Conditions are **dual-coded** (SNOMED CT problem + "
        "ICD-10-CM billing)."
    )
    idx = st.slider("Patient", 0, len(bundles) - 1, 0)
    bundle = bundles[idx]
    bp = next(
        (
            e["resource"]
            for e in bundle["entry"]
            if e["resource"]["resourceType"] == "Observation"
            and e["resource"].get("code", {}).get("coding", [{}])[0].get("code") == "85354-9"
        ),
        None,
    )
    if bp:
        with st.expander("Blood pressure panel (LOINC 85354-9) — components", expanded=True):
            st.json(bp)
    with st.expander("Full bundle JSON"):
        st.json(bundle)

    # --- Terminology binding + validation ---
    st.subheader("Terminology binding + conformance")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Coded fields matched",
        f"{binding['overall']['pct']}%",
        f"{binding['overall']['matched']}/{binding['overall']['total']}",
    )
    c2.metric("Base-FHIR valid", f"{validation['valid']}/{validation['total']}")
    c3.metric("US Core-profiled", validation["us_core_profiled"])

    per_system = pd.DataFrame(
        [
            {
                "vocabulary": k,
                "matched": v["matched"],
                "total": v["total"],
                "pct": round(100 * v["matched"] / v["total"], 1),
            }
            for k, v in binding["by_system"].items()
        ]
    )
    st.dataframe(per_system, hide_index=True, use_container_width=True)
    st.caption(
        "Binding uses a pinned value-set snapshot with **real, verified OMOP "
        "concept_ids** (OHDSI ATLAS, not fabricated). The only misses are "
        "messy-mode `[lb_av]` weight units, genuinely absent from the snapshot "
        "and left unmapped (concept_id 0) rather than invented. Full US Core "
        "profile conformance is the HL7 Validator + Inferno (documented "
        "upgrade); this page reports base-FHIR structural validity."
    )

    # --- OMOP CDM ---
    st.subheader("OMOP CDM v5.4 (the analytics model)")
    st.caption(
        "FHIR is transformed into OMOP CDM tables. Standard `*_concept_id`s are "
        "verified; `*_source_value`/`*_source_concept_id` preserve the original "
        "code, exactly as a real ETL does. A BP panel becomes two `measurement` "
        "rows (systolic + diastolic), which is how OMOP stores it."
    )
    tabs = st.tabs(
        ["person", "condition_occurrence", "drug_exposure", "measurement", "observation_period"]
    )
    for tab, name in zip(
        tabs,
        ["person", "condition_occurrence", "drug_exposure", "measurement", "observation_period"],
        strict=True,
    ):
        with tab:
            df = load_csv(f"data/omop/{name}.csv")
            if df is not None:
                st.caption(f"{len(df):,} rows")
                st.dataframe(df.head(15), hide_index=True, use_container_width=True)


def render_deid(label: str, description: str) -> None:
    """Stage 2 — de-identification, read off the committed de-id artifacts."""
    st.title(label)
    st.caption(description)

    raw_notes = load_jsonl("data/notes/raw_notes.jsonl")
    deid_notes = load_jsonl("data/deid/notes_deid.jsonl")
    recall = load_json("data/reports/deid_recall.json")
    leakage = load_json("data/reports/deid_leakage.json")
    raw_patients = {r["id"]: r for r in load_jsonl("data/landing/Patient.ndjson")}
    deid_patients = {
        r["id"]: r
        for r in load_jsonl("data/deid/resources_deid.ndjson")
        if r["resourceType"] == "Patient"
    }
    if not (raw_notes and deid_notes and recall and leakage):
        st.warning(
            "Stage 2 artifacts missing — run `python run_stage2.py` (needs presidio-analyzer)."
        )
        return

    st.info(
        "**Legal frame — HIPAA §164.514.** Dates are **shifted, not removed** — a "
        "per-entity, interval-preserving shift (diagnosis→treatment gaps intact), "
        "applied consistently in **both** the structured FHIR and the note text, "
        "so a date carries the same shifted value in both. **DOB is shifted "
        "independently** of visit/event dates (recovering one date can't unshift "
        "the rest). This is the **Limited Data Set** pattern (dates survive only "
        "under an LDS + DUA) — **not** Safe Harbor, which would collapse dates to "
        "year. Direct identifiers (name, MRN, address) are removed; geography is "
        "generalized to state + 3-digit ZIP. The shift ceiling is `date.today()` "
        "(dynamic), so a shifted date can never land in the future."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Per-patient leakage",
        f"{leakage['total_leaks']}/{leakage['patients']}",
        "raw identifiers surviving de-id",
    )
    c2.metric(
        "Free-text de-id recall (Presidio)",
        f"{recall['recall'] * 100:.1f}%",
        f"{recall['caught']}/{recall['total']} PHI spans",
    )
    c3.metric("Notes de-identified", len(deid_notes))

    # --- Before / after ---
    st.subheader("Before / after — a sample record")
    deid_by_pid = {n["patient_id"]: n for n in deid_notes}
    idx = st.slider("Patient", 0, len(raw_notes) - 1, 0)
    raw_note = raw_notes[idx]
    pid = raw_note["patient_id"]
    left, right = st.columns(2)
    with left:
        st.caption("Raw note (synthetic PHI)")
        st.code(raw_note["text"], language=None)
    with right:
        st.caption("De-identified note (layered: Presidio + known-identifier removal)")
        st.code(deid_by_pid[pid]["text"], language=None)
    if pid in raw_patients and pid in deid_patients:
        pl, pr = st.columns(2)
        keys = ["identifier", "name", "birthDate", "address"]
        pl.caption("Raw Patient")
        pl.json({k: raw_patients[pid].get(k) for k in keys})
        pr.caption(
            "De-identified Patient (name dropped, MRN→pseudonym, DOB shifted, geo generalized)"
        )
        pr.json({k: deid_patients[pid].get(k) for k in keys})

    # --- Measured recall (honest) ---
    st.subheader("Free-text de-id — measured recall, stated honestly")
    st.caption(
        f"Recall of the **generalizable NLP layer alone** ({recall['engine']}) "
        "against the ground-truth PHI labels — the honest measure of what would "
        "protect unseen notes, since a miss is a breach. The committed notes are "
        "safe (0 leaks) because a second deterministic layer removes the "
        "identifiers already known from the structured record; recall here is "
        "*not* inflated by that layer."
    )
    by_type = pd.DataFrame(
        [
            {
                "PHI type": t,
                "caught": v["caught"],
                "total": v["total"],
                "recall %": round(100 * v["caught"] / v["total"], 1),
            }
            for t, v in recall["by_type"].items()
        ]
    ).sort_values("recall %")
    st.dataframe(by_type, hide_index=True, use_container_width=True)
    st.caption(
        "The gaps are real and instructive: Presidio catches names, dates, MRN "
        "(custom recognizer), and provider names well, but under-detects address "
        "components (ZIP, state abbreviations) — which is exactly why a human QA "
        "sample and a layered approach exist. Missed examples:"
    )
    if recall.get("missed_sample"):
        st.dataframe(
            pd.DataFrame(recall["missed_sample"]).head(15),
            hide_index=True,
            use_container_width=True,
        )


def render_extraction(label: str, description: str) -> None:
    """Stage 3 — LLM extraction, read off the committed extractions + cache."""
    st.title(label)
    st.caption(description)

    extractions = load_jsonl("data/extracted/extractions.jsonl")
    cache = load_json("extraction/cache/extraction_cache.json")
    if not extractions:
        st.warning(
            "Stage 3 artifacts missing — run `python -m extraction.extractor` locally "
            "(Lane 1, paid). CI only runs the free cache-only coverage check."
        )
        return

    st.info(
        "**De-identify → then extract (decision #2).** Extraction runs on the "
        "**de-identified** notes, so the text sent to the external model (Anthropic "
        "Haiku) carries no PHI. It's **cache-first**, keyed on a hash of the note "
        "text + model + prompt version: the committed cache means **zero API calls** "
        "on rerun, and CI's coverage check (`--no-api`) is free by construction."
    )

    model = extractions[0].get("model", "?")
    prompt_v = extractions[0].get("prompt_version", "?")
    cache_n = len(cache) if cache else 0
    low = sum(1 for e in extractions if e.get("low_confidence"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Notes extracted", len(extractions))
    c2.metric("Cache coverage", f"{cache_n}/{len(extractions)}", "committed → $0 rerun")
    c3.metric("Low-confidence (HITL)", low, f"< {0.7:.2f}")
    c4.metric("Provenance", model, f"prompt {prompt_v}")

    # --- Confidence distribution ---
    st.subheader("Confidence distribution")
    st.caption(
        "The model's **self-assessed** confidence per record (honestly labeled — the "
        "API exposes no token logprobs). Low-confidence records are flagged for "
        "human-in-the-loop review rather than trusted blindly."
    )
    confs = pd.DataFrame({"confidence": [e.get("confidence", 0.0) for e in extractions]})
    st.bar_chart(confs["confidence"].value_counts().sort_index())

    # --- Sample extraction ---
    st.subheader("Sample extraction (structured tool-use output)")
    idx = st.slider("Record", 0, len(extractions) - 1, 0)
    rec = extractions[idx]
    st.caption(
        f"confidence {rec.get('confidence')} · {rec.get('model')} · {rec.get('prompt_version')}"
    )
    cols = st.columns(3)
    cols[0].caption("Diagnoses")
    cols[0].dataframe(pd.DataFrame(rec["diagnoses"]), hide_index=True, use_container_width=True)
    cols[1].caption("Medications")
    cols[1].dataframe(pd.DataFrame(rec["medications"]), hide_index=True, use_container_width=True)
    cols[2].caption("Vitals")
    cols[2].dataframe(pd.DataFrame(rec["vitals"]), hide_index=True, use_container_width=True)


PAGES = {
    "0/1 — Ingestion + Canonical Storage": render_stage01,
    "2 — De-identification": render_deid,
    "3 — Extraction": render_extraction,
}


def main() -> None:
    st.set_page_config(page_title="Clinical Data Pipeline — Production-True", layout="wide")

    labels = [s[0] for s in STAGES]
    with st.sidebar:
        st.header("Clinical Data Pipeline")
        st.caption("Production-true patterns · fully synthetic data")
        choice = st.radio("Stage", labels, label_visibility="collapsed")

    label, description, planned_step = next(s for s in STAGES if s[0] == choice)
    renderer = PAGES.get(label)
    if renderer:
        renderer(label, description)
    else:
        render_stub(label, description, planned_step)


if __name__ == "__main__":
    main()
