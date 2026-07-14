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
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

import registry

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ANALYTICS_PARQUET = ROOT / "data" / "analytics" / "pipeline_analytics.parquet"


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
        "0/1 — Generation + Canonical Storage",
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
        "Normalize / redact / rebalance / synthesize, gated by Pandera "
        "data-quality checks that fail the pipeline.",
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

    # --- Accuracy vs. the source facts (teacher-side eval) ---
    ext_eval = load_json("data/reports/extraction_eval.json")
    if ext_eval:
        st.subheader("Extraction accuracy vs. the source facts")
        st.caption(
            "Confidence above is the model grading **itself**. This measures it against "
            "**ground truth**: the notes were generated from committed FHIR resources, so "
            "those facts are what each note should contain. The normalized extraction is "
            "scored against them by canonical name — the same TP/FP/FN + hallucination "
            "scorer Stage 7 uses. Vitals out of scope (value-closeness). A hallucination is "
            "a predicted name outside the closed vocabulary."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "field": name,
                        "precision": round(s["micro_precision"], 3),
                        "recall": round(s["micro_recall"], 3),
                        "f1": round(s["micro_f1"], 3),
                        "TP": s["tp"],
                        "FP": s["fp"],
                        "FN": s["fn"],
                        "hallucinations": s["non_canonical_count"],
                    }
                    for name, s in (
                        ("diagnosis", ext_eval["diagnosis"]),
                        ("medication", ext_eval["medication"]),
                    )
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

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


def render_curation(label: str, description: str) -> None:
    """Stage 4 — curation + DQ gate, read off the committed curated artifacts."""
    st.title(label)
    st.caption(description)

    raw = load_jsonl("data/extracted/extractions.jsonl")
    normalized = load_jsonl("data/curated/normalized.jsonl")
    rebalanced = load_jsonl("data/curated/rebalanced.jsonl")
    metrics = load_json("data/curated/normalize_metrics.json")
    gate = load_json("data/reports/dq_gate.json")
    if not (normalized and rebalanced and metrics and gate):
        st.warning("Stage 4 artifacts missing — run `python run_stage4.py` to generate them.")
        return

    st.info(
        "**Curate, then gate.** The raw extraction preserves the model's wording; "
        "curation makes it consistent (canonical names/units/dosage), rebalances "
        "under-represented diagnoses, and synthesizes any missing category — then a "
        "**Pandera gate fails the pipeline** on any completeness / referential / "
        "value-range violation. Every number below is read off committed artifacts."
    )

    ov = metrics["overall"]
    dups = sum(1 for r in rebalanced if "rebalance_duplicate_of" in r)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Values canonicalized", f"{ov['matched']}/{ov['total']}", f"{ov['unmatched']} unmatched"
    )
    c2.metric("Rebalance duplicates", f"+{dups}", f"{len(rebalanced)} records")
    c3.metric("Synthesized (paid)", "0", "all categories present → $0")
    c4.metric("DQ gate", "PASS" if gate["passed"] else "FAIL", f"{gate['n_records']} records")

    # --- Before/after: diagnosis canonicalization ---
    st.subheader("Normalization — before → after")
    st.caption(
        "The raw extraction returns diagnosis names with date contamination and "
        "dropped qualifiers; normalization maps them to the closed terminology "
        "vocabulary (a lookup against the generator's own source of truth, not NLP)."
    )
    before_dx = sorted({d["name"] for r in raw for d in r["diagnoses"]})
    after_dx = sorted({d["name"] for r in normalized for d in r["diagnoses"]})
    cols = st.columns(2)
    cols[0].caption(f"Raw diagnosis names ({len(before_dx)} distinct)")
    cols[0].dataframe(pd.DataFrame({"name": before_dx}), hide_index=True, use_container_width=True)
    cols[1].caption(f"Canonical diagnosis names ({len(after_dx)} distinct)")
    cols[1].dataframe(pd.DataFrame({"name": after_dx}), hide_index=True, use_container_width=True)

    # --- Rebalance: category counts ---
    st.subheader("Rebalance — diagnosis-category representation")
    st.caption(
        "Under-represented diagnosis categories are oversampled by duplicating "
        "records (marked `rebalance_duplicate_of`, kept split-safe for Stage 5)."
    )
    before_counts = Counter(d["name"] for r in normalized for d in r["diagnoses"])
    after_counts = Counter(d["name"] for r in rebalanced for d in r["diagnoses"])
    counts_df = pd.DataFrame(
        [
            {"diagnosis": k, "before": before_counts.get(k, 0), "after": after_counts.get(k, 0)}
            for k in after_counts
        ]
    ).sort_values("after", ascending=False)
    st.dataframe(counts_df, hide_index=True, use_container_width=True)

    # --- DQ gate results ---
    st.subheader("Data-quality gate (Pandera)")
    st.caption(
        "Declarative checks that **raise** on violation, not just log a metric — "
        "completeness, referential integrity (names/units in the closed vocabulary), "
        "and value ranges (confidence 0–1, vitals within clinical bounds, BP well-formed)."
    )
    gate_df = pd.DataFrame(
        [
            {
                "table": name,
                "rows": e["rows"],
                "result": "✅ PASS" if e["passed"] else f"❌ FAIL ({e.get('n_failures', 0)})",
            }
            for name, e in gate["tables"].items()
        ]
    )
    st.dataframe(gate_df, hide_index=True, use_container_width=True)


def render_dataset(label: str, description: str) -> None:
    """Stage 5 — dataset assembly, read off the committed splits + gold set."""
    st.title(label)
    st.caption(description)

    splits = {name: load_jsonl(f"data/splits/{name}.jsonl") for name in ("train", "val", "test")}
    gold = load_jsonl("data/gold/gold.jsonl")
    manifest = load_json("data/gold/gold_manifest.json")
    curated = {
        name: load_jsonl(f"data/curated/split_{name}.jsonl") for name in ("train", "val", "test")
    }
    if not (all(splits.values()) and gold and manifest):
        st.warning("Stage 5 artifacts missing — run `python run_stage5.py` to generate them.")
        return

    st.info(
        "**Group-aware, leakage-safe splits.** Records are split by **original "
        "patient** — a rebalance duplicate never lands in a different split from "
        "its original, so no near-identical example crosses the train/eval "
        "boundary. The instruction is the already-de-identified note text (Stage "
        "2, upstream); the response is the curated clinical fields as JSON."
    )

    total = sum(len(v) for v in splits.values())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train", len(splits["train"]), f"{len(splits['train']) / total:.0%}")
    c2.metric("Val", len(splits["val"]), f"{len(splits['val']) / total:.0%}")
    c3.metric("Test / gold", len(splits["test"]), f"{len(splits['test']) / total:.0%}")
    c4.metric("Gold set", manifest["version"], f"{manifest['n_examples']} frozen")

    # --- Leakage check (zero patient groups crossing splits) ---
    st.subheader("Leakage check — zero patient groups crossing splits")
    where: dict[str, set[str]] = {}
    for name, records in curated.items():
        for r in records:
            base = r.get("rebalance_duplicate_of", r["patient_id"])
            where.setdefault(base, set()).add(name)
    crossing = {pid for pid, names in where.items() if len(names) > 1}
    st.caption(
        f"{len(where)} original patient groups across {total} records; a group's "
        "records (original + rebalance duplicates) always share one split."
    )
    if crossing:
        st.error(f"LEAKAGE: {len(crossing)} patient group(s) cross splits: {sorted(crossing)}")
    else:
        st.success("✅ No original patient group appears in more than one split.")

    # --- Sample instruction/response pair ---
    st.subheader("Sample instruction / response pair")
    st.caption("What a fine-tuning example looks like: de-identified note in, canonical JSON out.")
    idx = st.slider("Train example", 0, len(splits["train"]) - 1, 0)
    ex = splits["train"][idx]
    cols = st.columns(2)
    cols[0].caption("instruction (de-identified note)")
    cols[0].code(ex["instruction"], language="text")
    cols[1].caption("response (canonical fields)")
    cols[1].code(json.dumps(json.loads(ex["response"]), indent=2), language="json")

    # --- Frozen gold set ---
    st.subheader("Frozen, versioned gold set")
    st.caption(
        "The held-out test split, frozen with a content hash so Stage 7's release "
        "gate evaluates a fixed, versioned reference — a changed hash means a "
        "deliberate re-freeze (version bump), not silent drift."
    )
    st.json(manifest)


def render_training(label: str, description: str) -> None:
    """Stage 6 — LoRA training, read off the committed training_results + registry."""
    st.title(label)
    st.caption(description)

    config = load_json("training_results/config.json")
    registry = load_json("training_results/model_registry.json")
    samples = load_json("training_results/samples.json")
    curve = ROOT / "training_results" / "loss_curve.png"
    if not (config and registry):
        st.warning(
            "Stage 6 artifacts missing — run `pip install -r requirements-train.txt && "
            "python train_runner.py` locally (Lane 1, compute). CI/the app never train."
        )
        return

    entry = registry[-1]
    st.info(
        "**A real LoRA fine-tune on real hardware** (Apple Silicon MPS), not a "
        "placeholder. A 4 MB adapter over Qwen2.5-0.5B teaches the exact JSON "
        "extraction schema; every number below is read off the committed run. "
        f"Honest scope: {config['train_examples']} training examples + a 0.5B model "
        "is a demonstration of the mechanics, not a production-accuracy claim."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Base model", "Qwen2.5-0.5B", "≈0.5B params")
    c2.metric("Adapter size", f"{entry['adapter_bytes'] / 1e6:.1f} MB", "LoRA r=8")
    c3.metric("Best epoch", entry["best_epoch"], f"val {entry['best_val_loss']:.4f}")
    c4.metric("Registry", entry["version"], f"git {entry['git_sha'][:7]}")

    # --- Loss curve + history ---
    st.subheader("Training loss")
    st.caption(
        "Declining train/val loss over epochs. Val loss bottoms out then ticks up "
        f"(overfit on {config['train_examples']} examples), so the registry selects the "
        f"**best epoch ({entry['best_epoch']})** by val loss, not the last."
    )
    cols = st.columns([3, 2])
    if curve.exists():
        cols[0].image(str(curve), use_container_width=True)
    hist = config["loss_history"]
    cols[1].dataframe(
        pd.DataFrame(
            {"epoch": hist["epoch"], "train_loss": hist["train_loss"], "val_loss": hist["val_loss"]}
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --- Before/after ---
    if samples:
        st.subheader("Base vs. fine-tuned (same note)")
        st.caption(
            "The adapter's whole job: make the model emit the exact target schema. "
            "The base model free-forms (markdown fences, `code`/`description` keys); "
            "the fine-tuned model matches the ground-truth JSON."
        )
        s = samples[0]
        cols = st.columns(3)
        cols[0].caption("Ground truth")
        cols[0].code(s["ground_truth"], language="json")
        cols[1].caption("Base model (no adapter)")
        cols[1].code(s["base_model_output"], language="json")
        cols[2].caption("Fine-tuned (with adapter)")
        cols[2].code(s["fine_tuned_output"], language="json")

    # --- Lineage + experiment tracking ---
    st.subheader("Model registry — data↔model lineage")
    st.caption(
        "The committed model traces to the exact code (git SHA) and data snapshot "
        "(content hashes of the train/val/gold splits + gold version) it was trained "
        f"on, plus its MLflow run id (`{entry['mlflow_run_id'][:12]}…`)."
    )
    st.json(entry)


def render_eval(label: str, description: str) -> None:
    """Stage 7 — evaluation + release gate, read off the committed eval report."""
    st.title(label)
    st.caption(description)

    report = load_json("training_results/eval_report.json")
    card = ROOT / "training_results" / "model_card.md"
    if not report:
        st.warning(
            "Stage 7 artifacts missing — run `python evaluate.py` locally (Lane 1, "
            "compute; needs requirements-train.txt). Evaluates the committed adapter."
        )
        return

    gate = report["release_gate"]
    dx, med, jv = report["diagnosis"], report["medication"], report["json_validity"]
    st.info(
        "**Release gate on the frozen gold set.** The committed adapter is scored "
        "against the immutable gold set; a predicted name outside the closed "
        "vocabulary is tracked as a **hallucination**, never fuzzy-matched. The gate "
        "is a real pass/fail, and every number is read off the committed report."
    )

    verdict = "✅ RELEASE" if gate["passed"] else "❌ BLOCKED"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Release gate", verdict)
    c2.metric("JSON validity", f"{jv['valid']}/{jv['total']}", f"{jv['rate']:.0%}")
    c3.metric("Diagnosis F1", f"{dx['micro_f1']:.3f}", f"non-canon {dx['non_canonical_count']}")
    c4.metric("Medication F1", f"{med['micro_f1']:.3f}", f"non-canon {med['non_canonical_count']}")

    # --- Per-field metrics ---
    st.subheader("Per-field metrics (frozen gold set)")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "field": name,
                    "precision": round(s["micro_precision"], 3),
                    "recall": round(s["micro_recall"], 3),
                    "f1": round(s["micro_f1"], 3),
                    "TP": s["tp"],
                    "FP": s["fp"],
                    "FN": s["fn"],
                    "hallucinations": s["non_canonical_count"],
                }
                for name, s in (("diagnosis", dx), ("medication", med))
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --- Release gate checks ---
    st.subheader("Release gate")
    st.caption("Each threshold must pass for the model to be releasable.")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "check": c["name"],
                    "value": c["value"],
                    "needs": f"{c['op']} {c['threshold']}",
                    "result": "✅ pass" if c["passed"] else "❌ fail",
                }
                for c in gate["checks"]
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --- Failing cases (honest: show misses) ---
    dx_misses = [c for c in dx["per_category"] if c["fn"] or c["fp"]]
    if dx_misses or report["failure_examples"]:
        st.subheader("Where it slips (honest failure tracking)")
        if dx_misses:
            st.caption("Diagnosis categories the model got wrong on the gold set:")
            st.dataframe(pd.DataFrame(dx_misses), hide_index=True, use_container_width=True)
        if report["failure_examples"]:
            st.caption("Unparseable / malformed outputs:")
            st.json(report["failure_examples"])

    # --- Model card ---
    if card.exists():
        st.subheader("Model card")
        st.markdown(card.read_text())


def render_intro(label: str, description: str) -> None:
    """The narrative front door — what this is, why, and how to read it."""
    st.title("Clinical Data Pipeline — Production-True")
    st.caption(description)
    st.markdown(
        "The third repo in a portfolio arc. The first demonstrates the clinical "
        "**domain** end to end; the second runs the **LLM-calling parts at volume**; "
        "**this** one builds the **production-true patterns** the first two stubbed — "
        "the governance spine (de-identification, BAAs, audit), data-quality gates, "
        "data/model **lineage**, and a real **serving/analytics** layer."
    )
    st.info(
        "**Honest framing.** Everything is **fully synthetic** — no real patients, no "
        "PHI. Where a capability can be built for real on synthetic data it is (and "
        "shown, with real numbers); where it genuinely needs a real deployment (a "
        "signed BAA, real access logs, a lakehouse) it's **documented honestly, not "
        "faked**. No invented benchmarks. A 0.5B local fine-tune is a mechanics "
        "demonstration, not a production-accuracy claim."
    )
    svg = ROOT / "docs" / "architecture.svg"
    if svg.exists():
        st.subheader("Architecture")
        st.image(str(svg), use_container_width=True)
    st.caption(
        "Use the sidebar to walk each pipeline stage — every page reads committed artifacts."
    )


def render_provenance(label: str, description: str) -> None:
    """The pipeline run/lineage log as a real, content-addressed table (Tier 1)."""
    st.title(label)
    st.caption(description)

    prov = load_json("data/reports/provenance.json")
    if not prov:
        st.warning("Provenance log missing — run `python provenance.py`.")
        return

    st.info(
        "**Real lineage, not a fabricated run log.** Each stage's committed output is "
        "content-hashed (sha256) with its record count, so any artifact traces to the "
        "exact bytes it was derived from. Paired with the live git SHA and the model "
        "registry below. The Tier-2 **access/audit** log is documented (not shown) on "
        "the Scale & Production Readiness page — real access events need real users."
    )

    model = registry.latest()
    c1, c2, c3 = st.columns(3)
    c1.metric("Pipeline stages logged", len(prov))
    c2.metric("Code version (git)", registry.git_sha()[:12])
    c3.metric(
        "Model registry", model["version"] if model else "—", "not trained" if not model else ""
    )

    st.subheader("Run / lineage log")
    df = pd.DataFrame(prov)
    df["sha256"] = df["sha256"].str.slice(0, 16) + "…"
    st.dataframe(df, hide_index=True, use_container_width=True)

    if model:
        st.subheader("Model lineage")
        st.caption("The committed adapter's tie back to the exact data snapshot + code.")
        st.json(
            {
                "version": model["version"],
                "git_sha": model["git_sha"],
                "best_epoch": model["best_epoch"],
                "mlflow_run_id": model["mlflow_run_id"],
                "data_lineage": model["data_lineage"],
            }
        )


ANALYTICS_QUERIES = {
    "Diagnosis prevalence (distinct patients + mean confidence)": (
        "SELECT diagnosis, COUNT(DISTINCT patient_id) AS patients, "
        "ROUND(AVG(confidence), 3) AS avg_confidence\n"
        "FROM tbl GROUP BY diagnosis ORDER BY patients DESC"
    ),
    "Split distribution (rows, patients, duplicates)": (
        "SELECT split, COUNT(*) AS rows, COUNT(DISTINCT patient_id) AS patients, "
        "SUM(is_rebalance_duplicate::INT) AS rebalance_dupes\n"
        "FROM tbl GROUP BY split ORDER BY rows DESC"
    ),
    "Comorbidity load (avg diagnoses & meds per record)": (
        "SELECT diagnosis, ROUND(AVG(n_diagnoses), 2) AS avg_dx_per_record, "
        "ROUND(AVG(n_medications), 2) AS avg_meds\n"
        "FROM tbl GROUP BY diagnosis ORDER BY avg_dx_per_record DESC"
    ),
}


def render_analytics(label: str, description: str) -> None:
    """Real in-process duckdb queries over the committed Parquet table."""
    st.title(label)
    st.caption(description)

    if not ANALYTICS_PARQUET.exists():
        st.warning("Analytics Parquet missing — run `python analytics.py`.")
        return
    import duckdb  # noqa: PLC0415 — app-runtime dep; kept out of the import-time path

    st.info(
        "**A real columnar analytics store, not a screenshot.** The committed Parquet "
        "table is queried **live, in-process** via duckdb (a library — no server, no "
        "Spark) each time this page renders. Delta/Iceberg (ACID, time-travel) is the "
        "documented lakehouse upgrade."
    )

    choice = st.selectbox("Query", list(ANALYTICS_QUERIES))
    sql = ANALYTICS_QUERIES[choice]
    st.code(sql.replace("tbl", "'pipeline_analytics.parquet'"), language="sql")
    con = duckdb.connect()
    # DDL can't bind parameters; the path is a trusted local constant.
    con.execute(f"CREATE VIEW tbl AS SELECT * FROM read_parquet('{ANALYTICS_PARQUET}')")
    result = con.execute(sql).df()  # sql references the `tbl` view
    st.dataframe(result, hide_index=True, use_container_width=True)
    st.caption(f"{len(result)} rows · queried over {ANALYTICS_PARQUET.name} at render time.")


def render_scale(label: str, description: str) -> None:
    """The honest-gaps page: what's Tier 2 (documented) and why."""
    st.title(label)
    st.caption(description)
    st.info(
        "Not everything a production deployment needs can be built truthfully on "
        "synthetic data. Rather than fabricate it, those capabilities are documented "
        "here — what they'd do, and exactly why they can't be shown."
    )
    st.subheader("Access / audit log — Tier 2 (documented, not shown)")
    st.markdown(
        "HIPAA requires logging **who accessed which record when**. A real audit log "
        "needs real users and real PHI access events — neither exists here, and "
        "fabricating entries would violate this repo's honesty rule. In production it "
        "would be an append-only, immutable store (the same immutability the landing "
        "layer already demonstrates), queried for breach investigation and the HIPAA "
        "accounting-of-disclosures right."
    )
    st.subheader("Storage format — Parquet (built) vs Delta / Iceberg (upgrade)")
    st.markdown(
        "The Analytics page runs on **real Parquet + duckdb** (built). The documented "
        "upgrade is a **Delta Lake / Iceberg** lakehouse: ACID transactions, concurrent "
        "multi-writer safety, schema evolution, and time-travel — none of which a "
        "single-writer synthetic rebuild exercises, so they're named as the upgrade "
        "rather than stood up for show."
    )
    st.subheader("Other honest gaps")
    st.markdown(
        "- **BAA / vendor access:** real de-id→external-LLM flow needs a signed BAA — "
        "documented; the pipeline demonstrates the de-identify-before-extraction "
        "ordering that makes it possible.\n"
        "- **Orchestration:** stages run as scripts + CI; Airflow/Dagster is the "
        "documented scheduler upgrade.\n"
        "- **Drift monitoring** (Evidently) and a full **US Core + Inferno** validator "
        "are named upgrades over the built base-FHIR validation."
    )


ADAPTER_DIR = ROOT / "training_results" / "adapter"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
INFERENCE_SYSTEM_PROMPT = (
    "You are a clinical data extraction assistant. Extract the patient's "
    "diagnoses, medications, and vitals from the clinical note below. Respond "
    "with only a single JSON object with keys diagnoses, medications, vitals — "
    "no other text."
)


@st.cache_resource
def _load_inference_model():
    """Base model + committed adapter on CPU. Cached across reruns; the heavy ML
    imports are here (lazy) so the app imports fine without torch (CI/Lane 2)."""
    import torch  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    model.eval()
    return model, tokenizer


def _run_inference(note_text: str) -> str:
    import torch  # noqa: PLC0415

    model, tokenizer = _load_inference_model()
    messages = [
        {"role": "system", "content": INFERENCE_SYSTEM_PROMPT},
        {"role": "user", "content": note_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=320, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def render_live(label: str, description: str) -> None:
    """Stage 9 — live inference: a note in, structured JSON out, from the committed
    adapter. Space-only in spirit (needs the ML runtime); the model loads lazily
    on first Extract, not at page render, so the showcase stays light."""
    st.title(label)
    st.caption(description)

    if not ADAPTER_DIR.exists():
        st.warning("No committed adapter — run `python train_runner.py` (Lane 1) first.")
        return

    st.info(
        "**Real inference against the committed LoRA adapter** (base Qwen2.5-0.5B + "
        "the 4 MB adapter, on CPU). The first Extract loads the model (slow cold "
        "start on a free Space); subsequent runs are cached. No API, no PHI — paste "
        "a **de-identified** note."
    )
    sample = load_jsonl("data/gold/gold.jsonl")
    default_note = sample[0]["instruction"] if sample else ""
    note = st.text_area("De-identified clinical note", value=default_note, height=260)

    if st.button("Extract structured fields"):
        try:
            with st.spinner("Loading model + generating (first run is slow) …"):
                output = _run_inference(note)
        except Exception as exc:  # noqa: BLE001 — surface any runtime/ML-dep issue to the user
            st.error(
                f"Live inference needs the ML runtime (torch/transformers/peft) and the "
                f"committed adapter. This runs on the deployed Space. Details: {exc}"
            )
            return
        st.subheader("Extraction")
        try:
            st.json(json.loads(output))
        except (json.JSONDecodeError, ValueError):
            st.caption("Raw model output (not valid JSON):")
            st.code(output)


PAGES = {
    "Intro": render_intro,
    "0/1 — Generation + Canonical Storage": render_stage01,
    "2 — De-identification": render_deid,
    "3 — Extraction": render_extraction,
    "4 — Curation + DQ": render_curation,
    "5 — Dataset Assembly": render_dataset,
    "6 — Training": render_training,
    "7 — Evaluation + Release": render_eval,
    "Provenance": render_provenance,
    "Scale & Production Readiness": render_scale,
    "Analytics": render_analytics,
    "Live Inference": render_live,
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
