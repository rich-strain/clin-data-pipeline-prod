# Clinical Data Pipeline — Production-True

Third companion repo in the clinical-data-pipeline portfolio arc. Where the
first repo (`../clin-data-pipeline`) demonstrates the **domain** end to end and
the second (`../clin-data-pipeline-scale`) demonstrates **running the
LLM-calling parts at volume**, this repo demonstrates the **production-true
patterns** the first two deliberately stubbed: the governance spine
(de-identification, BAAs, audit), orchestration, data/model versioning and
lineage, data-quality gates, and a real serving/analytics layer — built where
credible on synthetic data, documented honestly where a real deployment is the
only place they can exist.

> **Fully synthetic data only, ever.** No real patient data under any
> circumstance. Honesty over polish: no invented benchmarks, no claimed
> capability without a real run behind it.

The full architecture, the Tier 1 / Tier 2 split, the three-lane execution
model, the cost guardrails, and the numbered Working Plan live in
[`build_spec.md`](build_spec.md). This README is a stub that grows with the
build.

## Execution lanes (why nothing costs money on a push)

- **Lane 1 — local / manual:** the paid steps (Stage 3 extraction, Stage 4
  synthesize) and compute-heavy step (Stage 6 LoRA training). Run by hand where
  the credentials and hardware live; outputs are committed.
- **Lane 2 — CI on every push (free):** lint, type-check, tests, the free
  pipeline stages against committed inputs, DQ gates, and a cache-only
  extraction coverage check. **Never** calls the Anthropic API. **Never**
  trains.
- **Lane 3 — hosted showcase / serving:** a read-only Streamlit app over
  committed artifacts, plus live inference on the committed adapter.

**Cost guardrail:** `ANTHROPIC_API_KEY` never enters GitHub/CI. CI has no
credential to bill with, so no bug or loop can spend money. CI even enforces
this — a workflow that references the key fails the build.

## Build status

Built incrementally, one verified Working Plan step per commit.

- [x] **Step 1 — Repo scaffold + CI skeleton + app shell.** Project structure,
  CI (lint/type-check/tests, cost guardrail, secret-gated Space deploy), and a
  Streamlit sidebar-stepper `app.py` with every stage page stubbed empty.
- [x] **Step 2 — Stage 0/1: FHIR generation + terminology binding.** US Core
  FHIR (BP panel + components, dual-coded conditions), immutable NDJSON landing
  layer, verified-OMOP terminology binding + `fhir.resources` validation, and
  the OMOP CDM ETL.
- [x] **Step 3 — Stage 2: de-identification framework.** Interval-preserving
  date shift (LDS) + structured redaction, layered free-text NLP de-id
  (Presidio) with measured recall, and the per-patient leakage check (0/N).
- [x] **Step 4 — Stage 3: LLM extraction (Lane 1, paid).** Cache-first structured
  extraction (Haiku) from the *de-identified* notes, with confidence scoring +
  provenance; committed `extractions.jsonl` + cache; CI runs a free `--no-api`
  cache-only coverage check.
- [ ] Step 5 — Stage 4: curation + DQ gates
- [ ] Step 6 — Stage 5: dataset assembly
- [ ] Step 7 — Stage 6: training + model lifecycle (Lane 1, compute)
- [ ] Step 8 — Stage 7: evaluation + release gate
- [ ] Step 9 — Showcase polish + serving
- [ ] Step 10 — Scale to 1000 records (Lane 1, paid, one-time)

## Develop

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

streamlit run app.py          # the showcase app
python -m pytest              # tests (incl. headless app check)
ruff check . && ruff format --check .

# Rebuild committed artifacts (free, deterministic):
python run_stage01.py         # Stage 0/1: FHIR + terminology + OMOP CDM

# Stage 2 free-text de-id needs the heavy NLP model (local only; CI validates
# the committed outputs instead of re-running it):
pip install -r requirements-deid.txt
python run_stage2.py          # Stage 2: de-identification + recall + leakage
```
