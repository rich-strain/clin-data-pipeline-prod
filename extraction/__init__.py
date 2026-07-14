"""Stage 3 — LLM structured extraction (Lane 1, paid, local).

Extracts clinical fields (diagnoses / medications / vitals) from the
DE-IDENTIFIED notes — de-id ran first (decision #2), so the text sent to the
external model carries no PHI. Cache-first (committed cache = zero calls on
rerun); adds confidence scoring + provenance (model + prompt version) per
record. CI only ever runs the cache-only coverage check (`--no-api`); it never
imports or calls the Anthropic SDK.
"""
