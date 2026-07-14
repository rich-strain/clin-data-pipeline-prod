"""Stage 2 — de-identification + governance.

Two complementary techniques for the two data modalities:
- Structured FHIR: deterministic removal of HIPAA §164.514 direct identifiers
  (redact.py) + per-entity, interval-preserving date shift (dateshift.py) — the
  Limited Data Set (LDS + DUA) pattern, NOT Safe Harbor (which would collapse
  dates to year). Documented legally in the app.
- Free-text notes: NLP-based de-id (freetext.py, Microsoft Presidio) layered
  with deterministic removal of the identifiers known from the structured
  record. Recall of the generalizable NLP component is MEASURED against
  ground-truth PHI labels and reported honestly — missed PHI is a breach.

De-id is run locally (heavy NLP model) and its outputs are committed; CI
validates the committed artifacts (leakage == 0, recall report) without
re-running Presidio. See build_spec.md Stage 2.
"""
