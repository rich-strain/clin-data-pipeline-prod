"""Free-text de-identification (Microsoft Presidio) + measured recall.

Two actions on the two kinds of PHI a note carries, mirroring the structured
de-id:
- **Direct identifiers** (name, MRN, address, city, state, ZIP, provider) are
  REMOVED — replaced with a typed placeholder.
- **Dates** (DOB and visit/event dates) are SHIFTED, not removed — by the same
  per-(patient, category) offset the structured de-id uses (deid/dateshift), so
  a date that appears in both the note and the FHIR lands on the same shifted
  value and intra-category intervals survive. DOB uses the ``dob`` offset;
  every other date uses the ``visit`` offset.

The committed transform uses the KNOWN spans (we generated the notes, so we know
every PHI position and category) — deterministic and reproducible. Separately,
Presidio is run to MEASURE how much of that PHI a generalizable NLP de-id would
catch on its own (recall) — the honest answer to "how much would protect unseen
notes?", where a missed date can't be shifted and a missed name can't be
removed. Recall is not inflated by the known-span transform.

Run locally (needs presidio-analyzer + the spaCy model); outputs are committed.
"""

from __future__ import annotations

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

from deid.dateshift import shift_date

Span = tuple[int, int, str]

# Span types that are dates (shifted); everything else is a removable identifier.
DATE_CATEGORY = {"DOB": "dob", "DATE": "visit"}


def build_analyzer() -> AnalyzerEngine:
    analyzer = AnalyzerEngine()
    mrn = PatternRecognizer(
        supported_entity="MRN",
        patterns=[Pattern(name="mrn", regex=r"\bMRN\d{4,}\b", score=0.9)],
    )
    analyzer.registry.add_recognizer(mrn)
    return analyzer


def presidio_spans(analyzer: AnalyzerEngine, text: str) -> list[Span]:
    results = analyzer.analyze(text=text, language="en")
    return [(r.start, r.end, r.entity_type) for r in results]


def apply_note_deid(text: str, phi_spans: list[Span], patient_id: str) -> str:
    """Deterministic committed transform: shift date spans, redact identifiers.

    Ground-truth spans are non-overlapping (we generated them), so a single
    left-to-right pass rebuilds the note.
    """
    out: list[str] = []
    cursor = 0
    for start, end, label in sorted(phi_spans):
        out.append(text[cursor:start])
        category = DATE_CATEGORY.get(label)
        if category is not None:
            out.append(shift_date(patient_id, category, text[start:end]))
        else:
            out.append(f"[{label}]")
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def deidentify_note(
    text: str, phi_spans: list[Span], patient_id: str, analyzer: AnalyzerEngine
) -> tuple[str, list[Span]]:
    """Return (committed de-identified text, presidio-detected spans for recall).

    The committed text applies the deterministic known-span transform (dates
    shifted, identifiers redacted). Presidio is run only to measure recall.
    """
    detected = presidio_spans(analyzer, text)
    deid = apply_note_deid(text, phi_spans, patient_id)
    return deid, detected


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def measure_recall(ground_truth: list[dict], detected: list[Span]) -> dict:
    """Entity-level recall of the Presidio layer against ground-truth PHI.

    A ground-truth span counts as caught if any detected span overlaps it.
    """
    det = [(s, e) for s, e, _ in detected]
    by_type: dict[str, dict[str, int]] = {}
    missed: list[dict] = []
    for gt in ground_truth:
        t = gt["type"]
        by_type.setdefault(t, {"caught": 0, "total": 0})
        by_type[t]["total"] += 1
        if any(_overlaps((gt["start"], gt["end"]), d) for d in det):
            by_type[t]["caught"] += 1
        else:
            missed.append({"type": t, "text": gt["text"]})
    caught = sum(v["caught"] for v in by_type.values())
    total = sum(v["total"] for v in by_type.values())
    return {
        "caught": caught,
        "total": total,
        "recall": round(caught / total, 4) if total else 0.0,
        "by_type": by_type,
        "missed": missed,
    }
