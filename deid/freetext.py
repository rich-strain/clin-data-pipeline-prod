"""Free-text de-identification (Microsoft Presidio) + measured recall.

Layered de-id for clinical notes:
  Layer 1 (deterministic) — remove the identifiers KNOWN from the structured
    record / note metadata (patient name, MRN, DOB, address, encounter dates,
    authoring provider). In production you already hold these; you don't rely
    on NLP to catch the patient's own MRN.
  Layer 2 (generalizable NLP) — Microsoft Presidio catches identifiers you do
    NOT hold structurally (names/dates/locations appearing in narrative). A
    custom recognizer adds the clinical MRN pattern.
The committed de-identified note applies both layers.

Recall is measured for the PRESIDIO layer ALONE against the ground-truth PHI
spans — the honest answer to "how much would the generalizable component catch
on unseen notes?", where a miss is a breach. It is deliberately not inflated by
Layer 1.

Run locally (needs presidio-analyzer + the spaCy model); outputs are committed.
"""

from __future__ import annotations

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

Span = tuple[int, int, str]


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


def _merge(spans: list[Span]) -> list[Span]:
    """Merge overlapping spans; the region keeps the first span's label."""
    merged: list[Span] = []
    for start, end, label in sorted(spans):
        if merged and start <= merged[-1][1]:
            ps, pe, pl = merged[-1]
            merged[-1] = (ps, max(pe, end), pl)
        else:
            merged.append((start, end, label))
    return merged


def redact_text(text: str, spans: list[Span]) -> str:
    """Replace each (merged) span with a typed [LABEL] placeholder."""
    out = []
    cursor = 0
    for start, end, label in _merge(spans):
        out.append(text[cursor:start])
        out.append(f"[{label}]")
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def deidentify_note(
    text: str, known_spans: list[Span], analyzer: AnalyzerEngine
) -> tuple[str, list[Span]]:
    """Return (committed de-identified text, presidio-only spans for recall).

    The committed text redacts BOTH known identifiers and Presidio hits.
    """
    detected = presidio_spans(analyzer, text)
    redacted = redact_text(text, known_spans + detected)
    return redacted, detected


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
