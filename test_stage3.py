"""Stage 3 verification (Working Plan step 4 verify gate).

Free, deterministic, CI-safe: never imports/calls the Anthropic SDK. Validates
the committed extractions + cache and exercises the cache-only coverage path
(the same check CI runs), asserting the committed cache covers every committed
de-identified note.

Run: python -m pytest test_stage3.py -v
"""

import json
from pathlib import Path

from extraction.extractor import (
    LOW_CONFIDENCE,
    MODEL,
    PROMPT_VERSION,
    IncrementalCache,
    extract_notes,
    read_notes,
)

ROOT = Path(__file__).parent
DATA = ROOT / "data"
CACHE_PATH = ROOT / "extraction" / "cache" / "extraction_cache.json"


def _committed_extractions() -> list[dict]:
    path = DATA / "extracted" / "extractions.jsonl"
    assert path.exists(), "committed extractions missing — run `python -m extraction.extractor`"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_cache_covers_every_committed_note_no_api() -> None:
    """The CI coverage check: cache-only extraction must succeed for every note."""
    notes = list(read_notes(DATA / "deid" / "notes_deid.jsonl"))
    cache = IncrementalCache(CACHE_PATH)
    # no_api=True raises on any cache miss; reaching the end means full coverage.
    records = list(extract_notes(notes, client=None, cache=cache, no_api=True))
    assert len(records) == len(notes) == 100


def test_every_extraction_has_provenance_and_valid_confidence() -> None:
    for e in _committed_extractions():
        assert e["model"] == MODEL and e["prompt_version"] == PROMPT_VERSION
        assert 0.0 <= e["confidence"] <= 1.0
        assert e["low_confidence"] == (e["confidence"] < LOW_CONFIDENCE)


def test_extraction_shape_and_alignment_with_notes() -> None:
    ext = _committed_extractions()
    note_ids = {
        json.loads(line)["note_id"]
        for line in (DATA / "deid" / "notes_deid.jsonl").read_text().splitlines()
        if line.strip()
    }
    assert {e["note_id"] for e in ext} == note_ids, "extractions must align 1:1 with de-id notes"
    for e in ext:
        assert isinstance(e["diagnoses"], list)
        assert all("name" in d for d in e["diagnoses"])
        assert all({"name", "dosage"} <= m.keys() for m in e["medications"])
        assert all({"name", "value", "unit"} <= v.keys() for v in e["vitals"])
