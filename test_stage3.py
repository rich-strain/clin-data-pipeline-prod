"""Stage 3 verification (Working Plan step 4 verify gate).

Free, deterministic, CI-safe: never imports/calls the Anthropic SDK. Validates
the committed extractions + cache and exercises the cache-only coverage path
(the same check CI runs), asserting the committed cache covers every committed
de-identified note.

Run: python -m pytest test_stage3.py -v
"""

import json
from pathlib import Path
from types import SimpleNamespace

from extraction.extractor import (
    LOW_CONFIDENCE,
    MODEL,
    PROMPT_VERSION,
    IncrementalCache,
    collect_batch_results,
    extract_notes,
    read_notes,
    submit_batch,
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


class _FakeBatches:
    """Minimal stand-in for client.messages.batches — no Anthropic SDK, so this
    stays CI-safe. Records submitted requests and echoes one succeeded result per
    request (mimicking the decoupled submit -> sleep -> retrieve flow)."""

    def __init__(self) -> None:
        self._submitted: list[dict] = []

    def create(self, requests):
        self._submitted = list(requests)
        return SimpleNamespace(id="batch_test123")

    def retrieve(self, batch_id):
        return SimpleNamespace(processing_status="ended")

    def results(self, batch_id):
        extraction = {"diagnoses": [], "medications": [], "vitals": [], "confidence": 0.9}
        for req in self._submitted:
            tool_use = SimpleNamespace(type="tool_use", input=extraction)
            message = SimpleNamespace(content=[tool_use])
            yield SimpleNamespace(
                custom_id=req["custom_id"],
                result=SimpleNamespace(type="succeeded", message=message),
            )


class _FakeClient:
    def __init__(self) -> None:
        self.messages = SimpleNamespace(batches=_FakeBatches())


def test_batch_submit_does_not_touch_cache_then_retrieve_populates_it(tmp_path) -> None:
    """The decoupled batch path that lets the machine sleep: submit returns a batch
    id WITHOUT caching anything (nothing paid-for is persisted until it finishes),
    and a later retrieve fills the cache so the --no-api write path then succeeds."""
    notes = [
        {"note_id": "n1", "patient_id": "p1", "text": "note one"},
        {"note_id": "n2", "patient_id": "p2", "text": "note two"},
    ]
    cache = IncrementalCache(tmp_path / "cache.json")
    client = _FakeClient()

    batch_id = submit_batch(notes, client, cache)
    assert batch_id == "batch_test123"
    assert len(cache) == 0, "submit must not populate the cache (results aren't ready yet)"

    collect_batch_results(batch_id, client, cache)
    assert len(cache) == 2, "retrieve must populate the cache for every submitted note"

    records = list(extract_notes(notes, client=None, cache=cache, no_api=True))
    assert len(records) == 2 and all(r["confidence"] == 0.9 for r in records)


def test_submit_batch_returns_none_when_everything_cached(tmp_path) -> None:
    notes = [{"note_id": "n1", "patient_id": "p1", "text": "note one"}]
    cache = IncrementalCache(tmp_path / "cache.json")
    client = _FakeClient()
    collect_batch_results(submit_batch(notes, client, cache), client, cache)
    # Second submit sees a full cache -> nothing to do.
    assert submit_batch(notes, client, cache) is None


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
