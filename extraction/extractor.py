"""Stage 3 extraction: de-identified notes -> structured clinical fields.

Runs LOCALLY (Lane 1) — it calls the Anthropic API and costs money. Cache-first,
keyed on a hash of (prompt version + model + note text): a committed cache means
zero calls on rerun, and a prompt/model change invalidates only the affected
entries. `anthropic` is imported lazily inside the call path, so `--no-api`
(the CI cache-only coverage check) and mypy never need the SDK and can never
trigger a call.

Output per note: the extracted clinical fields, the model's self-assessed
`confidence` (0-1, honestly labeled as self-assessment — the API exposes no
token logprobs), and provenance (model + prompt_version).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from common.resilient_client import CallStats, IncrementalCache, call_with_retry

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEID_NOTES_PATH = DATA_DIR / "deid" / "notes_deid.jsonl"
EXTRACTED_PATH = DATA_DIR / "extracted" / "extractions.jsonl"
CACHE_PATH = Path(__file__).resolve().parent / "cache" / "extraction_cache.json"

MODEL = "claude-haiku-4-5"
PROMPT_VERSION = "v1"
LOW_CONFIDENCE = 0.7  # below this -> flagged for human-in-the-loop review

EXTRACTION_TOOL: dict[str, Any] = {
    "name": "record_clinical_extraction",
    "description": (
        "Record the clinical fields extracted from a DE-IDENTIFIED clinical note "
        "(PHI is already redacted as [PATIENT_NAME], [DATE], etc. — ignore those "
        "placeholders). Extract diagnoses, medications, and vitals. Preserve the "
        "note's own wording (abbreviations, missing units, shorthand) rather than "
        "normalizing — normalization happens in a later curation step. Also report "
        "an overall confidence in this extraction from 0.0 to 1.0."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "diagnoses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
            "medications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "dosage": {"type": ["string", "null"]},
                    },
                    "required": ["name", "dosage"],
                    "additionalProperties": False,
                },
            },
            "vitals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "value": {"type": "string"},
                        "unit": {"type": ["string", "null"]},
                    },
                    "required": ["name", "value", "unit"],
                    "additionalProperties": False,
                },
            },
            "confidence": {
                "type": "number",
                "description": "Overall self-assessed confidence in this extraction, 0.0-1.0.",
            },
        },
        "required": ["diagnoses", "medications", "vitals", "confidence"],
        "additionalProperties": False,
    },
    "strict": True,
}


def _cache_key(note_text: str) -> str:
    payload = f"{PROMPT_VERSION}\x00{MODEL}\x00{note_text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _new_client():
    """Lazily construct the Anthropic client (import here so --no-api/CI/mypy
    never need the SDK and can never make a call)."""
    import anthropic  # noqa: PLC0415 — deliberate lazy import; see module docstring

    return anthropic.Anthropic()


def extract_fields(client, note_text: str, stats: CallStats | None = None) -> dict:
    response = call_with_retry(
        client.messages.create,
        model=MODEL,
        max_tokens=1024,
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "record_clinical_extraction"},
        messages=[{"role": "user", "content": note_text}],
        stats=stats,
    )
    tool_use = next(b for b in response.content if b.type == "tool_use")
    return dict(tool_use.input)


def read_notes(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _record(note: dict, extracted: dict) -> dict:
    return {
        "patient_id": note["patient_id"],
        "note_id": note["note_id"],
        **extracted,
        "low_confidence": extracted.get("confidence", 1.0) < LOW_CONFIDENCE,
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
    }


def extract_notes(
    notes: Iterable[dict],
    client,
    cache,
    *,
    no_api: bool = False,
    refresh: bool = False,
    stats: CallStats | None = None,
) -> Iterator[dict]:
    """Cache-first extraction. With no_api=True (CI coverage check), a cache miss
    is a hard failure instead of an API call."""
    for note in notes:
        key = _cache_key(note["text"])
        if refresh or key not in cache:
            if no_api:
                raise RuntimeError(
                    f"cache miss for note {note['note_id']} with --no-api: the committed "
                    f"cache does not cover every committed note (run extraction locally)."
                )
            cache[key] = extract_fields(client, note["text"], stats=stats)
        yield _record(note, cache[key])


def _batch_request(key: str, note_text: str) -> dict:
    return {
        "custom_id": key,
        "params": {
            "model": MODEL,
            "max_tokens": 1024,
            "tools": [EXTRACTION_TOOL],
            "tool_choice": {"type": "tool", "name": "record_clinical_extraction"},
            "messages": [{"role": "user", "content": note_text}],
        },
    }


def extract_notes_batch(
    notes: Iterable[dict],
    client,
    cache,
    *,
    refresh: bool = False,
    stats: CallStats | None = None,
    poll_interval: float = 10.0,
    timeout: float = 1800.0,
    sleep=time.sleep,
) -> Iterator[dict]:
    """Submit all uncached notes as one Message Batch (50% cheaper; used for the
    1000-record scale-up). Deduplicates identical note text by cache key."""
    notes = list(notes)
    to_submit: dict[str, str] = {}
    for note in notes:
        key = _cache_key(note["text"])
        if refresh or key not in cache:
            to_submit.setdefault(key, note["text"])

    if to_submit:
        requests = [_batch_request(k, t) for k, t in to_submit.items()]
        batch = call_with_retry(client.messages.batches.create, requests=requests, stats=stats)
        start = time.monotonic()
        while True:
            batch = call_with_retry(client.messages.batches.retrieve, batch.id, stats=stats)
            if batch.processing_status == "ended":
                break
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"batch {batch.id} unfinished after {timeout}s")
            sleep(poll_interval)
        responses = call_with_retry(
            lambda: list(client.messages.batches.results(batch.id)), stats=stats
        )
        failed = []
        for r in responses:
            if r.result.type == "succeeded":
                tool_use = next(b for b in r.result.message.content if b.type == "tool_use")
                cache[r.custom_id] = dict(tool_use.input)
            else:
                failed.append((r.custom_id, r.result.type))
        if failed:
            raise RuntimeError(f"{len(failed)} batch request(s) did not succeed: {failed}")

    for note in notes:
        yield _record(note, cache[_cache_key(note["text"])])


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 extraction (Lane 1, paid — cache-first).")
    parser.add_argument("--in", dest="in_path", type=Path, default=DEID_NOTES_PATH)
    parser.add_argument("--out", type=Path, default=EXTRACTED_PATH)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--mode", choices=["sequential", "batch"], default="sequential")
    parser.add_argument(
        "--refresh", action="store_true", help="Re-call the API even for cached notes"
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Cache-only coverage check: never call the API; fail on any cache miss (CI, Lane 2).",
    )
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--batch-timeout", type=float, default=1800.0)
    args = parser.parse_args()

    notes = list(read_notes(args.in_path))
    cache = IncrementalCache(args.cache)
    stats = CallStats()

    # Only build a real client when we might actually call the API. Load the
    # local .env (ANTHROPIC_API_KEY) here, guarded so the CI --no-api coverage
    # check runs without python-dotenv installed.
    client = None
    if not args.no_api:
        try:
            from dotenv import load_dotenv  # noqa: PLC0415 — local-only, optional

            load_dotenv()
        except ImportError:
            pass
        client = _new_client()

    if args.mode == "batch" and not args.no_api:
        extractions = extract_notes_batch(
            notes,
            client,
            cache,
            refresh=args.refresh,
            stats=stats,
            poll_interval=args.poll_interval,
            timeout=args.batch_timeout,
        )
    else:
        extractions = extract_notes(
            notes, client, cache, no_api=args.no_api, refresh=args.refresh, stats=stats
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.out.open("w") as f:
        for extraction in extractions:
            f.write(json.dumps(extraction) + "\n")
            count += 1

    mode = "no-api coverage check" if args.no_api else args.mode
    print(f"Wrote {count} extractions to {args.out} (cache: {len(cache)} entries, mode: {mode})")
    if not args.no_api:
        print(f"Call stats: {stats.summary()}")


if __name__ == "__main__":
    main()
