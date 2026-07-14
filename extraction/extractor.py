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
# Transient handoff between `--mode batch-submit` and `--mode batch-retrieve`, so
# the machine can sleep in between. Not committed (see .gitignore).
BATCH_STATE_PATH = Path(__file__).resolve().parent / "cache" / "pending_batch.json"

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


def submit_batch(
    notes: Iterable[dict],
    client,
    cache,
    *,
    refresh: bool = False,
    stats: CallStats | None = None,
) -> str | None:
    """Create a Message Batch for every uncached note and return its id WITHOUT
    waiting for it. Returns None if every note is already cached (nothing to do).

    This is the half that lets the machine sleep: it submits and returns
    immediately — the API processes the batch server-side (24h target), so the
    caller can quit, sleep, and `collect_batch_results` later. Deliberately does
    NOT touch the cache: nothing is paid-for-and-persisted until retrieval.
    """
    to_submit: dict[str, str] = {}
    for note in notes:
        key = _cache_key(note["text"])
        if refresh or key not in cache:
            to_submit.setdefault(key, note["text"])
    if not to_submit:
        return None
    requests = [_batch_request(k, t) for k, t in to_submit.items()]
    batch = call_with_retry(client.messages.batches.create, requests=requests, stats=stats)
    return batch.id


def batch_status(batch_id: str, client, *, stats: CallStats | None = None) -> str:
    """The processing_status of a submitted batch ('in_progress' | 'ended' | ...)."""
    batch = call_with_retry(client.messages.batches.retrieve, batch_id, stats=stats)
    return batch.processing_status


def collect_batch_results(
    batch_id: str, client, cache, *, stats: CallStats | None = None
) -> None:
    """Pull a FINISHED batch's results into the cache (persisted per-write). Raises
    if the batch is not finished yet (retrieve again later) or any request failed."""
    if batch_status(batch_id, client, stats=stats) != "ended":
        raise RuntimeError(f"batch {batch_id} is not finished yet — retrieve again later")
    responses = call_with_retry(
        lambda: list(client.messages.batches.results(batch_id)), stats=stats
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
    """Submit all uncached notes as one Message Batch and BLOCK until done (50%
    cheaper; deduplicates identical note text by cache key). Use this when you'll
    stay awake for the run; for a walk-away run that lets the machine sleep, use
    submit_batch + collect_batch_results (`--mode batch-submit`/`batch-retrieve`)."""
    notes = list(notes)
    batch_id = submit_batch(notes, client, cache, refresh=refresh, stats=stats)
    if batch_id is not None:
        start = time.monotonic()
        while batch_status(batch_id, client, stats=stats) != "ended":
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"batch {batch_id} unfinished after {timeout}s")
            sleep(poll_interval)
        collect_batch_results(batch_id, client, cache, stats=stats)

    for note in notes:
        yield _record(note, cache[_cache_key(note["text"])])


def _save_batch_state(batch_id: str, in_path: Path) -> None:
    BATCH_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BATCH_STATE_PATH.write_text(
        json.dumps(
            {
                "batch_id": batch_id,
                "in_path": str(in_path),
                "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        )
    )


def _load_batch_state() -> dict | None:
    if BATCH_STATE_PATH.exists():
        return json.loads(BATCH_STATE_PATH.read_text())
    return None


def _write_extractions(notes: list[dict], cache, out_path: Path) -> int:
    """Write one extraction line per note straight from the cache (no API). Raises
    on any cache miss — the same coverage guarantee as the --no-api path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w") as f:
        for extraction in extract_notes(notes, client=None, cache=cache, no_api=True):
            f.write(json.dumps(extraction) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 extraction (Lane 1, paid — cache-first).")
    parser.add_argument("--in", dest="in_path", type=Path, default=DEID_NOTES_PATH)
    parser.add_argument("--out", type=Path, default=EXTRACTED_PATH)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument(
        "--mode",
        choices=["sequential", "batch", "batch-submit", "batch-retrieve"],
        default="sequential",
        help=(
            "sequential/batch run to completion; batch-submit fires a batch and "
            "exits (machine may then sleep); batch-retrieve pulls a finished batch."
        ),
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Batch id for --mode batch-retrieve (defaults to the last submitted).",
    )
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

    # Decoupled batch: submit-and-exit, then retrieve-later, so the machine can
    # sleep in between (the batch runs server-side).
    if args.mode == "batch-submit":
        batch_id = submit_batch(notes, client, cache, refresh=args.refresh, stats=stats)
        if batch_id is None:
            count = _write_extractions(notes, cache, args.out)
            print(f"Every note already cached — wrote {count} extractions to {args.out}.")
        else:
            _save_batch_state(batch_id, args.in_path)
            print(
                f"Submitted batch {batch_id}. You can let the machine sleep now.\n"
                f"When it finishes (server-side, up to 24h), run:\n"
                f"  python -m extraction.extractor --mode batch-retrieve"
            )
        print(f"Call stats: {stats.summary()}")
        return

    if args.mode == "batch-retrieve":
        state = _load_batch_state()
        batch_id = args.batch_id or (state or {}).get("batch_id")
        if not batch_id:
            raise SystemExit("No --batch-id given and no pending batch state found.")
        status = batch_status(batch_id, client, stats=stats)
        if status != "ended":
            print(f"Batch {batch_id} is still '{status}' — not ready. Retrieve again later.")
            return
        collect_batch_results(batch_id, client, cache, stats=stats)
        count = _write_extractions(notes, cache, args.out)
        BATCH_STATE_PATH.unlink(missing_ok=True)
        print(f"Retrieved batch {batch_id}: wrote {count} extractions to {args.out}.")
        print(f"Call stats: {stats.summary()}")
        return

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
