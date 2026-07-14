"""Stage 5 — the frozen, versioned held-out gold set.

The gold set is a frozen snapshot of the Stage 5 **test** split (not a separate
4th carve — at 100 patients that would leave too little for training). Stage 7's
release gate evaluates against this immutable, versioned set so results are
comparable across model versions.

"Frozen" = committed with a content hash and a version. The manifest carries no
timestamp on purpose (byte-reproducible derivation, no git churn on re-run): any
change to the gold content changes its sha256, which `verify_gold` catches — a
deliberate re-freeze then bumps GOLD_VERSION.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
GOLD_PATH = DATA_DIR / "gold" / "gold.jsonl"
GOLD_MANIFEST_PATH = DATA_DIR / "gold" / "gold_manifest.json"

GOLD_VERSION = "v1"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _serialize(examples: list[dict]) -> str:
    return "".join(json.dumps(ex) + "\n" for ex in examples)


def freeze_gold(
    test_examples: list[dict], gold_path: Path = GOLD_PATH, manifest_path: Path = GOLD_MANIFEST_PATH
) -> dict:
    """Write the gold set + its manifest (version, count, sha256). Returns the manifest."""
    content = _serialize(test_examples)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.write_text(content)
    manifest = {
        "version": GOLD_VERSION,
        "n_examples": len(test_examples),
        "sha256": _sha256(content),
        "source": "frozen snapshot of the Stage 5 test split",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_gold(gold_path: Path = GOLD_PATH, manifest_path: Path = GOLD_MANIFEST_PATH) -> bool:
    """True iff the gold file's content hash matches its committed manifest."""
    manifest = json.loads(manifest_path.read_text())
    return _sha256(gold_path.read_text()) == manifest["sha256"]
