"""Loader over the pinned diagnosis -> medication lookup (terminology/dxmed_lookup.json).

The lookup pairs each generated ICD-10 diagnosis with the RxNorm medications that
RxClass says `may_treat` it (scoped to our closed vocabulary), each with a rough
prevalence weight. This module is the one place that reads it, so generation and
any consumer never drift on the pairings. Built by build_dxmed_lookup.py.
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

LOOKUP_PATH = Path(__file__).resolve().parent / "dxmed_lookup.json"


@lru_cache(maxsize=1)
def _raw() -> dict:
    return json.loads(LOOKUP_PATH.read_text())


def medications_for(icd10_code: str) -> list[dict]:
    """The [{rxcui, display, weight}, ...] a diagnosis may be treated with
    (empty if the diagnosis has no in-vocabulary medication)."""
    entry = _raw()["conditions"].get(icd10_code)
    return list(entry["medications"]) if entry else []


def sample_medication(icd10_code: str, rng: random.Random) -> str | None:
    """Weighted-sample one RxNorm code to treat this diagnosis, or None if the
    diagnosis has no in-vocabulary medication (e.g. a code flagged for review)."""
    meds = medications_for(icd10_code)
    if not meds:
        return None
    codes = [m["rxcui"] for m in meds]
    weights = [m["weight"] for m in meds]
    return rng.choices(codes, weights=weights, k=1)[0]


def flags() -> list[dict]:
    """Diagnoses the builder flagged for manual review (no/short/newly-added meds)."""
    return list(_raw()["_flags"])
