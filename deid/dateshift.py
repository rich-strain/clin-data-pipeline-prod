"""Per-entity, interval-preserving date shift (the LDS + DUA pattern).

Each patient gets a deterministic offset **per date category**, applied to all
of that patient's dates in that category. Two categories:

- ``dob``   — the date of birth
- ``visit`` — every visit/event date (condition onset, observation effective,
  medication authored, encounter/visit date in notes)

Because every date in a category moves by the same amount, intra-category
intervals are preserved exactly (diagnosis -> treatment gaps, visit -> visit
spacing), while absolute dates are obscured. DOB is shifted **independently**
from visit dates: a single shared offset would let anyone who recovers one true
date (e.g. a known birthdate) unshift every other date by simple subtraction —
independent category offsets mean recovering one category reveals nothing about
the other. (This is the clin-data-pipeline-scale pattern.)

Legally a Limited Data Set (dates survive only under an LDS + Data Use
Agreement), NOT Safe Harbor. The generator bounds event dates to
``today - SHIFT_RANGE_DAYS`` (a *dynamic* ceiling, not a hard-coded one — see
generate_fhir.TRUE_CEILING), so the maximum positive shift can never push a date
into the future.
"""

from __future__ import annotations

import copy
import hashlib
from datetime import date, timedelta

from generation.generate_fhir import SHIFT_RANGE_DAYS

# Which shift category each date field belongs to. DOB is its own category so it
# shifts independently of the visit/event dates.
FIELD_CATEGORY = {
    "birthDate": "dob",
    "onsetDateTime": "visit",
    "effectiveDateTime": "visit",
    "authoredOn": "visit",
}
DATE_FIELDS = tuple(FIELD_CATEGORY)


def patient_offset(patient_id: str, category: str) -> int:
    """Deterministic per-(patient, category) shift, NONZERO, in
    [-SHIFT_RANGE_DAYS, -1] U [1, +SHIFT_RANGE_DAYS]. Zero is excluded so a date
    is never left unshifted (which would leak the raw value)."""
    h = int(hashlib.sha256(f"{patient_id}:{category}".encode()).hexdigest(), 16)
    r = h % (2 * SHIFT_RANGE_DAYS)  # [0, 2*SHIFT-1]
    return r - SHIFT_RANGE_DAYS if r < SHIFT_RANGE_DAYS else r - SHIFT_RANGE_DAYS + 1


def shift_date(patient_id: str, category: str, date_str: str) -> str:
    """Shift one ISO date string by the patient's offset for that category."""
    offset = patient_offset(patient_id, category)
    return (date.fromisoformat(date_str) + timedelta(days=offset)).isoformat()


def _patient_id(resource: dict) -> str | None:
    if resource["resourceType"] == "Patient":
        return resource["id"]
    ref = resource.get("subject", {}).get("reference", "")
    return ref.split("/", 1)[1] if "/" in ref else None


def shift_resources(resources: list[dict]) -> list[dict]:
    """Return deep-copied resources with every date shifted by the owning
    patient's per-category offset."""
    shifted = copy.deepcopy(resources)
    for resource in shifted:
        pid = _patient_id(resource)
        if pid is None:
            continue
        for field, category in FIELD_CATEGORY.items():
            if isinstance(resource.get(field), str):
                resource[field] = shift_date(pid, category, resource[field])
    return shifted
