"""Per-entity, interval-preserving date shift (the LDS + DUA pattern).

Each patient gets a single deterministic offset in [-SHIFT_RANGE_DAYS,
+SHIFT_RANGE_DAYS], applied to ALL of that patient's dates. Because every date
for a patient moves by the same amount, every intra-patient interval
(diagnosis -> treatment, DOB -> encounter) is preserved exactly, while absolute
dates are obscured and patients can't be aligned to each other on a calendar.

This is legally a Limited Data Set (dates survive only under an LDS + Data Use
Agreement), NOT Safe Harbor — Safe Harbor would instead collapse dates to year
and aggregate ages > 89. The generator already reduced its date ceiling by
SHIFT_RANGE_DAYS so no shifted date can land in the future (single source of
truth for the constant; see generate_fhir.SHIFT_RANGE_DAYS).
"""

from __future__ import annotations

import copy
import hashlib
from datetime import date, timedelta

from generation.generate_fhir import SHIFT_RANGE_DAYS

# Every field across our resource types whose value is a full YYYY-MM-DD date.
DATE_FIELDS = ("birthDate", "onsetDateTime", "effectiveDateTime", "authoredOn")


def patient_offset(patient_id: str) -> int:
    """Deterministic per-patient shift, NONZERO, in [-SHIFT_RANGE_DAYS, -1] U
    [1, +SHIFT_RANGE_DAYS]. Zero is excluded so a date is never left unshifted
    (which would leak the raw value, e.g. a true DOB)."""
    h = int(hashlib.sha256(patient_id.encode()).hexdigest(), 16)
    r = h % (2 * SHIFT_RANGE_DAYS)  # [0, 2*SHIFT-1]
    return r - SHIFT_RANGE_DAYS if r < SHIFT_RANGE_DAYS else r - SHIFT_RANGE_DAYS + 1


def _patient_id(resource: dict) -> str | None:
    if resource["resourceType"] == "Patient":
        return resource["id"]
    ref = resource.get("subject", {}).get("reference", "")
    return ref.split("/", 1)[1] if "/" in ref else None


def _shift_value(value: str, offset: int) -> str:
    return (date.fromisoformat(value) + timedelta(days=offset)).isoformat()


def shift_resources(resources: list[dict]) -> list[dict]:
    """Return deep-copied resources with every date shifted by the owning
    patient's consistent offset."""
    shifted = copy.deepcopy(resources)
    for resource in shifted:
        pid = _patient_id(resource)
        if pid is None:
            continue
        offset = patient_offset(pid)
        for field in DATE_FIELDS:
            if field in resource and isinstance(resource[field], str):
                resource[field] = _shift_value(resource[field], offset)
    return shifted
