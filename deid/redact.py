"""Structured-field redaction of HIPAA §164.514 direct identifiers.

Removes the direct identifiers carried on the Patient resource (name, MRN,
street/city, full ZIP) and replaces the MRN with a stable research pseudonym
so records stay linkable without exposing PHI. Kept: gender and the (date-
shifted) birthDate — allowed under the Limited Data Set. Geography is
generalized to state + 3-digit ZIP, per Safe Harbor's geographic rule.

The other resource types carry no direct identifiers — their `subject`
reference is the patient's random surrogate UUID (already de-identified), and
their dates are handled by the interval-preserving shift (dateshift.py).
"""

from __future__ import annotations

import copy
import hashlib

RESEARCH_ID_SYSTEM = "urn:example:research-id"


def pseudonym(patient_id: str) -> str:
    """Stable research id for a patient (same in -> same out, not reversible)."""
    return "DEID-" + hashlib.sha256(patient_id.encode()).hexdigest()[:10].upper()


def _redact_patient(patient: dict) -> dict:
    patient.pop("name", None)
    patient["identifier"] = [{"system": RESEARCH_ID_SYSTEM, "value": pseudonym(patient["id"])}]
    addresses = patient.get("address")
    if addresses:
        addr = addresses[0]
        zip3 = (addr.get("postalCode", "")[:3] + "**") if addr.get("postalCode") else None
        generalized = {"state": addr.get("state", "")}
        if zip3:
            generalized["postalCode"] = zip3
        patient["address"] = [generalized]
    return patient


def redact_resources(resources: list[dict]) -> list[dict]:
    """Return deep-copied resources with Patient direct identifiers removed."""
    out = copy.deepcopy(resources)
    for resource in out:
        if resource["resourceType"] == "Patient":
            _redact_patient(resource)
    return out
