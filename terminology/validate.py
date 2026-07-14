"""FHIR R4 structural validation (Stage 1).

Validates each generated resource against the base FHIR R4 spec using the
pure-Python `fhir.resources` R4B models (free, offline, deterministic, CI-safe).

Scope, stated honestly: this checks base-FHIR structural conformance (element
cardinality, datatypes, value sets that the models enforce). It does NOT check
full US Core profile conformance or terminology-server binding validity — that
is the job of the HL7 FHIR Validator + Inferno (a Java/heavier, documented
upgrade). We separately report how many resources carry a US Core meta.profile
tag so the gap between "base-valid" and "US Core-conformant" is visible, not
hidden.
"""

from __future__ import annotations

from typing import Any

from fhir.resources.R4B.condition import Condition
from fhir.resources.R4B.medicationrequest import MedicationRequest
from fhir.resources.R4B.observation import Observation
from fhir.resources.R4B.patient import Patient

_MODELS: dict[str, Any] = {
    "Patient": Patient,
    "Condition": Condition,
    "Observation": Observation,
    "MedicationRequest": MedicationRequest,
}


def validate_resources(resources: list[dict]) -> dict:
    """Return a structural-validation report over a flat resource list."""
    by_type: dict[str, dict[str, int]] = {}
    failures: list[dict] = []
    profiled = 0

    for r in resources:
        rtype = r["resourceType"]
        by_type.setdefault(rtype, {"valid": 0, "total": 0})
        by_type[rtype]["total"] += 1
        if r.get("meta", {}).get("profile"):
            profiled += 1

        model = _MODELS.get(rtype)
        if model is None:
            failures.append({"resourceType": rtype, "id": r.get("id"), "error": "no R4B model"})
            continue
        try:
            model.model_validate(r)
            by_type[rtype]["valid"] += 1
        except Exception as e:  # noqa: BLE001 — collect, don't crash the report
            failures.append({"resourceType": rtype, "id": r.get("id"), "error": str(e)[:300]})

    total = len(resources)
    valid = sum(v["valid"] for v in by_type.values())
    return {
        "engine": "fhir.resources R4B (base-FHIR structural)",
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "us_core_profiled": profiled,
        "by_type": by_type,
        "failures": failures[:25],
        "note": (
            "Base-FHIR structural validation only. Full US Core profile conformance "
            "and terminology binding validity are the HL7 FHIR Validator + Inferno "
            "(documented upgrade)."
        ),
    }
