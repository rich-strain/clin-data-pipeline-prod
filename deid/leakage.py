"""Per-patient leakage check — the actual check, not an assertion.

For each patient, collect their direct identifier VALUES from the raw record
(name, MRN, street address, city, full ZIP, DOB) and search the de-identified
output — both the structured resources and the free-text note — for any literal
occurrence. Any hit is a leak. A correct de-id yields 0/N.
"""

from __future__ import annotations

import json

from generation.landing import group_by_patient


def _patient_identifiers(patient: dict) -> set[str]:
    values: set[str] = set()
    name = (patient.get("name") or [{}])[0]
    for given in name.get("given", []):
        if len(given) > 2:
            values.add(given)
    if name.get("family"):
        values.add(name["family"])
    given_full = " ".join(name.get("given", []))
    if given_full and name.get("family"):
        values.add(f"{given_full} {name['family']}".strip())
    for ident in patient.get("identifier", []):
        if ident.get("value"):
            values.add(ident["value"])
    for addr in patient.get("address", []):
        for line in addr.get("line", []):
            values.add(line)
        if addr.get("city"):
            values.add(addr["city"])
        if addr.get("postalCode"):
            values.add(addr["postalCode"])
    if patient.get("birthDate"):
        values.add(patient["birthDate"])
    return values


def leakage_check(
    raw_resources: list[dict],
    deid_resources: list[dict],
    deid_notes_by_patient: dict[str, str],
) -> dict:
    raw = group_by_patient(raw_resources)
    deid = group_by_patient(deid_resources)

    per_patient: list[dict] = []
    examples: list[dict] = []
    total_leaks = 0

    for pid, rec in raw.items():
        identifiers = _patient_identifiers(rec["patient"])
        haystack = json.dumps(deid.get(pid, {})) + "\n" + deid_notes_by_patient.get(pid, "")
        leaked = sorted(v for v in identifiers if v and v in haystack)
        total_leaks += len(leaked)
        per_patient.append({"patient_id": pid, "leaks": len(leaked)})
        for v in leaked:
            if len(examples) < 20:
                examples.append({"patient_id": pid, "value": v})

    return {
        "patients": len(per_patient),
        "total_leaks": total_leaks,
        "leaked_examples": examples,
    }
