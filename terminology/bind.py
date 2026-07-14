"""Terminology binding report (Stage 1).

Walks every coded field in the generated resources and checks whether its
(system, code) resolves to a REAL OMOP standard concept in the pinned snapshot
(terminology/omop_concept_snapshot.json). Reports % of coded fields matched per
vocabulary — the "terminology-binding results" the Stage 0/1 page shows.

A code is "matched" only if it maps to a standard concept (concept_id != 0), so
messy-mode's unmapped `[lb_av]` units show up honestly as a small miss rather
than being silently counted as bound.
"""

from __future__ import annotations

import terminology as t

SNOMED = "http://snomed.info/sct"
ICD10 = "http://hl7.org/fhir/sid/icd-10-cm"
LOINC = "http://loinc.org"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
UCUM = "http://unitsofmeasure.org"


def _lookup_sets() -> dict:
    return {
        "snomed_standard": {
            t.get_condition(c).standard_code
            for c in t.list_conditions()
            if t.get_condition(c).is_mapped
        },
        "icd10": {c for c in t.list_conditions() if t.get_condition(c).is_mapped},
        "loinc": {o for o in t.list_observations() if t.get_observation(o).is_mapped},
        "rxnorm": {m for m in t.list_medications() if t.get_medication(m).is_mapped},
    }


def _matched(system: str, code: str, sets: dict) -> bool:
    if system == SNOMED:
        return code in sets["snomed_standard"]
    if system == ICD10:
        return code in sets["icd10"]
    if system == LOINC:
        return code in sets["loinc"]
    if system == RXNORM:
        return code in sets["rxnorm"]
    if system == UCUM:
        return t.unit_concept_id(code) != 0
    return False


def _codings(resources: list[dict]):
    """Yield (system, code) for every coded field, including UCUM units."""
    for r in resources:
        rtype = r["resourceType"]
        if rtype == "Condition":
            for c in r.get("code", {}).get("coding", []):
                yield c["system"], c["code"]
        elif rtype == "MedicationRequest":
            for c in r.get("medicationCodeableConcept", {}).get("coding", []):
                yield c["system"], c["code"]
        elif rtype == "Observation":
            for c in r.get("code", {}).get("coding", []):
                yield c["system"], c["code"]
            if "valueQuantity" in r:
                q = r["valueQuantity"]
                yield q.get("system", UCUM), q.get("code", "")
            for comp in r.get("component", []):
                for c in comp.get("code", {}).get("coding", []):
                    yield c["system"], c["code"]
                if "valueQuantity" in comp:
                    q = comp["valueQuantity"]
                    yield q.get("system", UCUM), q.get("code", "")


def binding_report(resources: list[dict]) -> dict:
    sets = _lookup_sets()
    by_system: dict[str, dict[str, int]] = {}
    unmatched_examples: list[dict] = []

    for system, code in _codings(resources):
        vocab = system.rsplit("/", 1)[-1]
        by_system.setdefault(vocab, {"matched": 0, "total": 0})
        by_system[vocab]["total"] += 1
        if _matched(system, code, sets):
            by_system[vocab]["matched"] += 1
        elif len(unmatched_examples) < 20:
            unmatched_examples.append({"system": vocab, "code": code})

    total = sum(v["total"] for v in by_system.values())
    matched = sum(v["matched"] for v in by_system.values())
    return {
        "overall": {
            "matched": matched,
            "total": total,
            "pct": round(100 * matched / total, 2) if total else 0.0,
        },
        "by_system": by_system,
        "unmatched_examples": unmatched_examples,
        "snapshot_coverage": t.coverage(),
    }
