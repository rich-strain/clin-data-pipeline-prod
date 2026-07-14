"""FHIR -> OMOP CDM v5.4 transform.

Maps each FHIR resource type to its OMOP clinical-event table, populating both
the standard concept_id (verified, from the pinned snapshot) and the *_source_*
columns (the original code), exactly as a real OMOP ETL does. A blood-pressure
panel Observation becomes TWO measurement rows (systolic + diastolic from its
components), which is how OMOP represents BP — there is no "panel" measurement.
"""

from __future__ import annotations

from datetime import date

import terminology as t
from generation.landing import group_by_patient

SNOMED = "http://snomed.info/sct"
ICD10 = "http://hl7.org/fhir/sid/icd-10-cm"
LOINC = "http://loinc.org"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"

CDM_TABLES = [
    "person",
    "condition_occurrence",
    "drug_exposure",
    "measurement",
    "observation_period",
]


def _coding(codings: list[dict], system: str) -> dict | None:
    return next((c for c in codings if c.get("system") == system), None)


def _birth_parts(birth_date: str | None) -> tuple[int | None, int | None, int | None]:
    if not birth_date:
        return None, None, None
    d = date.fromisoformat(birth_date)
    return d.year, d.month, d.day


def fhir_to_omop(resources: list[dict]) -> dict[str, list[dict]]:
    records = group_by_patient(resources)
    tables: dict[str, list[dict]] = {name: [] for name in CDM_TABLES}

    ids = {
        "person": 0,
        "condition_occurrence": 0,
        "drug_exposure": 0,
        "measurement": 0,
        "observation_period": 0,
    }

    def next_id(table: str) -> int:
        ids[table] += 1
        return ids[table]

    for rec in records.values():
        patient = rec["patient"]
        person_id = next_id("person")
        gender = patient.get("gender")
        year, month, day = _birth_parts(patient.get("birthDate"))
        mrn = (patient.get("identifier") or [{}])[0].get("value", "")
        tables["person"].append(
            {
                "person_id": person_id,
                "gender_concept_id": t.gender_concept_id(gender) if gender else 0,
                "year_of_birth": year,
                "month_of_birth": month,
                "day_of_birth": day,
                "race_concept_id": 0,
                "ethnicity_concept_id": 0,
                "person_source_value": mrn or patient["id"],
                "gender_source_value": gender or "",
            }
        )

        event_dates: list[str] = []

        for cond in rec["conditions"]:
            icd = _coding(cond.get("code", {}).get("coding", []), ICD10)
            concept = t.get_condition(icd["code"]) if icd else None
            start = cond.get("onsetDateTime")
            if start:
                event_dates.append(start)
            tables["condition_occurrence"].append(
                {
                    "condition_occurrence_id": next_id("condition_occurrence"),
                    "person_id": person_id,
                    "condition_concept_id": concept.standard_concept_id if concept else 0,
                    "condition_start_date": start,
                    "condition_type_concept_id": t.type_concept_ehr(),
                    "condition_source_value": icd["code"] if icd else "",
                    "condition_source_concept_id": concept.source_concept_id if concept else 0,
                }
            )

        for med in rec["medications"]:
            rx = _coding(med.get("medicationCodeableConcept", {}).get("coding", []), RXNORM)
            concept = t.get_medication(rx["code"]) if rx else None
            start = med.get("authoredOn")
            if start:
                event_dates.append(start)
            sig = (med.get("dosageInstruction") or [{}])[0].get("text", "")
            tables["drug_exposure"].append(
                {
                    "drug_exposure_id": next_id("drug_exposure"),
                    "person_id": person_id,
                    "drug_concept_id": concept.standard_concept_id if concept else 0,
                    "drug_exposure_start_date": start,
                    "drug_type_concept_id": t.type_concept_ehr(),
                    "drug_source_value": rx["code"] if rx else "",
                    "drug_source_concept_id": concept.source_concept_id if concept else 0,
                    "sig": sig,
                }
            )

        for obs in rec["observations"]:
            mdate = obs.get("effectiveDateTime")
            if mdate:
                event_dates.append(mdate)
            components = obs.get("component")
            if components:  # BP panel -> one measurement row per component
                for comp in components:
                    _emit_measurement(
                        tables,
                        next_id,
                        person_id,
                        comp.get("code", {}),
                        comp.get("valueQuantity", {}),
                        mdate,
                    )
            else:
                _emit_measurement(
                    tables,
                    next_id,
                    person_id,
                    obs.get("code", {}),
                    obs.get("valueQuantity", {}),
                    mdate,
                )

        if event_dates:
            tables["observation_period"].append(
                {
                    "observation_period_id": next_id("observation_period"),
                    "person_id": person_id,
                    "observation_period_start_date": min(event_dates),
                    "observation_period_end_date": max(event_dates),
                    "period_type_concept_id": t.type_concept_ehr(),
                }
            )

    return tables


def _emit_measurement(tables, next_id, person_id, code_obj, value_quantity, mdate) -> None:
    loinc = _coding(code_obj.get("coding", []), LOINC)
    concept = (
        t.get_observation(loinc["code"])
        if (loinc and loinc["code"] in t.list_observations())
        else None
    )
    unit_code = value_quantity.get("code", "")
    tables["measurement"].append(
        {
            "measurement_id": next_id("measurement"),
            "person_id": person_id,
            "measurement_concept_id": concept.standard_concept_id if concept else 0,
            "measurement_date": mdate,
            "measurement_type_concept_id": t.type_concept_ehr(),
            "value_as_number": value_quantity.get("value"),
            "unit_concept_id": t.unit_concept_id(unit_code),
            "measurement_source_value": loinc["code"] if loinc else "",
            "measurement_source_concept_id": concept.source_concept_id if concept else 0,
            "unit_source_value": unit_code,
        }
    )
