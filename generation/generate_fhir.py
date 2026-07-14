"""Synthetic US Core-shaped FHIR R4 generation (Stage 0/1).

Produces per-patient resource groups (Patient + Conditions + Observations +
MedicationRequests) as the system of record. Everything downstream (the
immutable NDJSON landing layer, the OMOP CDM, the notes in Stage 2/3) is
derived from these resources, not the other way around.

This is the production-true upgrade of the demo generator, fixing the three
simplifications the demo made and documented:
  1. Blood pressure is ONE Observation (LOINC 85354-9) with `component`
     entries for systolic (8480-6) + diastolic (8462-4) — not a flat
     standalone systolic Observation.
  2. Conditions are dual-coded: a SNOMED CT problem code (US Core primary)
     AND the ICD-10-CM billing code — not ICD-10 alone.
  3. Resources carry `meta.profile` US Core profile URLs and the US Core
     must-have elements (category, clinicalStatus/verificationStatus, UCUM
     units).
All codes come from the verified pinned terminology snapshot (terminology/),
so codes/concept_ids never drift from the OMOP mapping.

`messy=True` injects realistic EHR problems (inconsistent units, missing
optional fields, free-text dosage shorthand) for Stage 4 curation to fix.
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import terminology as t

SNOMED_SYSTEM = "http://snomed.info/sct"
ICD10_SYSTEM = "http://hl7.org/fhir/sid/icd-10-cm"
LOINC_SYSTEM = "http://loinc.org"
RXNORM_SYSTEM = "http://www.nlm.nih.gov/research/umls/rxnorm"
UCUM_SYSTEM = "http://unitsofmeasure.org"

US_CORE = "http://hl7.org/fhir/us/core/StructureDefinition"

# Stage 2's per-entity date shift (deid/dateshift) moves event dates by up to
# +/- SHIFT_RANGE_DAYS. Generation must never emit a date whose post-shift value
# could exceed "now", so event dates are bounded to TRUE_CEILING - SHIFT_RANGE_DAYS.
# TRUE_CEILING is DYNAMIC (date.today()) — not a hard-coded constant — so a
# rebuild can never leave a shifted date in the future, and the landing honestly
# reflects "captured at time T". Defined here (the producer); dateshift imports
# SHIFT_RANGE_DAYS, so the shift range and the generation bound never drift.
SHIFT_RANGE_DAYS = 365
TRUE_CEILING = date.today()
GENERATION_CEILING = TRUE_CEILING - timedelta(days=SHIFT_RANGE_DAYS)

FIRST_NAMES_MALE = [
    "James",
    "Robert",
    "Michael",
    "Brian",
    "David",
    "Joshua",
    "Joseph",
    "Thomas",
    "Charles",
    "Daniel",
    "Matthew",
    "Stanley",
    "Mark",
    "Paul",
    "Steven",
]
FIRST_NAMES_FEMALE = [
    "Mary",
    "Patricia",
    "Jennifer",
    "Linda",
    "Elizabeth",
    "Barbara",
    "Susan",
    "Jessica",
    "Sarah",
    "Michelle",
    "Lauren",
    "Lisa",
    "Stephanie",
    "Betty",
    "Amanda",
]
LAST_NAMES = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Baker",
    "Miller",
    "Davis",
    "Webber",
    "Stevens",
    "Simpson",
    "Lee",
    "Whitney",
    "Wilson",
    "Anderson",
    "Thomas",
    "Taylor",
    "Moore",
    "Jackson",
    "Martin",
]
STREET_NAMES = [
    "Main St",
    "Oak Ave",
    "Maple Dr",
    "Cedar Ln",
    "Elm St",
    "Pine Rd",
    "Washington Ave",
    "Park Blvd",
    "Sunset Dr",
    "Lake St",
]
CITIES_STATES = [
    ("Springfield", "IL"),
    ("Franklin", "TX"),
    ("Greenville", "SC"),
    ("Clinton", "OH"),
    ("Salem", "OR"),
    ("Georgetown", "KY"),
    ("Arlington", "VA"),
    ("Madison", "WI"),
    ("Bristol", "CT"),
    ("Fairview", "NC"),
]


def _c_to_f(c: float) -> float:
    return round(c * 9 / 5 + 32, 1)


def _kg_to_lb(kg: float) -> float:
    return round(kg * 2.20462, 1)


def _cm_to_in(cm: float) -> float:
    return round(cm / 2.54, 1)


# Clinical generation params keyed by RxNorm ingredient code (the coded value +
# display come from the terminology snapshot; strength/form + dosage live here).
MED_CONTENT = {
    "6809": {
        "label": "Metformin 500 MG Oral Tablet",
        "short": "500mg PO BID",
        "full": "Take 500 mg by mouth twice daily",
    },
    "29046": {
        "label": "Lisinopril 10 MG Oral Tablet",
        "short": "10mg PO QD",
        "full": "Take 10 mg by mouth once daily",
    },
    "83367": {
        "label": "Atorvastatin 20 MG Oral Tablet",
        "short": "20mg PO QHS",
        "full": "Take 20 mg by mouth at bedtime",
    },
    "435": {
        "label": "Albuterol 90 MCG Inhaler",
        "short": "2puff q4h PRN",
        "full": "Inhale 2 puffs every 4 hours as needed",
    },
    "10582": {
        "label": "Levothyroxine 75 MCG Oral Tablet",
        "short": "75mcg PO QAM",
        "full": "Take 75 mcg by mouth every morning",
    },
    "36437": {
        "label": "Sertraline 50 MG Oral Tablet",
        "short": "50mg PO QD",
        "full": "Take 50 mg by mouth once daily",
    },
    "7646": {
        "label": "Omeprazole 20 MG Oral Capsule",
        "short": "20mg PO QD",
        "full": "Take 20 mg by mouth once daily",
    },
    "17767": {
        "label": "Amlodipine 5 MG Oral Tablet",
        "short": "5mg PO QD",
        "full": "Take 5 mg by mouth once daily",
    },
    "69120": {
        "label": "Tiotropium 18 MCG Inhalation Powder",
        "short": "1cap inh QD",
        "full": "Inhale the contents of 1 capsule once daily",
    },
    "37418": {
        "label": "Sumatriptan 50 MG Oral Tablet",
        "short": "50mg PO PRN",
        "full": "Take 50 mg by mouth at onset of migraine; may repeat once after 2 hours",
    },
    "8152": {
        "label": "Phentermine 37.5 MG Oral Tablet",
        "short": "37.5mg PO QAM",
        "full": "Take 37.5 mg by mouth every morning before breakfast",
    },
}

# Simple (single-value) vital signs + the one lab, keyed by LOINC. BP is
# handled separately as a panel. `category`: US Core vital-signs vs laboratory.
# `alt_unit`/`convert`: the messy-mode inconsistent-unit swap.
SIMPLE_OBS: dict[str, dict[str, Any]] = {
    "8867-4": {
        "low": 55,
        "high": 100,
        "unit": "/min",
        "alt_unit": None,
        "convert": None,
        "category": "vital-signs",
        "profile": f"{US_CORE}/us-core-heart-rate",
    },
    "8310-5": {
        "low": 36.1,
        "high": 37.8,
        "unit": "Cel",
        "alt_unit": "[degF]",
        "convert": _c_to_f,
        "category": "vital-signs",
        "profile": f"{US_CORE}/us-core-body-temperature",
    },
    "29463-7": {
        "low": 55,
        "high": 110,
        "unit": "kg",
        "alt_unit": "[lb_av]",
        "convert": _kg_to_lb,
        "category": "vital-signs",
        "profile": f"{US_CORE}/us-core-body-weight",
    },
    "8302-2": {
        "low": 150,
        "high": 190,
        "unit": "cm",
        "alt_unit": "[in_i]",
        "convert": _cm_to_in,
        "category": "vital-signs",
        "profile": f"{US_CORE}/us-core-body-height",
    },
    "2339-0": {
        "low": 70,
        "high": 180,
        "unit": "mg/dL",
        "alt_unit": None,
        "convert": None,
        "category": "laboratory",
        "profile": f"{US_CORE}/us-core-observation-lab",
    },
}
BP_PANEL = "85354-9"
BP_SYSTOLIC = "8480-6"
BP_DIASTOLIC = "8462-4"


def _random_date(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randint(0, max((end - start).days, 0)))


def _new_id(rng: random.Random) -> str:
    """Deterministic resource id drawn from the seeded stream, so a rebuild is
    byte-for-byte reproducible (this repo's reproducible-derivation discipline),
    unlike an unseeded uuid4."""
    return str(uuid.UUID(int=rng.getrandbits(128)))


def _obs_category(code: str) -> dict:
    system = "http://terminology.hl7.org/CodeSystem/observation-category"
    return {"coding": [{"system": system, "code": code}]}


def _make_address(rng: random.Random) -> dict:
    city, state = rng.choice(CITIES_STATES)
    return {
        "line": [f"{rng.randint(100, 9999)} {rng.choice(STREET_NAMES)}"],
        "city": city,
        "state": state,
        "postalCode": f"{rng.randint(10000, 99999)}",
    }


def make_patient(rng: random.Random, messy: bool) -> tuple[dict, str, date]:
    gender = rng.choice(["male", "female"])
    given = rng.choice(FIRST_NAMES_MALE if gender == "male" else FIRST_NAMES_FEMALE)
    family = rng.choice(LAST_NAMES)
    birth_date = _random_date(rng, date(1940, 1, 1), date(2005, 12, 31))
    patient_id = _new_id(rng)
    addr_rng = random.Random(patient_id)  # deterministic given the seeded patient_id

    resource = {
        "resourceType": "Patient",
        "id": patient_id,
        "meta": {"profile": [f"{US_CORE}/us-core-patient"]},
        "identifier": [
            {
                "system": "urn:oid:2.16.840.1.113883.19.5.99999.1",  # synthetic MRN authority
                "value": f"MRN{rng.randint(100000, 999999)}",
            }
        ],
        "name": [{"family": family, "given": [given]}],
        "gender": gender,
        "birthDate": birth_date.isoformat(),
        "address": [_make_address(addr_rng)],
    }
    if messy and rng.random() < 0.15:
        del resource["gender"]
    if messy and addr_rng.random() < 0.15:
        del resource["identifier"]
    if messy and addr_rng.random() < 0.15:
        del resource["address"]
    return resource, patient_id, birth_date


def make_condition(rng: random.Random, patient_id: str, birth_date: date, messy: bool) -> dict:
    icd10 = rng.choice(t.list_conditions())
    c = t.get_condition(icd10)
    onset = _random_date(rng, max(birth_date, date(2015, 1, 1)), GENERATION_CEILING)

    # Primary coding SNOMED CT (US Core problem), secondary ICD-10-CM (billing).
    codings = []
    if c.is_mapped:
        codings.append(
            {"system": SNOMED_SYSTEM, "code": c.standard_code, "display": c.standard_name}
        )
    codings.append({"system": ICD10_SYSTEM, "code": c.source_code, "display": c.source_name})

    resource = {
        "resourceType": "Condition",
        "id": _new_id(rng),
        "meta": {"profile": [f"{US_CORE}/us-core-condition-problems-health-concerns"]},
        "clinicalStatus": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    "code": "active",
                }
            ]
        },
        "verificationStatus": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                    "code": "confirmed",
                }
            ]
        },
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                        "code": "problem-list-item",
                    }
                ]
            }
        ],
        "code": {"coding": codings, "text": c.standard_name or c.source_name},
        "subject": {"reference": f"Patient/{patient_id}"},
        "onsetDateTime": onset.isoformat(),
    }
    if messy and rng.random() < 0.2:
        del resource["clinicalStatus"]
    return resource


def _quantity(value: float, ucum: str) -> dict:
    return {"value": value, "unit": ucum, "system": UCUM_SYSTEM, "code": ucum}


def make_bp_observation(rng: random.Random, patient_id: str, birth_date: date, messy: bool) -> dict:
    systolic = round(rng.uniform(100, 140), 0)
    diastolic = round(rng.uniform(60, 90), 0)
    effective = _random_date(rng, max(birth_date, date(2020, 1, 1)), GENERATION_CEILING)
    panel = t.get_observation(BP_PANEL)
    sys_c = t.get_observation(BP_SYSTOLIC)
    dia_c = t.get_observation(BP_DIASTOLIC)
    return {
        "resourceType": "Observation",
        "id": _new_id(rng),
        "meta": {"profile": [f"{US_CORE}/us-core-blood-pressure"]},
        "status": "final",
        "category": [_obs_category("vital-signs")],
        "code": {
            "coding": [
                {"system": LOINC_SYSTEM, "code": panel.source_code, "display": panel.source_name}
            ],
            "text": "Blood pressure panel",
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": effective.isoformat(),
        "component": [
            {
                "code": {
                    "coding": [
                        {
                            "system": LOINC_SYSTEM,
                            "code": sys_c.source_code,
                            "display": sys_c.source_name,
                        }
                    ]
                },
                "valueQuantity": _quantity(systolic, "mm[Hg]"),
            },
            {
                "code": {
                    "coding": [
                        {
                            "system": LOINC_SYSTEM,
                            "code": dia_c.source_code,
                            "display": dia_c.source_name,
                        }
                    ]
                },
                "valueQuantity": _quantity(diastolic, "mm[Hg]"),
            },
        ],
    }


def make_simple_observation(
    rng: random.Random, patient_id: str, birth_date: date, messy: bool
) -> dict:
    loinc = rng.choice(list(SIMPLE_OBS.keys()))
    spec = SIMPLE_OBS[loinc]
    obs = t.get_observation(loinc)
    value = round(rng.uniform(spec["low"], spec["high"]), 1)
    unit = spec["unit"]
    if messy and spec["convert"] and rng.random() < 0.4:
        value = spec["convert"](value)
        unit = spec["alt_unit"]
    effective = _random_date(rng, max(birth_date, date(2020, 1, 1)), GENERATION_CEILING)

    resource = {
        "resourceType": "Observation",
        "id": _new_id(rng),
        "meta": {"profile": [spec["profile"]]},
        "status": "final",
        "category": [_obs_category(spec["category"])],
        "code": {
            "coding": [
                {"system": LOINC_SYSTEM, "code": obs.source_code, "display": obs.source_name}
            ],
            "text": obs.source_name,
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": effective.isoformat(),
        "valueQuantity": _quantity(value, unit),
    }
    if messy and rng.random() < 0.15:
        del resource["effectiveDateTime"]
    return resource


def _condition_icd10(condition: dict) -> str:
    """The ICD-10-CM code a Condition is dual-coded with (used to look up meds)."""
    for c in condition["code"]["coding"]:
        if c["system"] == ICD10_SYSTEM:
            return c["code"]
    raise ValueError(f"Condition {condition.get('id')} has no ICD-10-CM coding")


def make_medication_request(
    rng: random.Random, patient_id: str, birth_date: date, messy: bool, condition: dict
) -> dict | None:
    """A MedicationRequest that plausibly treats `condition`: the RxNorm code is
    weighted-sampled from the diagnosis's RxClass may_treat lookup, and
    reasonReference points back at the Condition (both share the Patient subject).
    Returns None if the diagnosis has no in-vocabulary medication (a code flagged
    in dxmed_lookup.json for review)."""
    icd10 = _condition_icd10(condition)
    rxnorm = t.sample_medication(icd10, rng)
    if rxnorm is None:
        return None
    med = t.get_medication(rxnorm)
    content = MED_CONTENT[rxnorm]
    authored = _random_date(rng, max(birth_date, date(2020, 1, 1)), GENERATION_CEILING)
    dosage_text = content["short"] if (messy and rng.random() < 0.5) else content["full"]

    resource = {
        "resourceType": "MedicationRequest",
        "id": _new_id(rng),
        "meta": {"profile": [f"{US_CORE}/us-core-medicationrequest"]},
        "status": "active",
        "intent": "order",
        "medicationCodeableConcept": {
            "coding": [
                {"system": RXNORM_SYSTEM, "code": med.source_code, "display": med.source_name}
            ],
            "text": content["label"],
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "authoredOn": authored.isoformat(),
        "dosageInstruction": [{"text": dosage_text}],
        "reasonReference": [{"reference": f"Condition/{condition['id']}"}],
    }
    if messy and rng.random() < 0.15:
        del resource["dosageInstruction"]
    return resource


def generate_patient_resources(rng: random.Random, messy: bool) -> list[dict]:
    """One patient's full resource group; Patient resource is always first."""
    patient, patient_id, birth_date = make_patient(rng, messy)
    resources = [patient]
    conditions = [
        make_condition(rng, patient_id, birth_date, messy) for _ in range(rng.randint(1, 3))
    ]
    resources.extend(conditions)
    resources.append(make_bp_observation(rng, patient_id, birth_date, messy))
    for _ in range(rng.randint(1, 4)):
        resources.append(make_simple_observation(rng, patient_id, birth_date, messy))
    # One medication per condition, plausibly treating it (dx->med lookup +
    # reasonReference), instead of drugs drawn at random. A condition whose
    # diagnosis has no in-vocabulary medication simply gets none.
    for cond in conditions:
        med = make_medication_request(rng, patient_id, birth_date, messy, cond)
        if med is not None:
            resources.append(med)
    return resources


def generate_dataset(n_patients: int, messy: bool = False, seed: int = 42) -> list[list[dict]]:
    """List of per-patient resource groups (deterministic given seed)."""
    rng = random.Random(seed)
    return [generate_patient_resources(rng, messy) for _ in range(n_patients)]


def as_bundle(resources: list[dict]) -> dict:
    """Wrap one patient's resources in a collection Bundle (for display/inspection)."""
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": r} for r in resources],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic US Core FHIR patient resources."
    )
    parser.add_argument("--n-patients", type=int, default=100)
    parser.add_argument("--messy", action="store_true", help="Introduce realistic messiness")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/canonical/fhir_bundles.json"))
    args = parser.parse_args()

    dataset = generate_dataset(args.n_patients, messy=args.messy, seed=args.seed)
    bundles = [as_bundle(r) for r in dataset]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundles, indent=2))
    print(
        f"Wrote {len(bundles)} patient bundles ({'messy' if args.messy else 'clean'}) to {args.out}"
    )


if __name__ == "__main__":
    main()
