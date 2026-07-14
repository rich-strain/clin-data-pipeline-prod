"""One-off builder for the diagnosis -> medication lookup (terminology/dxmed_lookup.json).

Seeds clinically plausible dx->med pairings from the NLM RxClass API's `may_treat`
(MED-RT) relationships, scoped to THIS repo's closed vocabulary only: for each
generated ICD-10 diagnosis it resolves a RxClass DISEASE class, pulls the
may_treat ingredient members, and intersects them with the RxNorm medications in
terminology/omop_concept_snapshot.json. Pairings are therefore RxClass-derived,
not hand-authored.

This mirrors the OMOP snapshot's discipline: a one-off pull against a public API
whose OUTPUT is a committed static asset (never a live call at generation time).
Rerun to refresh:  python -m terminology.build_dxmed_lookup

Prevalence weights are a rough MANUAL seed (RxClass exposes no prevalence) and
any diagnosis whose only in-vocabulary option is a newly-added medication is
flagged in `_flags` for human review before the table is trusted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

import terminology as t

RXCLASS = "https://rxnav.nlm.nih.gov/REST/rxclass"
OUT_PATH = Path(__file__).resolve().parent / "dxmed_lookup.json"
RESOLVED_DATE = "2026-07-14"

# Our ICD-10 display names don't match RxClass DISEASE class names 1:1 (ICD-10 ->
# MeSH/MED-RT isn't automatic), so the search term per diagnosis is curated. The
# class *id* is still resolved live via byName below — not hardcoded.
CLASS_SEARCH_TERM = {
    "E11.9": "Type 2 Diabetes Mellitus",
    "I10": "Hypertension",
    "J45.909": "Asthma",
    "E78.5": "Hyperlipidemia",
    "J44.9": "Pulmonary Disease, Chronic Obstructive",
    "G43.909": "Migraine Disorders",
    "E66.9": "Obesity",
    "K21.9": "Gastroesophageal Reflux",
    "E03.9": "Hypothyroidism",
    "F32.9": "Depressive Disorder",
}

# Rough manual prevalence weights where a diagnosis has >1 in-vocab option
# (RxClass gives no prevalence). Everything else defaults to an even split.
WEIGHT_OVERRIDES: dict[str, dict[str, float]] = {
    "I10": {"29046": 0.55, "17767": 0.45},  # lisinopril (ACE) vs amlodipine (CCB)
}

# Medications added to the vocabulary specifically to cover a diagnosis that had
# no in-vocab may_treat option; surfaced in _flags for clinical review.
NEWLY_ADDED = {"69120", "37418", "8152"}  # tiotropium, sumatriptan, phentermine


def _get(url: str) -> dict:
    r = httpx.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def _disease_class(term: str) -> dict | None:
    q = term.replace(" ", "%20").replace(",", "%2C")
    data = _get(f"{RXCLASS}/class/byName.json?className={q}&classTypes=DISEASE")
    infos = data.get("rxclassMinConceptList", {}).get("rxclassMinConcept", [])
    return {"classId": infos[0]["classId"], "className": infos[0]["className"]} if infos else None


def _may_treat_rxcuis(class_id: str) -> set[str]:
    data = _get(f"{RXCLASS}/classMembers.json?classId={class_id}&relaSource=MEDRT&rela=may_treat")
    members = data.get("drugMemberGroup", {}).get("drugMember", [])
    return {m["minConcept"]["rxcui"] for m in members}


def build() -> dict:
    our_meds = {rx: t.get_medication(rx).source_name for rx in t.list_medications()}
    conditions: dict[str, dict] = {}
    flags: list[dict] = []

    for icd10 in t.list_conditions():
        cond = t.get_condition(icd10)
        term = CLASS_SEARCH_TERM.get(icd10)
        if not term:
            flags.append({"icd10": icd10, "reason": "no curated RxClass search term"})
            continue
        klass = _disease_class(term)
        if not klass:
            flags.append({"icd10": icd10, "reason": f"no RxClass DISEASE class for '{term}'"})
            continue

        may_treat = _may_treat_rxcuis(klass["classId"])
        matched = [rx for rx in our_meds if rx in may_treat]
        weights = WEIGHT_OVERRIDES.get(icd10, {})
        meds = [
            {
                "rxcui": rx,
                "display": our_meds[rx],
                "weight": round(weights.get(rx, 1.0 / len(matched)), 3),
            }
            for rx in matched
        ]
        conditions[icd10] = {
            "display": cond.standard_name or cond.source_name,
            "rxclass": klass,
            "medications": meds,
        }

        if not matched:
            flags.append({"icd10": icd10, "reason": "no in-vocabulary may_treat medication"})
        elif all(rx in NEWLY_ADDED for rx in matched):
            flags.append(
                {
                    "icd10": icd10,
                    "reason": f"only option is newly-added med {matched}; review fit + dosage",
                }
            )
        time.sleep(0.3)

    # Obesity's RxClass list is dominated by withdrawn agents (sibutramine,
    # lorcaserin) — call that out explicitly even though phentermine matched.
    if "E66.9" in conditions:
        flags.append(
            {
                "icd10": "E66.9",
                "reason": "RxClass may_treat for obesity is heavy with withdrawn drugs; "
                "phentermine chosen as a still-approved classic agent — verify.",
            }
        )

    return {
        "_provenance": {
            "source": "NLM RxClass API (may_treat, MED-RT)",
            "api": RXCLASS,
            "relationship": "may_treat",
            "rela_source": "MEDRT",
            "resolved_date": RESOLVED_DATE,
            "method": (
                "Per generated ICD-10 diagnosis: resolve a RxClass DISEASE class "
                "(curated search term), pull may_treat ingredient members, intersect "
                "with this repo's closed RxNorm vocabulary (omop_concept_snapshot.json). "
                "Pairings are RxClass-derived, not hand-authored."
            ),
            "class_search_terms": CLASS_SEARCH_TERM,
            "weights_note": (
                "prevalence weights are a rough MANUAL seed (RxClass exposes no "
                "prevalence); single-option diagnoses default to 1.0."
            ),
        },
        "conditions": conditions,
        "_flags": flags,
    }


def main() -> None:
    table = build()
    OUT_PATH.write_text(json.dumps(table, indent=2) + "\n")
    n_flags = len(table["_flags"])
    print(f"Wrote {len(table['conditions'])} diagnoses to {OUT_PATH} ({n_flags} flags)")
    for f in table["_flags"]:
        print(f"  FLAG {f['icd10']}: {f['reason']}")


if __name__ == "__main__":
    main()
