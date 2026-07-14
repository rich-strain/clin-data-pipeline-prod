"""Loader over the pinned OMOP concept snapshot (terminology/omop_concept_snapshot.json).

The snapshot maps each source code in our closed synthetic vocabulary to its
verified OMOP standard concept. This module is the one place that reads it, so
the generator, the OMOP ETL, and the binding report never drift on codes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SNAPSHOT_PATH = Path(__file__).resolve().parent / "omop_concept_snapshot.json"

# OMOP's sentinel for "no standard concept mapping available".
UNMAPPED = 0


@dataclass(frozen=True)
class Concept:
    """One coded value: its source coding plus the OMOP standard concept it maps to."""

    source_code: str
    source_name: str
    source_vocabulary: str
    source_concept_id: int
    standard_code: str
    standard_name: str
    standard_vocabulary: str
    standard_concept_id: int  # 0 == unmapped (honest)
    domain: str

    @property
    def is_mapped(self) -> bool:
        return self.standard_concept_id != UNMAPPED


@lru_cache(maxsize=1)
def _raw() -> dict:
    return json.loads(SNAPSHOT_PATH.read_text())


def _concept(source_code: str, entry: dict) -> Concept:
    src = entry.get("source") or {}
    std = entry.get("standard") or {}
    return Concept(
        source_code=source_code,
        source_name=src.get("concept_name", ""),
        source_vocabulary=src.get("vocabulary_id", ""),
        source_concept_id=src.get("concept_id", UNMAPPED),
        standard_code=std.get("concept_code", ""),
        standard_name=std.get("concept_name", ""),
        standard_vocabulary=std.get("vocabulary_id", ""),
        standard_concept_id=std.get("concept_id", UNMAPPED) or UNMAPPED,
        domain=(std.get("domain_id") or src.get("domain_id") or ""),
    )


def get_condition(icd10_code: str) -> Concept:
    return _concept(icd10_code, _raw()["conditions"][icd10_code])


def get_medication(rxnorm_code: str) -> Concept:
    return _concept(rxnorm_code, _raw()["medications"][rxnorm_code])


def get_observation(loinc_code: str) -> Concept:
    return _concept(loinc_code, _raw()["observations"][loinc_code])


def list_conditions() -> list[str]:
    return list(_raw()["conditions"].keys())


def list_medications() -> list[str]:
    return list(_raw()["medications"].keys())


def list_observations() -> list[str]:
    return list(_raw()["observations"].keys())


def unit_concept_id(ucum_code: str) -> int:
    """OMOP standard concept_id for a UCUM unit code (0 if unmapped/unknown)."""
    entry = _raw()["units"].get(ucum_code)
    if not entry or not entry.get("standard"):
        return UNMAPPED
    return entry["standard"].get("concept_id", UNMAPPED) or UNMAPPED


def gender_concept_id(gender: str) -> int:
    g = _raw()["metadata_concepts"]["gender"]
    return {"male": g["male"]["concept_id"], "female": g["female"]["concept_id"]}.get(
        gender, UNMAPPED
    )


def type_concept_ehr() -> int:
    return _raw()["metadata_concepts"]["type_concept"]["ehr"]["concept_id"]


def coverage() -> dict[str, dict[str, int]]:
    """Per-section standard-mapping coverage (mapped / total), read honestly off
    the snapshot — used by the binding report and the Stage 0/1 page."""
    raw = _raw()
    out: dict[str, dict[str, int]] = {}
    for section in ("conditions", "medications", "observations", "units"):
        entries = raw[section]
        mapped = sum(1 for v in entries.values() if (v.get("standard") or {}).get("concept_id"))
        out[section] = {"mapped": mapped, "total": len(entries)}
    return out
