"""FHIR R4 -> OMOP CDM v5.4 ETL (the analytics model for this repo).

Derives OMOP Common Data Model tables from the FHIR landing layer. Standard
concept_ids come from the verified pinned snapshot (terminology/); nothing is
fabricated. Unmapped source codes carry concept_id 0 (OMOP's own sentinel),
consistent with a real ETL where the vocabulary lacks a mapping.
"""

from omop.etl import CDM_TABLES, fhir_to_omop

__all__ = ["CDM_TABLES", "fhir_to_omop"]
