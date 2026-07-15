"""Stage 4 — normalization of Stage 3's raw extraction output.

Takes `data/extracted/extractions.jsonl` (the "LLM dump": note wording preserved
on purpose, per `extraction/extractor.py`) and produces a consistently-formatted
version: canonical diagnosis names, canonical vital names/units (values
converted, not just relabeled), and canonical medication dosage text.

**Grounded in the generator's own source of truth, not free-form NLP.** The
closed vocabulary this synthetic dataset can ever contain is defined once in the
pinned terminology snapshot (`terminology/`) and the strength/dosage table in
`generation/generate_fhir.py` (`MED_CONTENT`, `SIMPLE_OBS`). Extraction pulls
from that same closed vocabulary, so normalization here is a lookup against those
known canonical forms plus the handful of paraphrase/contamination variants Haiku
actually produced at the 100-record scale — not a general medical-abbreviation
parser. A real EHR feed would need a live terminology service (the documented
upgrade). Values that don't match a known form are left as-is and *counted*, not
silently dropped or guessed at.

The date contamination this fixes (e.g. `"Migraine, diagnosed 2017-07-17"`) comes
from `generate_notes.py`'s HPI templates ("... diagnosed {onset}") bleeding the
onset date into the captured diagnosis name; the name field should be just the
name, so the trailing date clause is stripped here.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import terminology as t
from generation.generate_fhir import MED_CONTENT, SIMPLE_OBS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXTRACTED_PATH = DATA_DIR / "extracted" / "extractions.jsonl"
NORMALIZED_PATH = DATA_DIR / "curated" / "normalized.jsonl"
METRICS_PATH = DATA_DIR / "curated" / "normalize_metrics.json"

MAX_UNMATCHED_EXAMPLES = 50

# --- Diagnoses -------------------------------------------------------------

CANONICAL_CONDITIONS = [
    t.get_condition(icd).standard_name or t.get_condition(icd).source_name
    for icd in t.list_conditions()
]
_CONDITIONS_BY_LENGTH = sorted(CANONICAL_CONDITIONS, key=len, reverse=True)

# Paraphrases Haiku produced that aren't a canonical prefix (so the startswith
# check below misses them): it dropped a qualifier that sits *before* the head
# noun ("Uncomplicated asthma" -> "asthma") or *after* it ("... without
# complication").
_DIAGNOSIS_PARAPHRASES = {
    "type 2 diabetes mellitus": "Type 2 diabetes mellitus without complication",
    "asthma": "Uncomplicated asthma",
}

# generate_notes.py renders history as "{condition}, diagnosed {onset}",
# "{condition}, first diagnosed {onset}", or "{condition} (dx {onset})";
# extraction sometimes captures that trailing date clause inside the name.
_TRAILING_DX_DATE_RE = re.compile(
    r"\s*(?:,\s*(?:first\s+)?diagnosed\s+\d{4}-\d{2}-\d{2}|\(dx\s+[^)]*\))\s*$",
    re.IGNORECASE,
)


def normalize_diagnosis_name(raw_name: str) -> tuple[str, bool]:
    """Return (canonical_name, matched)."""
    stripped = _TRAILING_DX_DATE_RE.sub("", raw_name).strip()

    paraphrase = _DIAGNOSIS_PARAPHRASES.get(stripped.lower())
    if paraphrase:
        return paraphrase, True

    for canonical in _CONDITIONS_BY_LENGTH:
        if stripped.lower() == canonical.lower():
            return canonical, True
        # "<canonical> management" bled in from a "Continue management of ..." line.
        if stripped.lower().startswith(canonical.lower()):
            return canonical, True

    return stripped, False


# --- Vitals ----------------------------------------------------------------

# Canonical LOINC display -> its generate_fhir SIMPLE_OBS spec (unit/alt_unit).
_OBS_SPEC = {t.get_observation(loinc).source_name: spec for loinc, spec in SIMPLE_OBS.items()}

_VITAL_ALIASES = {name.lower(): name for name in _OBS_SPEC}
_VITAL_ALIASES.update({"glucose": "Glucose [Mass/volume] in Blood", "hr": "Heart rate"})

# Inverse of generate_fhir's forward converters (canonical -> alt), since messy
# *input* may already be in the alt unit.
_INVERSE = {
    "Body temperature": lambda f: round((f - 32) * 5 / 9, 1),  # [degF] -> Cel
    "Body weight": lambda lb: round(lb / 2.20462, 1),  # [lb_av] -> kg
    "Body height": lambda inch: round(inch * 2.54, 1),  # [in_i] -> cm
}
_BP_NAMES = {"bp", "blood pressure", "blood pressure panel"}
_BP_VALUE_RE = re.compile(r"^\d{2,3}/\d{2,3}$")


# Bare unit spellings Haiku uses for the UCUM alt units ([lb_av]/[in_i]/[degF]).
_UNIT_ALIASES = {
    "in": "in_i",
    "inch": "in_i",
    "inches": "in_i",
    "lb": "lb_av",
    "lbs": "lb_av",
    "pound": "lb_av",
    "pounds": "lb_av",
    "f": "degf",
    "degf": "degf",
    "c": "cel",
    "celsius": "cel",
}


def _canon_unit(u: str) -> str:
    # Drop the degree sign Haiku emits ("°F"/"°C") so the bare-spelling aliases
    # below match; strip UCUM brackets too ([degF] -> degf).
    token = u.strip().strip("[]").replace("°", "").strip().lower()
    return _UNIT_ALIASES.get(token, token)


def _unit_matches(unit: str | None, canonical: str) -> bool:
    """UCUM comparison tolerant of the brackets extraction strips ([lb_av] vs
    lb_av) and of bare unit spellings ('in' for [in_i])."""
    if unit is None:
        return False
    return _canon_unit(unit) == canonical.strip("[]").lower()


def normalize_vital(
    raw_name: str, raw_value, unit: str | None
) -> tuple[str, object, str | None, bool]:
    """Return (canonical_name, canonical_value, canonical_unit, matched).

    Blood pressure stays a compound "sys/dia" string (canonical unit mm[Hg]);
    single-value vitals are coerced to float and converted to canonical units.
    """
    name = raw_name.strip().lower()
    if name in _BP_NAMES and isinstance(raw_value, str) and _BP_VALUE_RE.match(raw_value.strip()):
        return "Blood pressure", raw_value.strip(), "mm[Hg]", True

    canonical_name = _VITAL_ALIASES.get(name)
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return raw_name, raw_value, unit, False
    if canonical_name is None:
        return raw_name, value, unit, False

    spec = _OBS_SPEC[canonical_name]
    canonical_unit, alt_unit = spec["unit"], spec["alt_unit"]
    if unit is None or _unit_matches(unit, canonical_unit):
        return canonical_name, value, canonical_unit, True
    if alt_unit is not None and _unit_matches(unit, alt_unit):
        return canonical_name, _INVERSE[canonical_name](value), canonical_unit, True
    return canonical_name, value, unit, False  # unknown unit: keep, flag


# --- Medications -----------------------------------------------------------


def _dosage_variants(content: dict) -> dict[str, str]:
    """Known dosage strings for a med -> its canonical full-text dosage. Includes
    the short shorthand, the full text, and the full text minus a leading "Take "
    (Haiku routinely drops the imperative verb) — all deterministic closed-vocab
    forms, not fuzzy matches."""
    full, short = content["full"], content["short"]
    variants = {short: full, full: full}
    if full.lower().startswith("take "):
        variants[full[len("take ") :]] = full
    return variants


# {canonical label: {known dosage variant: canonical full-text dosage}}.
_DOSAGE_LOOKUP = {c["label"]: _dosage_variants(c) for c in MED_CONTENT.values()}
# Bare drug name ("Amlodipine") -> full canonical label; base names are unique.
_MED_BASE_TO_LABEL = {c["label"].split(" ", 1)[0].lower(): c["label"] for c in MED_CONTENT.values()}
_MISSING_DOSAGE = {"", "dosage not recorded", "none", "n/a"}


def resplit_bare_medication_name(name: str, dosage_text: str | None) -> tuple[str, str | None]:
    """Upgrade a bare drug name to its full canonical label, stripping any
    bundled "{strength} {form}" prefix (however it's separated — " - ", ", ", …)
    back out of dosage_text. No-op if `name` isn't a recognized bare drug name."""
    label = _MED_BASE_TO_LABEL.get(name.strip().lower())
    if label is None or label.lower() == name.strip().lower():
        return name, dosage_text
    prefix = label[len(name.strip()) :].strip()  # e.g. "5 MG Oral Tablet"
    if dosage_text is not None and dosage_text.lower().startswith(prefix.lower()):
        rest = dosage_text[len(prefix) :].lstrip(" -,;")
        return label, rest or None
    return label, dosage_text


def normalize_dosage(med_name: str, dosage_text: str | None) -> tuple[str | None, bool]:
    """Return (canonical_dosage_or_None, matched)."""
    if dosage_text is None or dosage_text.strip().lower() in _MISSING_DOSAGE:
        return None, True
    variants = _DOSAGE_LOOKUP.get(med_name)
    if variants and dosage_text in variants:
        return variants[dosage_text], True
    return dosage_text, False


# --- Record-level ----------------------------------------------------------


def normalize_record(record: dict) -> tuple[dict, list[dict]]:
    """Normalize one extraction record. Returns (normalized_record, unmatched)
    where each unmatched item is {"field_type", "detail"}. Canonicalization
    surfaces re-worded mentions of the same diagnosis/medication as exact
    duplicates, so they're de-duplicated here."""
    unmatched: list[dict] = []

    diagnoses, seen_dx = [], set()
    for dx in record["diagnoses"]:
        name, matched = normalize_diagnosis_name(dx["name"])
        if not matched:
            unmatched.append({"field_type": "diagnosis", "detail": f"diagnosis: {dx['name']!r}"})
        if name not in seen_dx:
            seen_dx.add(name)
            diagnoses.append({"name": name})

    medications, seen_med = [], set()
    for med in record["medications"]:
        name, dosage_text = resplit_bare_medication_name(med["name"], med.get("dosage"))
        dosage, matched = normalize_dosage(name, dosage_text)
        if not matched:
            unmatched.append(
                {"field_type": "dosage", "detail": f"dosage for {name!r}: {dosage_text!r}"}
            )
        key = (name, dosage)
        if key not in seen_med:
            seen_med.add(key)
            medications.append({"name": name, "dosage": dosage})

    vitals = []
    for v in record["vitals"]:
        name, value, unit, matched = normalize_vital(v["name"], v["value"], v.get("unit"))
        vitals.append({"name": name, "value": value, "unit": unit})
        if not matched:
            unmatched.append(
                {
                    "field_type": "vital",
                    "detail": f"vital: {v['name']!r} ({v['value']!r} {v.get('unit')!r})",
                }
            )

    normalized = {**record, "diagnoses": diagnoses, "medications": medications, "vitals": vitals}
    return normalized, unmatched


def read_extractions(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def compute_metrics(records: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Normalize every record; return (normalized, all_unmatched, metrics). Pure
    (no I/O) so it's unit-testable against a hand-built record list."""
    normalized_records, all_unmatched = [], []
    totals = {"diagnosis": 0, "dosage": 0, "vital": 0}
    for record in records:
        normalized, unmatched = normalize_record(record)
        normalized_records.append(normalized)
        all_unmatched.extend(unmatched)
        totals["diagnosis"] += len(record["diagnoses"])
        totals["dosage"] += len(record["medications"])
        totals["vital"] += len(record["vitals"])

    unmatched_counts = {"diagnosis": 0, "dosage": 0, "vital": 0}
    for item in all_unmatched:
        unmatched_counts[item["field_type"]] += 1

    by_field = {
        ft: {
            "total": totals[ft],
            "matched": totals[ft] - unmatched_counts[ft],
            "unmatched": unmatched_counts[ft],
        }
        for ft in ("diagnosis", "dosage", "vital")
    }
    overall_total = sum(totals.values())
    overall_unmatched = len(all_unmatched)
    metrics = {
        "total_records": len(records),
        "overall": {
            "total": overall_total,
            "matched": overall_total - overall_unmatched,
            "unmatched": overall_unmatched,
            "unmatched_rate": (overall_unmatched / overall_total) if overall_total else 0.0,
        },
        "by_field": by_field,
        "unmatched_examples": [i["detail"] for i in all_unmatched[:MAX_UNMATCHED_EXAMPLES]],
    }
    return normalized_records, all_unmatched, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 normalize (extraction -> canonical).")
    parser.add_argument("--in", dest="in_path", type=Path, default=EXTRACTED_PATH)
    parser.add_argument("--out", type=Path, default=NORMALIZED_PATH)
    parser.add_argument("--metrics-out", type=Path, default=METRICS_PATH)
    args = parser.parse_args()

    records = list(read_extractions(args.in_path))
    normalized_records, all_unmatched, metrics = compute_metrics(records)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for normalized in normalized_records:
            f.write(json.dumps(normalized) + "\n")
    args.metrics_out.write_text(json.dumps(metrics, indent=2))

    print(f"Wrote {metrics['total_records']} normalized records to {args.out}")
    print(
        f"Matched {metrics['overall']['matched']}/{metrics['overall']['total']} values "
        f"({metrics['overall']['unmatched']} unmatched)"
    )


if __name__ == "__main__":
    main()
