"""Synthetic free-text clinical notes with ground-truth PHI labels (Stage 2 input).

Reads the committed FHIR landing layer and writes one note per patient, grounded
in that patient's real structured facts (conditions/meds/vitals) so the note
narrates a real history rather than inventing facts. Each note embeds the
patient's PHI (name, MRN, DOB, address, provider name, dates) in realistic
chart prose.

Crucially, as each PHI value is inserted the exact character span is recorded —
so free-text de-id recall can be MEASURED against ground truth (this is the
"labeled sample" HIPAA-style de-id QA needs), not asserted.

Deterministic: each note is built from a per-patient seed, so it neither depends
on iteration order nor perturbs Stage 0/1's committed structured landing.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from generation.landing import group_by_patient, read_landing

PROVIDER_FAMILY = [
    "Reyes",
    "Nakamura",
    "Okafor",
    "Bergstrom",
    "Ali",
    "Castellano",
    "Novak",
    "Petrov",
]

CC_TEMPLATES = [
    "Follow-up visit for {condition} management.",
    "Presents for routine follow-up of {condition}.",
    "Here today to discuss {condition}.",
]
HPI_TEMPLATES = [
    "Patient reports doing well on current regimen for {condition}, diagnosed {onset}.",
    "{condition} (dx {onset}) remains stable; no new complaints today.",
    "Continues management of {condition}, first diagnosed {onset}.",
]
PLAN_TEMPLATES = [
    "Continue current management of {condition}.",
    "Continue present treatment for {condition}; recheck at next visit.",
]


class NoteBuilder:
    """Accumulates note text while recording char spans of inserted PHI."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._pos = 0
        self.spans: list[dict] = []

    def add(self, text: str) -> None:
        self._parts.append(text)
        self._pos += len(text)

    def phi(self, text: str, phi_type: str) -> None:
        self.spans.append(
            {"start": self._pos, "end": self._pos + len(text), "type": phi_type, "text": text}
        )
        self.add(text)

    def line(self, text: str = "") -> None:
        self.add(text + "\n")

    @property
    def text(self) -> str:
        return "".join(self._parts).strip()


def _name(patient: dict) -> tuple[str, str]:
    n = (patient.get("name") or [{}])[0]
    return " ".join(n.get("given", [])), n.get("family", "")


def _address(patient: dict) -> dict | None:
    return (patient.get("address") or [None])[0]


def _cond_display(cond: dict) -> str:
    return cond.get("code", {}).get("text") or cond["code"]["coding"][0].get("display", "condition")


def _cond_onset(cond: dict) -> str | None:
    return cond.get("onsetDateTime")


def _med_label(med: dict) -> str:
    return med.get("medicationCodeableConcept", {}).get("text", "medication")


def _bp_values(obs: dict) -> tuple[float, float] | None:
    comps = {
        c["code"]["coding"][0]["code"]: c["valueQuantity"]["value"]
        for c in obs.get("component", [])
    }
    if "8480-6" in comps and "8462-4" in comps:
        return comps["8480-6"], comps["8462-4"]
    return None


def build_note(record: dict, rng: random.Random) -> dict:
    patient = record["patient"]
    conditions = record["conditions"]
    given, family = _name(patient)
    addr = _address(patient)

    # Encounter date: shortly after the latest known event date.
    event_dates = [c["onsetDateTime"] for c in conditions if c.get("onsetDateTime")]
    event_dates += [
        o["effectiveDateTime"] for o in record["observations"] if o.get("effectiveDateTime")
    ]
    latest = max((date.fromisoformat(d) for d in event_dates), default=date(2026, 5, 1))
    visit_date = (latest + timedelta(days=rng.randint(1, 60))).isoformat()
    provider = f"Dr. {rng.choice(PROVIDER_FAMILY)}"

    b = NoteBuilder()

    # Header — dense PHI.
    b.add("Patient: ")
    if given or family:
        b.phi(f"{given} {family}".strip(), "PATIENT_NAME")
    if patient.get("birthDate"):
        b.add(", DOB ")
        b.phi(patient["birthDate"], "DOB")  # shifted by the DOB offset (separate from visit dates)
    if mrn := (patient.get("identifier") or [{}])[0].get("value"):
        b.add(", MRN ")
        b.phi(mrn, "MRN")
    b.line()
    if addr:
        b.add("Address: ")
        b.phi(", ".join(addr.get("line", [])), "ADDRESS")
        b.add(", ")
        b.phi(addr.get("city", ""), "CITY")
        b.add(", ")
        b.phi(addr.get("state", ""), "STATE")
        b.add(" ")
        b.phi(addr.get("postalCode", ""), "ZIP")
        b.line()
    b.add("Visit Date: ")
    b.phi(visit_date, "DATE")
    b.add("  Seen by ")
    b.phi(provider, "PROVIDER_NAME")
    b.line()
    b.line()

    primary = conditions[0] if conditions else None
    if primary:
        b.line(
            "Chief Complaint: " + rng.choice(CC_TEMPLATES).format(condition=_cond_display(primary))
        )
        b.line()

    for cond in conditions:
        onset = _cond_onset(cond)
        b.add("HPI: ")
        # Split the HPI template around the {onset} date so the date span is exact.
        template = rng.choice(HPI_TEMPLATES).format(condition=_cond_display(cond), onset="\x00")
        pre, _, post = template.partition("\x00")
        b.add(pre)
        if onset:
            b.phi(onset, "DATE")
        b.line(post)
    if conditions:
        b.line()

    if record["medications"]:
        # One entry per medication, most recent prescription wins (mirrors the
        # vitals dedup below). The same drug recurs across encounters, and
        # listing it twice — often once with a dosage and once without — reads
        # as one visit contradicting itself, which the extractor flags as
        # low-confidence. Undated prescriptions rank oldest.
        latest_med: dict[str, tuple[str, str]] = {}
        for m in record["medications"]:
            label = _med_label(m)
            dose = (m.get("dosageInstruction") or [{}])[0].get("text", "n/a")
            when = m.get("authoredOn") or ""
            if label not in latest_med or when >= latest_med[label][0]:
                latest_med[label] = (when, f"{label} - {dose}")  # first use fixes order
        meds = "; ".join(text for _, text in latest_med.values())
        b.line(f"Current Medications: {meds}.")
        b.line()

    # One reading per vital. A patient has multiple encounters' worth of
    # Observations for the same measurement (often in different units); dumping
    # them all into a single Vitals line reads as one visit reporting
    # contradictory values ("Body weight 91.7 kg, Body weight 145.5 [lb_av]"),
    # which the downstream extractor rightly flags as low-confidence. Keep only
    # the most recent reading per vital (undated readings rank oldest).
    latest: dict[str, tuple[str, str]] = {}
    for obs in record["observations"]:
        bp = _bp_values(obs)
        if bp:
            key, text = "BP", f"BP {bp[0]:.0f}/{bp[1]:.0f} mmHg"
        elif "valueQuantity" in obs:
            key = obs.get("code", {}).get("text", "value")
            q = obs["valueQuantity"]
            text = f"{key} {q['value']} {q.get('unit', '')}".strip()
        else:
            continue
        when = obs.get("effectiveDateTime") or ""
        if key not in latest or when >= latest[key][0]:
            latest[key] = (when, text)  # first appearance fixes order; latest date wins
    vitals = [text for _, text in latest.values()]
    if vitals:
        b.line(f"Vitals: {', '.join(vitals)}.")
        b.line()

    if primary:
        b.line(
            "Assessment/Plan: "
            + rng.choice(PLAN_TEMPLATES).format(condition=_cond_display(primary))
        )

    return {
        "patient_id": patient["id"],
        "note_id": f"note-{patient['id']}",
        "visit_date": visit_date,
        "text": b.text,
        "phi_spans": b.spans,
    }


def generate_notes(resources: list[dict], seed: int = 42) -> list[dict]:
    records = group_by_patient(resources)
    notes = []
    for pid, rec in records.items():
        rng = random.Random(f"{seed}-{pid}")  # per-patient seed -> order-independent
        notes.append(build_note(rec, rng))
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic clinical notes with PHI labels."
    )
    parser.add_argument("--landing", type=Path, default=Path("data/landing"))
    parser.add_argument("--out", type=Path, default=Path("data/notes/raw_notes.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("data/notes/phi_labels.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    notes = generate_notes(read_landing(args.landing), seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f, args.labels.open("w") as lf:
        for note in notes:
            f.write(
                json.dumps({k: note[k] for k in ("patient_id", "note_id", "visit_date", "text")})
                + "\n"
            )
            lf.write(
                json.dumps({"note_id": note["note_id"], "phi_spans": note["phi_spans"]}) + "\n"
            )
    total_phi = sum(len(n["phi_spans"]) for n in notes)
    print(f"Wrote {len(notes)} notes ({total_phi} labeled PHI spans) to {args.out}")


if __name__ == "__main__":
    main()
