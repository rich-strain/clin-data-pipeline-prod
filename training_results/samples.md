# Before/after generation samples

## Sample 1

**Instruction:**
```
Patient: [PATIENT_NAME], DOB 1972-03-06, MRN [MRN]
Visit Date: 2025-09-13  Seen by [PROVIDER_NAME]

Chief Complaint: Here today to discuss Hypothyroidism.

HPI: Hypothyroidism (dx 2025-05-07) remains stable; no new complaints today.
HPI: Uncomplicated asthma (dx 2020-07-08) remains stable; no new complaints today.

Current Medications: Levothyroxine 75 MCG Oral Tablet - Take 75 mcg by mouth every morning; Albuterol 90 MCG Inhaler - Inhale 2 puffs every 4 hours as needed.

Vitals: BP 101/64 mmHg, Heart rate 80.3 /min.

Assessment/Plan: Continue present treatment for Hypothyroidism; recheck at next visit.
```

**Ground truth:**
```json
{"diagnoses": [{"name": "Hypothyroidism"}, {"name": "Uncomplicated asthma"}], "medications": [{"name": "Levothyroxine 75 MCG Oral Tablet", "dosage": "Take 75 mcg by mouth every morning"}, {"name": "Albuterol 90 MCG Inhaler", "dosage": "Inhale 2 puffs every 4 hours as needed"}], "vitals": [{"name": "Blood pressure", "value": "101/64", "unit": "mm[Hg]"}, {"name": "Heart rate", "value": 80.3, "unit": "/min"}]}
```

**Base (no adapter):**
```
```json
{
  "diagnoses": ["Hypothyroidism", "Uncomplicated asthma"],
  "medications": {
    "Levothyroxine": "Take 75 mcg by mouth every morning",
    "Albuterol": "Inhale 2 puffs every 4 hours as needed"
  },
  "vitals": {
    "BP": "101/64 mmHg",
    "Heart rate": "80.3/min"
  }
}
```
```

**Fine-tuned:**
```
{"diagnoses": [{"name": "Hypothyroidism"}, {"name": "Uncomplicated asthma"}], "medications": [{"name": "Levothyroxine 75 MCG Oral Tablet", "dosage": "Take 75 mcg by mouth every morning"}, {"name": "Albuterol 90 MCG Inhaler", "dosage": "Inhale 2 puffs every 4 hours as needed"}], "vitals": [{"name": "Blood pressure", "value": "101/64", "unit": "mm[Hg]"}, {"name": "Heart rate", "value": 80.3, "unit": "/min"}]}
```

## Sample 2

**Instruction:**
```
Patient: [PATIENT_NAME], DOB 1961-08-24, MRN [MRN]
Address: [ADDRESS], [CITY], [STATE] [ZIP]
Visit Date: 2025-09-26  Seen by [PROVIDER_NAME]

Chief Complaint: Presents for routine follow-up of Obesity.

HPI: Continues management of Obesity, first diagnosed 2021-02-17.
HPI: Patient reports doing well on current regimen for Gastroesophageal reflux disease without esophagitis, diagnosed 2019-05-18.
HPI: Obesity (dx 2022-12-01) remains stable; no new complaints today.

Current Medications: Phentermine 37.5 MG Oral Tablet - 37.5mg PO QAM; Omeprazole 20 MG Oral Capsule - 20mg PO QD.

Vitals: BP 131/89 mmHg, Glucose [Mass/volume] in Blood 87.3 mg/dL, Body weight 136.0 [lb_av].

Assessment/Plan: Continue present treatment for Obesity; recheck at next visit.
```

**Ground truth:**
```json
{"diagnoses": [{"name": "Obesity"}, {"name": "Gastroesophageal reflux disease without esophagitis"}], "medications": [{"name": "Phentermine 37.5 MG Oral Tablet", "dosage": "Take 37.5 mg by mouth every morning before breakfast"}, {"name": "Omeprazole 20 MG Oral Capsule", "dosage": "Take 20 mg by mouth once daily"}], "vitals": [{"name": "Blood pressure", "value": "131/89", "unit": "mm[Hg]"}, {"name": "Glucose [Mass/volume] in Blood", "value": 87.3, "unit": "mg/dL"}, {"name": "Body weight", "value": 61.7, "unit": "kg"}]}
```

**Base (no adapter):**
```
```json
{
  "diagnoses": [
    {
      "code": "Obesity",
      "description": "Diagnosis"
    }
  ],
  "medications": [
    {
      "code": "Phentermine 37.5 MG Oral Tablet",
      "description": "Medication"
    },
    {
      "code": "Omeprazole 20 MG Oral Capsule",
      "description": "Medication"
    }
  ],
  "vitals": [
    {
      "code": "BP",
      "value": "131/89 mmHg",
      "unit": "mmHg"
    },
    {
      "code": "Glucose",
      "value": "87.3 mg/dL",
      "unit": "%"
    },
    {
      "code": "Body Weight",
      "value": "136.0 lb",
      "unit": "lb"
    }
  ]
}
```
```

**Fine-tuned:**
```
{"diagnoses": [{"name": "Obesity"}, {"name": "Gastroesophageal reflux disease without esophagitis"}], "medications": [{"name": "Phentermine 37.5 MG Oral Tablet", "dosage": "Take 37.5 mg by mouth every morning before breakfast"}, {"name": "Omeprazole 20 MG Oral Capsule", "dosage": "Take 20 mg by mouth once daily"}], "vitals": [{"name": "Blood pressure", "value": "131/89", "unit": "mm[Hg]"}, {"name": "Glucose [Mass/volume] in Blood", "value": 87.3, "unit": "mg/dL"}, {"name": "Body weight", "value": 61.6, "unit": "kg"}]}
```

## Sample 3

**Instruction:**
```
Patient: [PATIENT_NAME], DOB 1945-06-10, MRN [MRN]
Address: [ADDRESS], [CITY], [STATE] [ZIP]
Visit Date: 2025-05-15  Seen by [PROVIDER_NAME]

Chief Complaint: Follow-up visit for Uncomplicated asthma management.

HPI: Continues management of Uncomplicated asthma, first diagnosed 2020-04-10.
HPI: Migraine (dx 2019-02-23) remains stable; no new complaints today.
HPI: Continues management of Essential hypertension, first diagnosed 2025-04-11.

Current Medications: Albuterol 90 MCG Inhaler - Inhale 2 puffs every 4 hours as needed; Sumatriptan 50 MG Oral Tablet - 50mg PO PRN; Lisinopril 10 MG Oral Tablet - Take 10 mg by mouth once daily.

Vitals: BP 125/81 mmHg, Body height 72.2 [in_i].

Assessment/Plan: Continue present treatment for Uncomplicated asthma; recheck at next visit.
```

**Ground truth:**
```json
{"diagnoses": [{"name": "Uncomplicated asthma"}, {"name": "Migraine"}, {"name": "Essential hypertension"}], "medications": [{"name": "Albuterol 90 MCG Inhaler", "dosage": "Inhale 2 puffs every 4 hours as needed"}, {"name": "Sumatriptan 50 MG Oral Tablet", "dosage": "Take 50 mg by mouth at onset of migraine; may repeat once after 2 hours"}, {"name": "Lisinopril 10 MG Oral Tablet", "dosage": "Take 10 mg by mouth once daily"}], "vitals": [{"name": "Blood pressure", "value": "125/81", "unit": "mm[Hg]"}, {"name": "Body height", "value": 183.4, "unit": "cm"}]}
```

**Base (no adapter):**
```
```json
{
  "diagnoses": [
    {
      "code": "Uncomplicated Asthma",
      "description": "Follow-up visit for Uncomplicated asthma management."
    }
  ],
  "medications": [
    {
      "code": "Albuterol 90 MCG Inhaler",
      "description": "Inhale 2 puffs every 4 hours as needed"
    },
    {
      "code": "Sumatriptan 50 MG Oral Tablet",
      "description": "50mg PO PRN"
    },
    {
      "code": "Lisinopril 10 MG Oral Tablet",
      "description": "Take 10 mg by mouth once daily"
    }
  ],
  "vitals": {
    "BP": "125/81 mmHg",
    "Body Height": "72.2 in"
  }
}
```
```

**Fine-tuned:**
```
{"diagnoses": [{"name": "Uncomplicated asthma"}, {"name": "Migraine"}, {"name": "Essential hypertension"}], "medications": [{"name": "Albuterol 90 MCG Inhaler", "dosage": "Inhale 2 puffs every 4 hours as needed"}, {"name": "Sumatriptan 50 MG Oral Tablet", "dosage": "Take 50 mg by mouth at onset of migraine; may repeat once after 2 hours"}, {"name": "Lisinopril 10 MG Oral Tablet", "dosage": "Take 10 mg by mouth once daily"}], "vitals": [{"name": "Blood pressure", "value": "125/81", "unit": "mm[Hg]"}, {"name": "Body height", "value": 183.3, "unit": "cm"}]}
```
