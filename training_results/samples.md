# Before/after generation samples

## Sample 1

**Instruction:**
```
Patient: [PATIENT_NAME], DOB 1990-01-13, MRN [MRN]
Address: [ADDRESS], [CITY], [STATE] [ZIP]
Visit Date: 2025-10-30  Seen by [PROVIDER_NAME]

Chief Complaint: Here today to discuss Chronic obstructive lung disease.

HPI: Patient reports doing well on current regimen for Chronic obstructive lung disease, diagnosed 2020-02-19.
HPI: Uncomplicated asthma (dx 2018-12-06) remains stable; no new complaints today.

Current Medications: Tiotropium 18 MCG Inhalation Powder - Inhale the contents of 1 capsule once daily; Albuterol 90 MCG Inhaler - n/a.

Vitals: BP 116/72 mmHg, Body weight 229.9 [lb_av], Heart rate 98.8 /min, Body height 179.6 cm.

Assessment/Plan: Continue present treatment for Chronic obstructive lung disease; recheck at next visit.
```

**Ground truth:**
```json
{"diagnoses": [{"name": "Chronic obstructive lung disease"}, {"name": "Uncomplicated asthma"}], "medications": [{"name": "Tiotropium 18 MCG Inhalation Powder", "dosage": "Inhale the contents of 1 capsule once daily"}, {"name": "Albuterol 90 MCG Inhaler", "dosage": null}], "vitals": [{"name": "Blood pressure", "value": "116/72", "unit": "mm[Hg]"}, {"name": "Body weight", "value": 104.3, "unit": "kg"}, {"name": "Heart rate", "value": 98.8, "unit": "/min"}, {"name": "Body height", "value": 179.6, "unit": "cm"}]}
```

**Base (no adapter):**
```
```json
{
  "diagnoses": [
    {
      "code": "Chronic Obstructive Lung Disease",
      "description": "Chronic obstructive lung disease"
    }
  ],
  "medications": [
    {
      "name": "Tiotropium 18 MCG Inhalation Powder",
      "dosage": "Inhale the contents of 1 capsule once daily"
    },
    {
      "name": "Albuterol 90 MCG Inhaler",
      "dosage": "n/a"
    }
  ],
  "vitals": {
    "blood_pressure": "116/72 mmHg",
    "body_weight": "229.9 lbs",
    "heart_rate": "98.8/min",
    "body_height": "179.6 cm"
  }
}
```
```

**Fine-tuned:**
```
{"diagnoses": [{"name": "Chronic obstructive lung disease"}, {"name": "Uncomplicated asthma"}], "medications": [{"name": "Tiotropium 18 MCG Inhalation Powder", "dosage": "Inhale the contents of 1 capsule once daily"}, {"name": "Albuterol 90 MCG Inhaler", "dosage": null}], "vitals": [{"name": "Blood pressure", "value": "116/72", "unit": "mm[Hg]"}, {"name": "Body weight", "value": 94.9, "unit": "kg"}, {"name": "Heart rate", "value": 98.8, "unit": "/min"}, {"name": "Body height", "value": 179.6, "unit": "cm"}]}
```

## Sample 2

**Instruction:**
```
Patient: [PATIENT_NAME], DOB 1982-03-14, MRN [MRN]
Address: [ADDRESS], [CITY], [STATE] [ZIP]
Visit Date: 2024-10-10  Seen by [PROVIDER_NAME]

Chief Complaint: Here today to discuss Migraine.

HPI: Continues management of Migraine, first diagnosed 2022-04-01.

Current Medications: Sumatriptan 50 MG Oral Tablet - 50mg PO PRN.

Vitals: BP 112/84 mmHg, Heart rate 76.2 /min, Body height 185.6 cm, Body temperature 36.6 Cel.

Assessment/Plan: Continue current management of Migraine.
```

**Ground truth:**
```json
{"diagnoses": [{"name": "Migraine"}], "medications": [{"name": "Sumatriptan 50 MG Oral Tablet", "dosage": "Take 50 mg by mouth at onset of migraine; may repeat once after 2 hours"}], "vitals": [{"name": "Blood pressure", "value": "112/84", "unit": "mm[Hg]"}, {"name": "Heart rate", "value": 76.2, "unit": "/min"}, {"name": "Body height", "value": 185.6, "unit": "cm"}, {"name": "Body temperature", "value": 36.6, "unit": "Cel"}]}
```

**Base (no adapter):**
```
```json
{
  "diagnoses": [
    {
      "name": "Migraine",
      "code": "1001"
    }
  ],
  "medications": [
    {
      "name": "Sumatriptan",
      "code": "1002"
    }
  ],
  "vitals": {
    "blood_pressure": {
      "value": 112,
      "unit": "mmHg"
    },
    "heart_rate": {
      "value": 76.2,
      "unit": "min/min"
    },
    "body_height": {
      "value": 185.6,
      "unit": "cm"
    },
    "body_temperature": {
      "value": 36.6,
      "unit": "Cel"
    }
  }
}
```
```

**Fine-tuned:**
```
{"diagnoses": [{"name": "Migraine"}], "medications": [{"name": "Sumatriptan 50 MG Oral Tablet", "dosage": "Take 50 mg by mouth at onset of migraine; may repeat once after 2 hours"}], "vitals": [{"name": "Blood pressure", "value": "112/84", "unit": "mm[Hg]"}, {"name": "Heart rate", "value": 76.2, "unit": "/min"}, {"name": "Body height", "value": 185.6, "unit": "cm"}, {"name": "Body temperature", "value": 36.6, "unit": "Cel"}]}
```

## Sample 3

**Instruction:**
```
Patient: [PATIENT_NAME], DOB 2000-02-15, MRN [MRN]
Address: [ADDRESS], [CITY], [STATE] [ZIP]
Visit Date: 2025-12-27  Seen by [PROVIDER_NAME]

Chief Complaint: Presents for routine follow-up of Essential hypertension.

HPI: Essential hypertension (dx 2024-03-22) remains stable; no new complaints today.

Current Medications: Lisinopril 10 MG Oral Tablet - Take 10 mg by mouth once daily.

Vitals: BP 110/67 mmHg, Body temperature 37.6 Cel.

Assessment/Plan: Continue present treatment for Essential hypertension; recheck at next visit.
```

**Ground truth:**
```json
{"diagnoses": [{"name": "Essential hypertension"}], "medications": [{"name": "Lisinopril 10 MG Oral Tablet", "dosage": "Take 10 mg by mouth once daily"}], "vitals": [{"name": "Blood pressure", "value": "110/67", "unit": "mm[Hg]"}, {"name": "Body temperature", "value": 37.6, "unit": "Cel"}]}
```

**Base (no adapter):**
```
```json
{
  "diagnoses": ["Essential hypertension"],
  "medications": {
    "Lisinopril": {
      "Dosage": "Take 10 mg by mouth once daily"
    }
  },
  "vitals": {
    "BP": "110/67 mmHg",
    "Body Temperature": "37.6 Cel."
  }
}
```
```

**Fine-tuned:**
```
{"diagnoses": [{"name": "Essential hypertension"}], "medications": [{"name": "Lisinopril 10 MG Oral Tablet", "dosage": "Take 10 mg by mouth once daily"}], "vitals": [{"name": "Blood pressure", "value": "110/67", "unit": "mm[Hg]"}, {"name": "Body temperature", "value": 37.6, "unit": "Cel"}]}
```
