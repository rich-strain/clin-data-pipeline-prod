# Model card — clinical extraction LoRA (v2)

## Overview
- **Base model:** Qwen/Qwen2.5-0.5B-Instruct (~0.5B params)
- **Adapter:** LoRA r=8, 4.4 MB, best epoch 6
- **Task:** extract diagnoses / medications / vitals from a de-identified note as JSON.
- **Trained on:** 886 examples; git `6097ab1d5db5`, gold set `v1`.

## Evaluation (frozen gold set)
- **JSON validity:** 111/111 (100%)
- **Diagnosis micro P/R/F1:** 1.000 / 1.000 / 1.000 (TP 186, FP 0, FN 0, non-canonical 0)
- **Medication micro P/R/F1:** 1.000 / 1.000 / 1.000 (TP 187, FP 0, FN 0, non-canonical 0)

## Release gate: ✅ PASS
- ✅ `json_validity_rate` = 1.0 (needs >= 0.9)
- ✅ `diagnosis_micro_f1` = 1.0 (needs >= 0.8)
- ✅ `medication_micro_f1` = 1.0 (needs >= 0.7)
- ✅ `non_canonical_total` = 0 (needs <= 2)

## Intended use & limits (honest framing)
- **Synthetic-data demonstration**, not a production medical device. A 0.5B model fine-tuned on ~100 synthetic notes is not a claim of clinical accuracy.
- Vitals are not scored (value-closeness is out of scope). Metrics are over a small (111-example) frozen gold set — treat as directional.
- Closed vocabulary only; behavior on out-of-vocabulary clinical language is untested.
- Not for clinical decisions. Inputs are synthetic, de-identified upstream (no PHI).
