"""Stage 7 — evaluate the committed LoRA adapter on the frozen gold set.

Loads the ALREADY-TRAINED, ALREADY-COMMITTED adapter (training_results/adapter/)
and runs inference over data/gold/gold.jsonl — no training, no checkpoint writes.
Reuses train_runner.generate (greedy, do_sample=False) and its model.eval() rule
(MPS produces garbage in .train() mode). Scoring + gate + model card come from the
pure eval_metrics module.

Lane 1, local-only (needs requirements-train.txt for torch); ~1-2 min for the
gold set, $0 (local MPS inference, no API). torch is imported lazily so this
module and eval_metrics stay importable without the ML deps.

    python evaluate.py

Writes to training_results/: eval_report.json (metrics + release gate),
eval_predictions.jsonl (raw per-example outputs, for audit), model_card.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import eval_metrics
import registry

DATA_DIR = Path(__file__).resolve().parent / "data"
GOLD_PATH = DATA_DIR / "gold" / "gold.jsonl"
OUT_DIR = Path(__file__).resolve().parent / "training_results"
ADAPTER_DIR = OUT_DIR / "adapter"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _load_model(adapter_dir: Path):
    """Base model + committed adapter, ready for inference. torch imports are
    local so the pure metric path (and CI) never need the ML deps."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import train_runner

    device = train_runner.get_device()
    print(f"Device: {device} (MPS available: {torch.backends.mps.is_available()})")
    tokenizer = AutoTokenizer.from_pretrained(train_runner.BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(train_runner.BASE_MODEL, dtype=torch.float16)
    base_model.to(device)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.to(device)
    model.eval()  # required before generating on MPS (see train_runner docstring)
    return model, tokenizer, device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 7 evaluate committed adapter on the gold set."
    )
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--adapter", type=Path, default=ADAPTER_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    import torch

    import train_runner

    gold = _read_jsonl(args.gold)
    print(f"Evaluating the committed adapter on {len(gold)} gold examples ...")
    model, tokenizer, device = _load_model(args.adapter)

    pairs, raw = [], []
    for i, ex in enumerate(gold, 1):
        output_text = train_runner.generate(model, tokenizer, ex["instruction"], device)
        ground_truth = json.loads(ex["response"])
        pairs.append((ground_truth, output_text))
        raw.append(
            {
                "instruction": ex["instruction"],
                "ground_truth": ex["response"],
                "prediction": output_text,
            }
        )
        if device.type == "mps":
            torch.mps.empty_cache()
        print(f"  generated {i}/{len(gold)}")

    metrics = eval_metrics.evaluate_predictions(pairs)
    gate = eval_metrics.release_gate(metrics)
    report = {**metrics, "release_gate": gate, "thresholds": eval_metrics.RELEASE_THRESHOLDS}

    entry = registry.latest()
    if entry is None:
        raise SystemExit("no registered model — run train_runner.py first")
    card = eval_metrics.build_model_card(entry, metrics, gate)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_report.json").write_text(json.dumps(report, indent=2))
    with (args.out / "eval_predictions.jsonl").open("w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")
    (args.out / "model_card.md").write_text(card)

    dx, med = metrics["diagnosis"], metrics["medication"]
    jv = metrics["json_validity"]
    dxf = f"{dx['micro_precision']:.3f}/{dx['micro_recall']:.3f}/{dx['micro_f1']:.3f}"
    medf = f"{med['micro_precision']:.3f}/{med['micro_recall']:.3f}/{med['micro_f1']:.3f}"
    print(
        f"\nJSON validity: {jv['valid']}/{jv['total']}\n"
        f"Diagnosis P/R/F1: {dxf}\n"
        f"Medication P/R/F1: {medf}\n"
        f"Release gate: {'PASS' if gate['passed'] else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
