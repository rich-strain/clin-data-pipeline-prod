"""Stage 6 — LoRA fine-tune of a small open model on Stage 5's train/val JSONL.

Runs LOCALLY on Apple Silicon MPS (Lane 1, compute) — not in CI or the deployed
app, neither of which has a GPU. The results (adapter, loss curve, metrics,
before/after samples), the MLflow run, and the model-registry entry are committed
so the app displays a real completed run rather than re-training live.

    pip install -r requirements-train.txt
    python train_runner.py

**Base model: Qwen2.5-0.5B-Instruct** — modern (late-2024), already instruction-
tuned (so the fine-tune only shifts behavior toward this extraction format, not
teaches instruction-following from scratch), ~988 MB single safetensors, standard
Qwen2 attention projections (`q/k/v/o_proj`) that are a well-trodden LoRA target.
No quantization (bitsandbytes is poor on MPS; fp16 LoRA fits at 0.5B).

**LoRA `r=8, alpha=16`, attention projections only, dropout 0.05** — deliberately
modest: with ~98 training examples a bigger rank / MLP-inclusive target set would
just memorize the examples rather than learn the pattern. Loss is masked to the
assistant continuation (the model learns to *produce* the JSON, not predict the
note it reads). Plain PyTorch loop (not `Trainer`) so batching / loss-masking /
device placement stay directly inspectable.

**Prod additions over the sibling repos:** MLflow experiment tracking (local
file-backed `mlruns/`), and a call to `registry.register_run` recording data↔model
lineage (git SHA + content hashes of the exact train/val/gold snapshot).

**Honest scope:** ~98 examples + a 0.5B model for a few epochs is not a claim of
production extraction accuracy. It demonstrates, on real hardware and real
pipeline output: correct chat-template formatting + loss masking, LoRA attachment,
a real training loop with declining loss, per-epoch checkpointing + best-epoch
selection, and a visible base-vs-tuned behavioral difference.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import registry  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_PATH = DATA_DIR / "splits" / "train.jsonl"
VAL_PATH = DATA_DIR / "splits" / "val.jsonl"
OUT_DIR = Path(__file__).resolve().parent / "training_results"
MLFLOW_DB = Path(__file__).resolve().parent / "mlflow.db"  # sqlite backend (file store is EOL)

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
EXPERIMENT = "clin-extraction-lora"

SYSTEM_PROMPT = (
    "You are a clinical data extraction assistant. Extract the patient's "
    "diagnoses, medications, and vitals from the clinical note below. Respond "
    "with only a single JSON object with keys diagnoses, medications, vitals — "
    "no other text."
)

LORA_CONFIG = dict(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

NUM_EPOCHS = 6
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
MAX_LENGTH = 768

COLOR_TRAIN = "#0072B2"  # Okabe-Ito colorblind-safe
COLOR_VAL = "#E69F00"


def get_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def build_messages(instruction: str, response: str | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return messages


def encode_example(tokenizer, instruction: str, response: str, max_length: int) -> dict:
    """Tokenize one (instruction, response) pair, masking labels to the assistant
    continuation only (prefix labels set to -100)."""
    prompt_text = tokenizer.apply_chat_template(
        build_messages(instruction), tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        build_messages(instruction, response), tokenize=False, add_generation_prompt=False
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:max_length]

    labels = list(full_ids)
    for i in range(min(len(prompt_ids), len(full_ids))):
        labels[i] = -100
    return {"input_ids": full_ids, "labels": labels}


def collate(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids, labels, attention_mask = [], [], []
    for ex in batch:
        pad = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_token_id] * pad)
        labels.append(ex["labels"] + [-100] * pad)
        attention_mask.append([1] * len(ex["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def batches(examples: list[dict], batch_size: int, shuffle: bool, generator=None):
    order = (
        torch.randperm(len(examples), generator=generator).tolist()
        if shuffle
        else list(range(len(examples)))
    )
    for i in range(0, len(order), batch_size):
        yield [examples[j] for j in order[i : i + batch_size]]


def run_eval(model, examples, pad_token_id, device, batch_size) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in batches(examples, batch_size, shuffle=False):
            enc = {k: v.to(device) for k, v in collate(batch, pad_token_id).items()}
            out = model(**enc, use_cache=False)
            n = int((enc["labels"] != -100).sum().item())
            total_loss += out.loss.item() * n
            total_tokens += n
            del out
            if device.type == "mps":
                torch.mps.empty_cache()
    model.train()
    return total_loss / max(total_tokens, 1)


def generate(model, tokenizer, instruction, device, max_new_tokens=320) -> str:
    prompt_text = tokenizer.apply_chat_template(
        build_messages(instruction), tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(
        out_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6 LoRA fine-tune (Lane 1, MPS).")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--n-samples", type=int, default=3)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device} (MPS available: {torch.backends.mps.is_available()})")

    train_data = read_jsonl(TRAIN_PATH)
    val_data = read_jsonl(VAL_PATH)
    print(f"Loaded {len(train_data)} train / {len(val_data)} val examples")

    print(f"Loading base model {BASE_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16)
    base_model.to(device)

    torch.manual_seed(42)  # LoRA A matrices are RNG-initialized — seed before attach
    model = get_peft_model(base_model, LoraConfig(**LORA_CONFIG))
    model.to(device)
    model.print_trainable_parameters()
    # No gradient checkpointing: at 0.5B / batch 2 / short notes it fits in memory
    # on 17 GB, and the recompute it trades for memory roughly doubles MPS step
    # time for no benefit here (the sibling repos enabled it for a tighter box).

    train_encoded = [
        encode_example(tokenizer, e["instruction"], e["response"], MAX_LENGTH) for e in train_data
    ]
    val_encoded = [
        encode_example(tokenizer, e["instruction"], e["response"], MAX_LENGTH) for e in val_data
    ]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    history: dict = {"epoch": [], "train_loss": [], "val_loss": []}
    gen = torch.Generator().manual_seed(42)
    checkpoints_dir = args.out / "checkpoints"
    best_epoch, best_val_loss = 1, float("inf")

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "base_model": BASE_MODEL,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "max_length": MAX_LENGTH,
                **LORA_CONFIG,
                "train_examples": len(train_data),
                "val_examples": len(val_data),
            }
        )

        print(f"\nTraining {args.epochs} epochs on {device} ...")
        start = time.time()
        model.train()
        for epoch in range(1, args.epochs + 1):
            epoch_loss, epoch_tokens = 0.0, 0
            for batch in batches(train_encoded, args.batch_size, shuffle=True, generator=gen):
                enc = {k: v.to(device) for k, v in collate(batch, tokenizer.pad_token_id).items()}
                optimizer.zero_grad()
                out = model(**enc, use_cache=False)
                n = int((enc["labels"] != -100).sum().item())
                loss_value = out.loss.item()
                out.loss.backward()
                optimizer.step()
                del out, enc
                if device.type == "mps":
                    torch.mps.empty_cache()
                epoch_loss += loss_value * n
                epoch_tokens += n

            train_loss = epoch_loss / max(epoch_tokens, 1)
            val_loss = run_eval(model, val_encoded, tokenizer.pad_token_id, device, args.batch_size)
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            print(
                f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            )

            (checkpoints_dir / f"epoch_{epoch}").mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoints_dir / f"epoch_{epoch}")
            if val_loss < best_val_loss:
                best_val_loss, best_epoch = val_loss, epoch

        elapsed = time.time() - start
        print(f"\nDone in {elapsed:.1f}s. Best epoch: {best_epoch} (val_loss={best_val_loss:.4f})")

        model.load_adapter(
            str(checkpoints_dir / f"epoch_{best_epoch}"),
            adapter_name="default",
            torch_device=str(device),
        )
        model.eval()  # leave train mode before generating (dropout/ckpt hooks corrupt MPS gen)

        print(f"Generating {args.n_samples} before/after samples ...")
        samples = []
        for ex in val_data[: args.n_samples]:
            with model.disable_adapter():
                base_output = generate(model, tokenizer, ex["instruction"], device)
            samples.append(
                {
                    "instruction": ex["instruction"],
                    "ground_truth": ex["response"],
                    "base_model_output": base_output,
                    "fine_tuned_output": generate(model, tokenizer, ex["instruction"], device),
                }
            )

        # --- artifacts ---
        args.out.mkdir(parents=True, exist_ok=True)
        adapter_dir = args.out / "adapter"
        model.save_pretrained(adapter_dir)

        metrics = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "train_examples": len(train_data),
            "val_examples": len(val_data),
        }
        config_out = {
            "base_model": BASE_MODEL,
            "device": str(device),
            "lora_config": LORA_CONFIG,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_length": MAX_LENGTH,
            "training_seconds": round(elapsed, 1),
            "loss_history": history,
            **metrics,
        }
        (args.out / "config.json").write_text(json.dumps(config_out, indent=2))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(
            history["epoch"],
            history["train_loss"],
            marker="o",
            color=COLOR_TRAIN,
            label="Train loss",
        )
        ax.plot(
            history["epoch"], history["val_loss"], marker="o", color=COLOR_VAL, label="Val loss"
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy loss")
        ax.set_title(
            f"LoRA fine-tune of {BASE_MODEL}\n{len(train_data)} train / {len(val_data)} val",
            fontsize=11,
        )
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out / "loss_curve.png", dpi=150)
        plt.close(fig)

        samples_md = ["# Before/after generation samples\n"]
        for i, s in enumerate(samples, 1):
            samples_md += [
                f"## Sample {i}\n",
                f"**Instruction:**\n```\n{s['instruction']}\n```\n",
                f"**Ground truth:**\n```json\n{s['ground_truth']}\n```\n",
                f"**Base (no adapter):**\n```\n{s['base_model_output']}\n```\n",
                f"**Fine-tuned:**\n```\n{s['fine_tuned_output']}\n```\n",
            ]
        (args.out / "samples.md").write_text("\n".join(samples_md))
        (args.out / "samples.json").write_text(json.dumps(samples, indent=2))

        # Artifacts live in committed training_results/; MLflow tracks params +
        # per-epoch metrics + the run id (recorded in the registry), not a
        # duplicate artifact copy.
        entry = registry.register_run(
            base_model=BASE_MODEL,
            lora_config=LORA_CONFIG,
            metrics=metrics,
            adapter_dir=adapter_dir,
            mlflow_run_id=run.info.run_id,
        )

    print(f"\nSaved adapter + artifacts to {args.out}")
    print(f"Registered model {entry['version']} (git {entry['git_sha'][:12]})")


if __name__ == "__main__":
    main()
