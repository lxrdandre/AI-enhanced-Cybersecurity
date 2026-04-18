"""QLoRA fine-tuning for ClawdBot SOC triage model using Unsloth.

Trains a LoRA adapter on top of the base Mistral model using synthetic
or real triage examples in conversation JSONL format.

Usage::

    python -m lora.train \\
        --dataset data/triage_train.jsonl \\
        --output lora_adapter \\
        --epochs 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Redirect HuggingFace cache to /data to avoid filling /home
os.environ.setdefault("HF_HOME", "/data/huggingface")
os.environ.setdefault("XDG_CACHE_HOME", "/data/.cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune triage LLM with QLoRA")
    parser.add_argument("--dataset", required=True, help="Training JSONL (messages format)")
    parser.add_argument(
        "--base-model",
        default="unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument("--output", default="lora_adapter", help="Output directory for adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    # ── Lazy imports (heavy dependencies) ────────────────────────────────
    print("Loading libraries...")
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    import torch

    # Disable caching_allocator_warmup in Transformers 5.5+ — it
    # pre-allocates based on full-precision sizes, causing OOM on
    # quantised models.
    import transformers.modeling_utils as _mu
    if hasattr(_mu, "caching_allocator_warmup"):
        _mu.caching_allocator_warmup = lambda *a, **kw: None

    # Transformers 5.5+ Trainer._move_model_to_device calls model.to()
    # which dequantises bitsandbytes 4-bit weights into full precision,
    # blowing up 14 GB → 140 GB → OOM.  The quantised model is already
    # on GPU, so we no-op the move at the *class* level before any
    # trainer is instantiated.
    from transformers import Trainer as _Trainer
    _Trainer._move_model_to_device = lambda self, model, device: None

    # ── Sanity-check: refuse to start unless GPU has enough free VRAM ──
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        free_gb = free / 1024**3
        print(f"GPU free memory: {free_gb:.1f} GB / {total / 1024**3:.1f} GB")
        if free_gb < 20:
            print(
                f"ERROR: Only {free_gb:.1f} GB free on GPU — need at least 20 GB.\n"
                "Kill zombie processes:  sudo nvidia-smi --query-compute-apps=pid "
                "--format=csv,noheader | sudo xargs -r kill -9",
                file=sys.stderr,
            )
            sys.exit(1)

    # ── Load model ───────────────────────────────────────────────────────
    print(f"Loading base model: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Tell Trainer the model is already placed — prevents .to(device)
    # which dequantises 4-bit weights → OOM.  See _move_model_to_device:
    # if getattr(model, "hf_device_map", None) is not None: return
    if not getattr(model, "hf_device_map", None):
        model.hf_device_map = {"": 0}

    # ── Add LoRA adapter ─────────────────────────────────────────────────
    print(f"Adding LoRA adapter (rank={args.lora_rank})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # ── Load dataset ─────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset}")
    records = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("Dataset is empty", file=sys.stderr)
        sys.exit(1)

    def _format(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(records).map(_format)
    print(f"Training examples: {len(dataset)}")

    # ── Train ────────────────────────────────────────────────────────────
    FastLanguageModel.for_training(model)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=5,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=not use_bf16,
            bf16=use_bf16,
            logging_steps=1,
            output_dir=args.output,
            optim="adamw_8bit",
            seed=args.seed,
            save_strategy="epoch",
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
        ),
    )

    print(f"\nTraining for {args.epochs} epochs...")
    stats = trainer.train()
    print(f"\nTraining complete.  Final loss: {stats.training_loss:.4f}")

    # ── Save adapter ─────────────────────────────────────────────────────
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"LoRA adapter saved to: {args.output}/")


if __name__ == "__main__":
    main()
