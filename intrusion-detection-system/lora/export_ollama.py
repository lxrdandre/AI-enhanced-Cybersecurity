"""Merge LoRA adapter into base model, export to GGUF, and register with Ollama.

Usage::

    python -m lora.export_ollama \\
        --adapter lora_adapter \\
        --output model_gguf \\
        --model-name clawdbot-triage
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Redirect HuggingFace cache to /data to avoid filling /home
os.environ.setdefault("HF_HOME", "/data/huggingface")
os.environ.setdefault("XDG_CACHE_HOME", "/data/.cache")


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="Export LoRA adapter to Ollama")
    parser.add_argument("--adapter", required=True, help="LoRA adapter directory")
    parser.add_argument(
        "--base-model",
        default="unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit",
        help="Base model (usually auto-detected from adapter config)",
    )
    parser.add_argument("--output", default="model_gguf", help="GGUF output directory")
    parser.add_argument("--model-name", default="clawdbot-triage", help="Ollama model name")
    parser.add_argument(
        "--quantization", default="q8_0",
        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
        help="GGUF quantization level (q8_0 recommended for H200)",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        print(f"Adapter not found: {adapter_path}", file=sys.stderr)
        sys.exit(1)

    # -- Lazy imports -----------------------------------------------------
    print("Loading libraries...")
    from unsloth import FastLanguageModel

    # -- Load base model + LoRA adapter -----------------------------------
    print(f"Loading adapter: {args.adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # -- Merge and save as GGUF -------------------------------------------
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Merging adapter and exporting GGUF ({args.quantization})...")
    model.save_pretrained_gguf(
        str(output_dir),
        tokenizer,
        quantization_method=args.quantization,
    )

    gguf_files = list(output_dir.glob("*.gguf"))
    if not gguf_files:
        print("ERROR: No GGUF file produced", file=sys.stderr)
        sys.exit(1)
    gguf_path = gguf_files[0]
    print(f"GGUF file: {gguf_path}  ({gguf_path.stat().st_size / 1e9:.1f} GB)")

    # -- Create Ollama Modelfile ------------------------------------------
    modelfile_path = output_dir / "Modelfile"
    modelfile_content = (
        f'FROM ./{gguf_path.name}\n'
        '\n'
        'SYSTEM "You are a SOC analyst assistant. Respond ONLY with valid JSON '
        'matching the requested output_schema. No markdown fences, no commentary."\n'
        '\n'
        'PARAMETER temperature 0.1\n'
        f'PARAMETER num_ctx {args.max_seq_length}\n'
    )
    modelfile_path.write_text(modelfile_content)
    print(f"Modelfile: {modelfile_path}")

    # -- Register with Ollama ---------------------------------------------
    print(f"\nRegistering with Ollama as '{args.model_name}'...")
    result = subprocess.run(
        ["ollama", "create", args.model_name, "-f", str(modelfile_path)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"Success! Model registered as: {args.model_name}")
        print(f"Verify with:  ollama list | grep {args.model_name}")
        print(f"Test with:    ollama run {args.model_name}")
    else:
        print(f"Ollama registration failed:\n{result.stderr}", file=sys.stderr)
        print(f"\nManual registration:")
        print(f"  cd {output_dir}")
        print(f"  ollama create {args.model_name} -f Modelfile")


if __name__ == "__main__":
    main()
