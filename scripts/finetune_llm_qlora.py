#!/usr/bin/env python3
"""
Phase 7.2: Fine-Tune Llama-3-8B (or smaller) with QLoRA on Medical Q&A

Uses TRL SFTTrainer + PEFT QLoRA for parameter-efficient fine-tuning.
- On CUDA: Full QLoRA (4-bit quantization + LoRA)
- On MPS/CPU: LoRA without quantization (use smaller model e.g. Llama-3.2-3B)

Usage:
    # Full QLoRA (requires CUDA GPU with 16GB+ VRAM)
    python scripts/finetune_llm_qlora.py --model meta-llama/Llama-3.2-3B-Instruct

    # LoRA only on Apple Silicon (use 3B model to fit memory)
    python scripts/finetune_llm_qlora.py --model meta-llama/Llama-3.2-3B-Instruct --no-quantize

    # Quick test run
    python scripts/finetune_llm_qlora.py --max-samples 20 --epochs 1
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

# Use project-local HuggingFace cache to avoid permission issues
_hf_cache = _project_root / "data" / "hf_cache"
_hf_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_hf_cache))
os.environ.setdefault("HF_HUB_CACHE", str(_hf_cache / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_hf_cache / "hub"))

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _format_prompt(example: dict, use_llama3_format: bool = False) -> str:
    """Format instruction/input/output into chat template.
    TinyLlama/others use <|system|>/<|user|>/<|assistant|>, Llama3 uses different tokens.
    """
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")
    if use_llama3_format:
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{instruction}<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n\n{inp}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
        )
    # TinyLlama / generic ChatML format
    return (
        f"<|system|>\n{instruction}\n<|user|>\n{inp}\n<|assistant|>\n{output}"
    )


def finetune(
    data_path: Path,
    output_dir: Path,
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
    use_4bit: bool = True,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_target_modules: list = None,
    max_seq_length: int = 1024,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    max_samples: int = None,
    seed: int = 42,
) -> Path:
    """
    Fine-tune an LLM with QLoRA (or LoRA) on medical Q&A data.

    Returns:
        Path to saved adapter/model
    """
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    device = get_device()
    use_4bit = use_4bit and device == "cuda"  # bitsandbytes requires CUDA

    if use_4bit:
        print("Using QLoRA (4-bit quantization + LoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
        print("Using LoRA (no quantization) - suitable for MPS/CPU")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Set padding side for training
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    model_kwargs = {}
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
        model_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        **model_kwargs,
    )

    if bnb_config:
        model = prepare_model_for_kbit_training(model)

    # LoRA config
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load dataset
    dataset = load_dataset("json", data_files=str(data_path), split="train")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Using subset: {len(dataset)} samples")

    # Format for chat (use Llama3 format only for meta-llama models)
    use_llama3 = "llama" in model_id.lower() and "meta-llama" in model_id.lower()

    def format_example(ex):
        return {"text": _format_prompt(ex, use_llama3_format=use_llama3)}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    # Training args (SFTConfig includes max_length for TRL)
    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
        gradient_checkpointing=False,  # More compatible with MPS/CPU
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=False,
        bf16=device == "cuda",
        remove_unused_columns=False,
        report_to="none",
        seed=seed,
        use_mps_device=(device == "mps"),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\nStarting training...")
    trainer.train()
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))

    # Save training config for eval
    config = {
        "model_id": model_id,
        "base_model": model_id,
        "adapter_path": str(output_dir / "final_adapter"),
        "use_4bit": use_4bit,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "max_seq_length": max_seq_length,
        "epochs": epochs,
        "train_samples": len(dataset),
    }
    with open(output_dir / "finetune_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nAdapter saved to {output_dir / 'final_adapter'}")
    return output_dir / "final_adapter"


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune LLM with QLoRA on medical Q&A"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/finetune/medqa_train.jsonl"),
        help="Training data JSONL path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/llm_finetune"),
        help="Output directory for adapters",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model ID. Use TinyLlama (ungated) or meta-llama/Llama-3.2-3B-Instruct (gated, needs HF login)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable 4-bit quantization (for MPS/CPU)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=64,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=128,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Max sequence length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit training samples (for quick tests)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Training data not found: {args.data}")
        print("Run: python scripts/generate_finetune_data.py")
        sys.exit(1)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    finetune(
        data_path=args.data,
        output_dir=output_dir,
        model_id=args.model,
        use_4bit=not args.no_quantize,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
