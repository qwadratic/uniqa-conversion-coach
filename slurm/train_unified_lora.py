"""
Unified LoRA SFT: ONE model for all personas (tagged with short "persona: X" system prompt).

Supports multi-GPU via torchrun/accelerate (DDP). No device_map="auto" — let DDP handle it.

    # Single GPU:
    python slurm/train_unified_lora.py --base ~/models/minicpm5-1b --data slurm/data_v4_unified

    # 4 GPU (torchrun):
    torchrun --nproc_per_node=4 slurm/train_unified_lora.py \
        --base ~/models/minicpm5-1b --data slurm/data_v4_unified --out slurm/out_v4_unified
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="openbmb/MiniCPM5-1B")
    ap.add_argument("--data", default="slurm/data_v4_unified")
    ap.add_argument("--out", default="slurm/out_v4_unified")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=4096)
    args = ap.parse_args(argv)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    data = Path(args.data)
    ds = load_dataset("json", data_files={
        "train": str(data / "train.jsonl"),
        "validation": str(data / "val.jsonl"),
    })

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # For DDP: no device_map; let trainer place model
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # device_map only for single-GPU; DDP needs bare model
        **({} if world_size > 1 else {"device_map": "auto"}),
    )

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # effective batch = bsz × grad_accum × world_size
    # 4 × 4 × 4 = 64 effective
    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.bsz,
        eval_strategy="epoch",
        logging_steps=10,
        bf16=True,
        max_length=args.max_len,
        gradient_checkpointing=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        assistant_only_loss=True,
        packing=False,
        report_to=[],
        # DDP settings
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        peft_config=lora,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tok,
    )

    trainer.train()

    # Only rank 0 saves
    if local_rank == 0:
        trainer.save_model(args.out)
        tok.save_pretrained(args.out)
        print(f"saved unified LoRA adapter -> {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
