"""
Coach LoRA SFT: train a coach decision model on MiniCPM-1B.

System prompt: "coach"
User: observation context (filtered feed + step + budget)
Assistant: coach JSON decision (persona_belief, pains, command, etc.)

Supports multi-GPU via torchrun (DDP).

    # 4 GPU:
    torchrun --nproc_per_node=4 slurm/train_coach_lora.py \
        --base ~/models/minicpm5-1b \
        --data ~/zero-one/leonardo/coach_data/coach_sft \
        --out ~/zero-one/leonardo/out_coach_lora
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="openbmb/MiniCPM5-1B")
    ap.add_argument("--data", required=True, help="Dir with train.jsonl + val.jsonl")
    ap.add_argument("--out", default="leonardo/out_coach_lora")
    ap.add_argument("--epochs", type=float, default=5.0)
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

    if local_rank == 0:
        print(f"Train: {len(ds['train'])} | Val: {len(ds['validation'])}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        **({} if world_size > 1 else {"device_map": "auto"}),
    )

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # Coach data is smaller than persona data → more epochs, same effective batch
    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.bsz,
        eval_strategy="epoch",
        logging_steps=5,
        bf16=True,
        max_length=args.max_len,
        gradient_checkpointing=True,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        assistant_only_loss=True,
        packing=False,
        report_to=[],
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

    if local_rank == 0:
        trainer.save_model(args.out)
        tok.save_pretrained(args.out)
        print(f"Saved coach LoRA adapter -> {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
