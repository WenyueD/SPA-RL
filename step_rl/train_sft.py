"""LoRA SFT and export of a merged Qwen3 model."""
import argparse
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from .data import encode_trajectory, load_trajectories, messages_from_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--data", default="data/webshop_sft.json")
    parser.add_argument("--output", default="ckpt/qwen3_webshop_sft")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--merge", action="store_true", help="Single-process export after training")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "gate_proj", "up_proj"], task_type="CAUSAL_LM"))
    model.enable_input_require_grads()
    dataset = []
    for row in load_trajectories(args.data):
        item = encode_trajectory(tokenizer, messages_from_row(row), args.max_length)
        item.pop("ends")
        dataset.append(item)
    trainer = Trainer(model=model, train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
        args=TrainingArguments(output_dir=args.output, num_train_epochs=args.epochs,
            per_device_train_batch_size=1, gradient_accumulation_steps=4, learning_rate=2e-4,
            bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            ddp_find_unused_parameters=False, save_strategy="epoch", logging_steps=5, report_to="none"))
    trainer.train()
    trainer.save_model(args.output)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output)
    if args.merge:
        if trainer.args.world_size != 1:
            raise ValueError("Use --merge only for single-process SFT, or export with step_rl.merge")
        model.merge_and_unload().save_pretrained(Path(args.output) / "merged")
        tokenizer.save_pretrained(Path(args.output) / "merged")


if __name__ == "__main__":
    main()
