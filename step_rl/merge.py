import argparse
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Original base model")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise ValueError("Choose a new export directory")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    PeftModel.from_pretrained(model, args.adapter).merge_and_unload().save_pretrained(args.output)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.output)


if __name__ == "__main__":
    main()
