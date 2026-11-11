"""Online, token-level clipped PPO with shared SFT reference and MLP attribution.

One complete trajectory per rank is the mini-batch. Accelerate synchronizes
gradients across ranks. Each new update collects fresh on-policy trajectories.
"""
import argparse
import json
import math
import random
from pathlib import Path
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .data import gae
from .grounding import QwenJudge
from .progress import make_head
from .rollout import collect


def token_outputs(model, value_head, ids):
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                   output_hidden_states=True, use_cache=False)
    # logits at t predict the sampled token at t+1, as does V(s_t).
    targets = ids[:, 1:]
    logits = output.logits[:, :-1].float()
    logprobs = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)
    values = value_head(output.hidden_states[-1][:, :-1].float()).squeeze(-1)
    return logprobs[0], values[0], output.hidden_states[-1][0]


def clipped_loss(logprobs, old_logprobs, values, old_values, advantages, returns, mask, clip=0.2):
    ratio = (logprobs[mask] - old_logprobs[mask]).exp()
    policy = -torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages).mean()
    predicted = values[mask]
    clipped = old_values[mask] + (predicted - old_values[mask]).clamp(-clip, clip)
    value = 0.5 * torch.maximum((predicted - returns).square(), (clipped - returns).square()).mean()
    return policy, value


def make_env():
    from webshop.web_agent_site.envs import WebAgentTextEnv
    return WebAgentTextEnv(observation_mode="text")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Merged SFT base; adapters disabled gives the reference")
    parser.add_argument("--progress", help="Head checkpoint from step_rl.progress, required for PPO")
    parser.add_argument("--indices", default="eval_agent/data/webshop/train_indices.json")
    parser.add_argument("--output", default="ckpt/qwen3_step_rl")
    parser.add_argument("--episodes", type=int, default=1000, help="Global episode budget (rounded up to world size)")
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--progress-weight", type=float, default=1.0)
    parser.add_argument("--grounding-weight", type=float, default=0.5)
    parser.add_argument("--kl-coef", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--mode", choices=["train", "collect", "eval"], default="train")
    args = parser.parse_args()
    if args.episodes < 1 or args.ppo_epochs < 1 or args.max_steps < 1 or args.max_new_tokens < 1:
        parser.error("Episode, epoch and step counts must be positive")
    if args.mode == "train" and not args.progress:
        parser.error("PPO requires --progress")
    accelerator = Accelerator(mixed_precision="bf16")
    set_seed(42 + accelerator.process_index)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa")
    hidden_size = model.config.hidden_size
    progress = None
    if args.progress:
        state = torch.load(args.progress, map_location="cpu", weights_only=True)
        if state["sft_model"] != args.model or state["hidden_size"] != hidden_size:
            raise ValueError("Progress head must match the exact merged SFT model")
        progress = make_head(hidden_size).to(accelerator.device).eval()
        progress.load_state_dict(state["head"])
    value_head = nn.Linear(hidden_size, 1).to(accelerator.device)
    if args.mode == "train":
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
            target_modules=["q_proj", "v_proj", "gate_proj", "up_proj"], task_type="CAUSAL_LM"))
        # Keep optimizer states and trainable LoRA weights in FP32.
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad] + list(value_head.parameters()), lr=args.lr)
        model, value_head, optimizer = accelerator.prepare(model, value_head, optimizer)
    else:
        model.to(accelerator.device)
    env = make_env()
    indices = json.loads(Path(args.indices).read_text(encoding="utf-8"))
    if not indices:
        raise ValueError("Empty task indices")
    random.Random(42).shuffle(indices)
    instruction = Path("eval_agent/prompt/instructions/webshop_inst.txt").read_text(encoding="utf-8")
    judge = QwenJudge() if args.judge else None
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory to preserve previous runs")
    accelerator.wait_for_everyone()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    episodes_per_rank = math.ceil(args.episodes / accelerator.num_processes)
    for iteration in range(episodes_per_rank):
        policy_model = accelerator.unwrap_model(model)
        index = iteration * accelerator.num_processes + accelerator.process_index
        row = collect(policy_model, tokenizer, env, indices[index % len(indices)], instruction,
                      args.max_length, args.max_new_tokens, args.max_steps, judge, args.mode == "eval")
        if not row["ends"]:
            raise ValueError("Initial observation exceeds context budget; increase --max-length")
        if args.mode == "train":
            ids = torch.tensor([row["input_ids"]], device=accelerator.device)
            mask = torch.tensor(row["mask"][1:], dtype=torch.bool, device=accelerator.device)
            # Rollout and likelihood evaluations use eval mode to disable dropout.
            model.eval()
            with torch.no_grad(), accelerator.autocast():
                old_logprobs, old_values, _ = token_outputs(policy_model, accelerator.unwrap_model(value_head), ids)
                with policy_model.disable_adapter():
                    ref_logprobs, _, reference_hidden = token_outputs(policy_model, accelerator.unwrap_model(value_head), ids)
                contributions = progress(reference_hidden[row["ends"]].float()).squeeze(-1)
                rewards = -args.kl_coef * (old_logprobs - ref_logprobs)
                combined = []
                for step, end in enumerate(row["ends"]):
                    score = args.progress_weight * contributions[step] + args.grounding_weight * row["grounding"][step]
                    rewards[end - 1] += score  # Includes the final action, with no position multiplier.
                    combined.append(float(score))
                advantages_list, returns_list = gae(rewards[mask].float().tolist(), old_values[mask].float().tolist())
                advantages = torch.tensor(advantages_list, device=accelerator.device)
                returns = torch.tensor(returns_list, device=accelerator.device)
                advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
                row["progress"] = contributions.float().tolist()
                row["reward"] = combined
            del reference_hidden, ref_logprobs
            model.train()  # Enable gradient checkpointing for update forwards.
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    module.eval()  # Match dropout-free rollout likelihoods.
            for _ in range(args.ppo_epochs):
                with accelerator.autocast():
                    logprobs, values, _ = token_outputs(model, value_head, ids)
                    policy_loss, value_loss = clipped_loss(logprobs, old_logprobs, values, old_values,
                                                          advantages, returns, mask)
                    loss = policy_loss + 0.1 * value_loss
                accelerator.backward(loss)
                accelerator.clip_grad_norm_([p for p in model.parameters() if p.requires_grad] + list(value_head.parameters()), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        records.append(row)
        # Rank-local files avoid concurrent writes and retain raw evaluation evidence.
        (output / f"trajectories-rank{accelerator.process_index}.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print({"rank": accelerator.process_index, "episode": index, "return": row["agent_final_reward"],
               "steps": len(row["ends"]), "seconds": row["duration_seconds"]}, flush=True)
    accelerator.wait_for_everyone()
    if args.mode == "train" and accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(output / "adapter")
        tokenizer.save_pretrained(output / "adapter")
        torch.save(accelerator.unwrap_model(value_head).state_dict(), output / "value_head.pt")


if __name__ == "__main__":
    main()
