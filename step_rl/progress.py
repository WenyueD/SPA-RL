"""Train a small MLP to attribute final return to assistant turn ends."""
import argparse
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .data import encode_trajectory, load_trajectories, messages_from_row


def make_head(hidden_size):
    return nn.Sequential(nn.Linear(hidden_size, 128), nn.SiLU(), nn.Linear(128, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Merged SFT model")
    parser.add_argument("--data", required=True, help="Exploration JSON with agent_final_reward or reward")
    parser.add_argument("--output", default="ckpt/qwen3_progress.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device).eval()
    model.requires_grad_(False)
    head = make_head(model.config.hidden_size).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    # Cache frozen features once. No vocabulary projection or full-model tuning.
    features = []
    for row in load_trajectories(args.data):
        messages = messages_from_row(row)
        # Ignore the protocol acknowledgement; only actual operations get scores.
        expected = sum(m["role"] == "assistant" for m in messages)
        if "input_ids" in row:
            item = row
            ends = row["ends"]
            if len(item["input_ids"]) > args.max_length:
                continue
        else:
            item = encode_trajectory(tokenizer, messages, args.max_length)
            if len(item["ends"]) != expected:
                continue  # A truncated trajectory must not inherit its final return.
            ends = [end for end, m in zip(item["ends"], [m for m in messages if m["role"] == "assistant"])
                    if "Action:" in m["content"]]
        if not ends:
            continue
        with torch.no_grad():
            hidden = model.base_model(input_ids=torch.tensor([item["input_ids"]], device=device),
                use_cache=False).last_hidden_state[0, ends].float().cpu()
        reward = float(row.get("agent_final_reward", row.get("reward")))
        features.append((hidden, reward))
    if not features:
        raise ValueError("No complete action trajectories fit the context length")
    for epoch in range(args.epochs):
        total = 0.0
        for hidden, reward in features:
            prediction = head(hidden.to(device)).sum()
            loss = (prediction - reward).square()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print({"epoch": epoch, "mse": total / len(features), "trajectories": len(features)})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.cpu().state_dict(), "hidden_size": model.config.hidden_size,
                "sft_model": args.model}, args.output)


if __name__ == "__main__":
    main()
