"""ShareGPT conversion and assistant-only supervision using the native template."""
import json


def load_trajectories(path):
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty JSON array of trajectories")
    return rows


def messages_from_row(row):
    roles = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant", "system": "system"}
    return [{"role": roles[item.get("from", item.get("role"))],
             "content": item.get("value", item.get("content"))}
            for item in row.get("conversations", row.get("agent_conversations", []))]


def prompt_ids(tokenizer, messages):
    # Keep the exact prior generation prefixes. Qwen3's full-history template
    # strips old think blocks, which otherwise changes earlier token positions.
    ids = []
    for message in messages:
        if message["role"] == "assistant":
            ids.extend(tokenizer.encode(message["content"] + tokenizer.eos_token + "\n", add_special_tokens=False))
        else:
            ids.extend(tokenizer.apply_chat_template([message], tokenize=True,
                add_generation_prompt=message["role"] == "user", enable_thinking=False))
    return ids


def encode_trajectory(tokenizer, messages, max_length):
    # Render each completed turn separately. This preserves Qwen3's generation
    # prefix (including the empty think block) identically in SFT and rollout.
    ids, labels, ends = [], [], []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = prompt_ids(tokenizer, messages[:index])
        suffix = tokenizer.encode(message["content"] + tokenizer.eos_token + "\n", add_special_tokens=False)
        if prefix[:len(ids)] != ids:
            raise ValueError("Chat template changed previous turns; cannot align assistant labels")
        if len(prefix) + len(suffix) > max_length:
            break  # Keep whole actions rather than supervising a partial action.
        labels.extend([-100] * (len(prefix) - len(ids)))
        ids = prefix + suffix
        labels.extend(suffix)
        ends.append(len(ids) - len(tokenizer.encode("\n", add_special_tokens=False)) - 1)
    if not ends:
        raise ValueError("No complete assistant turn fits in max_length")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels, "ends": ends}


def gae(rewards, values, gamma=0.99, lam=0.95):
    """GAE on assistant-token time steps, with a terminal finite-horizon state."""
    if len(rewards) != len(values):
        raise ValueError("Rewards and values must align")
    advantages = [0.0] * len(rewards)
    carry, next_value = 0.0, 0.0
    for index in reversed(range(len(rewards))):
        delta = rewards[index] + gamma * next_value - values[index]
        carry = delta + gamma * lam * carry
        advantages[index] = carry
        next_value = values[index]
    return advantages, [a + v for a, v in zip(advantages, values)]
