"""Online WebShop rollout shared by collection, evaluation and PPO."""
import time
import torch
from .data import prompt_ids
from .grounding import parse_action, validate_action


def collect(model, tokenizer, env, task_id, instruction, max_length=2048,
            max_new_tokens=128, max_steps=12, judge=None, greedy=False):
    env.reset(task_id)
    observation = env.observation
    task = env.get_instruction_text()
    messages = [{"role": "system", "content": instruction}, {"role": "user", "content": observation}]
    ids, mask, ends, grounding, events, history = [], [], [], [], [], []
    final_reward, done = 0.0, False
    started = time.perf_counter()
    model.eval()
    prefix = prompt_ids(tokenizer, messages)
    for _ in range(max_steps):
        if len(prefix) + max_new_tokens > max_length:
            break
        if prefix[:len(ids)] != ids:
            raise ValueError("Rollout history changed tokenization")
        inputs = torch.tensor([prefix], device=next(model.parameters()).device)
        with torch.no_grad():
            generated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                max_new_tokens=max_new_tokens, do_sample=not greedy, use_cache=True,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                **({} if greedy else {"temperature": 1.0, "top_p": 1.0, "top_k": 0}))
        response_ids = generated[0, len(prefix):].tolist()
        text = tokenizer.decode(response_ids, skip_special_tokens=True)
        # Preserve the exact sampled tokens for the PPO likelihood ratio.
        mask.extend([0] * (len(prefix) - len(ids)) + [1] * len(response_ids))
        ids = prefix + response_ids
        ends.append(len(ids) - 1)
        available = env.get_available_actions()
        try:
            action = parse_action(text)
            valid, reason = validate_action(action, available, history, observation)
        except ValueError:
            action, valid, reason = "", False, "invalid_format"
        score = 1.0 if valid else -1.0
        judge_score = None
        if valid and judge is not None:
            judge_score = judge.score(task, observation, action)
            valid = judge_score > 0
            score = judge_score
            reason = "judge_valid" if valid else "judge_rejected"
        previous = observation
        if valid:
            observation, final_reward, done, _ = env.step(action)
            if observation == previous and not done:
                score, reason = -1.0, "no_state_change"
        else:
            observation = previous
        history.append((previous, action))
        grounding.append(score)
        events.append({"action": action, "rule_valid": reason not in {
            "invalid_format", "search_unavailable", "target_unavailable", "repeated_state_action"},
            "grounding": score, "judge": judge_score, "reason": reason, "done": bool(done)})
        messages.append({"role": "assistant", "content": text})
        if done:
            break
        feedback = observation if valid else observation + "\nInvalid action: " + reason
        messages.append({"role": "user", "content": feedback})
        # Native Qwen history may change whitespace/token boundaries. Build the
        # next prefix by retaining generated IDs and appending only the new turn.
        newline = tokenizer.encode("\n", add_special_tokens=False)
        if ids[-1] == tokenizer.eos_token_id:
            tail = newline
        else:
            tail = tokenizer.encode(tokenizer.eos_token + "\n", add_special_tokens=False)
        new_user = prompt_ids(tokenizer, [messages[-1]])
        # Store token history separately; it is never reconstructed from text.
        ids.extend(tail)
        mask.extend([0] * len(tail))
        prefix = ids + new_user
    return {"input_ids": ids, "mask": mask, "ends": ends, "grounding": grounding,
            "conversations": [{"from": {"system": "system", "user": "human", "assistant": "gpt"}[m["role"]],
                               "value": m["content"]} for m in messages],
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "agent_final_reward": float(final_reward), "success": bool(done and final_reward >= 1.0),
            "duration_seconds": time.perf_counter() - started, "events": events, "id": task_id}
