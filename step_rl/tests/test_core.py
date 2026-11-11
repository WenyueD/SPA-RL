import unittest
from unittest.mock import patch
from step_rl.data import encode_trajectory, gae, prompt_ids
from step_rl.grounding import QwenJudge, parse_action, validate_action
from step_rl.metrics import summarize


class TemplateTokenizer:
    """Character tokenizer preserving the Qwen chat/empty-think boundaries."""
    eos_token = "<|im_end|>"

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        text = "".join("<|im_start|>" + m["role"] + "\n" + m["content"] + self.eos_token + "\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return self.encode(text)


class CoreTests(unittest.TestCase):
    def test_multiturn_assistant_mask_and_prefix(self):
        tokenizer = TemplateTokenizer()
        messages = [{"role": "user", "content": "task"},
                    {"role": "assistant", "content": "Action: search[x]"},
                    {"role": "user", "content": "result"},
                    {"role": "assistant", "content": "Action: click[buy now]"}]
        item = encode_trajectory(tokenizer, messages, 4096)
        self.assertEqual(len(item["ends"]), 2)
        self.assertEqual(len(item["labels"]), len(item["input_ids"]))
        prefix = prompt_ids(tokenizer, messages[:3])
        self.assertEqual(item["input_ids"][:len(prefix)], prefix)
        self.assertTrue(all(label == -100 for label in item["labels"][:len(prompt_ids(tokenizer, messages[:1]))]))
        shortened = encode_trajectory(tokenizer, messages, len(prefix))
        self.assertEqual(len(shortened["ends"]), 1)

    def test_gae_propagates_terminal_reward(self):
        advantages, returns = gae([0, 0, 1], [0, 0, 0], gamma=1, lam=1)
        self.assertEqual(advantages, [1, 1, 1])
        self.assertEqual(returns, [1, 1, 1])

    def test_grounding_and_repeated_states(self):
        available = {"has_search_bar": False, "clickables": ["Buy Now"]}
        action = parse_action("Thought: purchase\nAction: click[Buy Now]")
        self.assertTrue(validate_action(action, available, [], "page")[0])
        self.assertFalse(validate_action("search[x]", available, [], "page")[0])
        self.assertFalse(validate_action("click[missing]", available, [], "page")[0])
        self.assertFalse(validate_action(action, available, [("page", action)] * 2, "page")[0])
        self.assertTrue(validate_action(action, available, [("other", action)] * 2, "page")[0])
        with self.assertRaises(ValueError):
            parse_action("Action: click[Buy Now]\nAction: broken")

    def test_judge_requires_explicit_key(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            QwenJudge()

    def test_metrics_require_ground_truth(self):
        rows = [{"success": True, "duration_seconds": 2,
                 "events": [{"rule_valid": True, "grounding": 1}]}]
        self.assertIsNone(summarize(rows)["grounding_accuracy"])
        rows[0]["events"][0]["human_valid"] = False
        self.assertEqual(summarize(rows)["grounding_accuracy"], 0)


if __name__ == "__main__":
    unittest.main()
