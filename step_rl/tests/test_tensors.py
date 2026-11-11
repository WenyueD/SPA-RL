import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers")
                     and importlib.util.find_spec("peft") and importlib.util.find_spec("accelerate"),
                     "Training dependencies unavailable")
class TensorTests(unittest.TestCase):
    def test_tiny_qwen3_ppo_backward(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from step_rl.train_ppo import clipped_loss, token_outputs
        torch.manual_seed(7)
        base = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=32,
            intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8))
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4,
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        head = torch.nn.Linear(32, 1)
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            old, old_values, _ = token_outputs(model, head, ids)
        current, values, _ = token_outputs(model, head, ids)
        mask = torch.tensor([False, True, True, True])
        policy, value = clipped_loss(current, old, values, old_values,
            torch.tensor([1., 0., -1.]), torch.ones(3), mask)
        (policy + 0.1 * value).backward()
        self.assertIsNotNone(head.weight.grad)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for name, p in model.named_parameters() if "lora_B" in name))

    def test_clipping_stops_positive_advantage_gradient(self):
        import torch
        from step_rl.train_ppo import clipped_loss
        current = torch.tensor([0.3], requires_grad=True)
        zero = torch.zeros(1)
        policy, _ = clipped_loss(current, zero, zero, zero, torch.ones(1), zero,
                                 torch.tensor([True]))
        policy.backward()
        self.assertEqual(current.grad.item(), 0)

    def test_policy_gradient_and_value_target(self):
        import torch
        from step_rl.train_ppo import clipped_loss
        current = torch.zeros(2, requires_grad=True)
        values = torch.zeros(2, requires_grad=True)
        policy, value = clipped_loss(current, torch.zeros(2), values, torch.zeros(2),
                                    torch.ones(2), torch.ones(2), torch.tensor([True, True]))
        (policy + 0.1 * value).backward()
        self.assertTrue(torch.allclose(current.grad, torch.tensor([-0.5, -0.5])))
        self.assertTrue(torch.allclose(values.grad, torch.tensor([-0.05, -0.05])))


if __name__ == "__main__":
    unittest.main()
