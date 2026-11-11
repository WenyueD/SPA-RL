# Step-RL / Qwen3-8B

This entry point uses the existing **text WebShop simulator**. Available operations
are search and click (including Buy Now). The environment does not provide a
mobile screen, real-device control, or a separate cart API.

Run commands from the repository root in a new Linux Python 3.10/3.11 environment. Qwen3
requires Transformers >=4.51.0. Install a matching CUDA PyTorch build, then:

```bash
pip install -r step_rl/requirements.txt
pip install --no-deps -e envs/webshop
python -m spacy download en_core_web_lg
```

The --no-deps flag avoids the simulator setup.py's legacy Torch-era dependency
pins overriding this environment. Install Java 11 and prepare the WebShop products
and search index using the parent README. The old
Llama scripts and dependencies remain available. The new PPO implementation uses
PyTorch and Accelerate directly, avoiding the old TRL/Transformers dependency pair.

## SFT baseline

```bash
accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 \
  -m step_rl.train_sft --model Qwen/Qwen3-8B \
  --data data/webshop_sft.json --output ckpt/qwen3_webshop_sft
python -m step_rl.merge --model Qwen/Qwen3-8B \
  --adapter ckpt/qwen3_webshop_sft --output ckpt/qwen3_webshop_sft_merged
```

LoRA targets q_proj, v_proj, gate_proj and up_proj, with rank 8 and alpha 16.
SFT supervises assistant text only. Complete action turns are retained up to the
context budget. Native Qwen3 turn templates disable built-in thinking while
preserving the dataset's explicit Thought/Action protocol. Prior sampled token
prefixes are retained exactly during online interaction.

## Exploration and progress attribution

```bash
python -m step_rl.train_ppo --mode collect --model ckpt/qwen3_webshop_sft_merged \
  --episodes 1000 --output records/qwen3_exploration
python -m step_rl.progress --model ckpt/qwen3_webshop_sft_merged \
  --data records/qwen3_exploration/trajectories-rank0.json \
  --output ckpt/qwen3_progress.pt
```

The estimator freezes the merged SFT backbone. Its hidden states at action ends
feed a hidden_size→128→1 MLP. MSE fits the sum of step contributions to the final
environment score. No manual step labels are assumed. WebShop returns a graded
score, so the target is not restricted to binary success/failure. Existing
exploration JSON is also accepted. Truncated trajectories are excluded when the
final return cannot be aligned to a full trajectory.

## Online PPO

```bash
accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 \
  -m step_rl.train_ppo --model ckpt/qwen3_webshop_sft_merged \
  --progress ckpt/qwen3_progress.pt --episodes 1000 \
  --progress-weight 1.0 --grounding-weight 0.5 --output ckpt/qwen3_step_rl
```

Each rank collects one fresh trajectory per update, then performs four PPO
epochs. Rewards combine learned progress contributions and grounding scores.
Step scores are placed at the corresponding last sampled token. Token KL against
the frozen SFT reference shapes rewards. GAE skips environment/prompt tokens and
discounts once per assistant token. The finite episode horizon (including context
or step limits) is terminal. Advantages, old log probabilities and returns stay
fixed within the update. The loss is clipped policy loss + 0.1×clipped value loss.
Gradient clipping is 1.0. LoRA and value parameters use FP32 optimizer storage.
The SFT reference and progress backbone share frozen weights with the policy;
disabling the PPO adapter recovers the SFT backbone. There is no separate 8B
reference copy on each GPU. This is data parallel training, not model sharding.

Rules check exact action syntax, available targets, search availability and
repeated state/action pairs before environment execution. No-change transitions
receive a negative grounding score. Optional Qwen judgement additionally rejects
actions judged ungrounded or irrelevant:

```bash
export QWEN_API_KEY=your_key
export QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export QWEN_JUDGE_MODEL=qwen-plus
# Append --judge to collection, evaluation or training commands.
```

This option sends the task, current observation and proposed action to the
configured API. API/schema failures stop the run rather than fabricate scores.
It incurs service cost. No API request is made unless --judge is specified.

## Evaluation

```bash
python -m step_rl.train_ppo --mode eval --model ckpt/qwen3_webshop_sft_merged \
  --indices eval_agent/data/webshop/test_indices.json --episodes 500 --output records/baseline
python -m step_rl.merge --model ckpt/qwen3_webshop_sft_merged \
  --adapter ckpt/qwen3_step_rl/adapter --output ckpt/qwen3_step_rl_merged
python -m step_rl.train_ppo --mode eval --model ckpt/qwen3_step_rl_merged \
  --indices eval_agent/data/webshop/test_indices.json --episodes 500 --output records/step_rl
python -m step_rl.metrics records/baseline
python -m step_rl.metrics records/step_rl
```

Use identical indices, episode budgets, hardware, max lengths and judge settings
for both evaluations. Success means environment completion **and score >=1**.
Elapsed time includes model generation, grounding/API calls and environment
execution, excluding model/environment initialization. Rule validity is reported
separately from grounding accuracy. Grounding accuracy is null unless events
carry independent boolean `human_valid` labels.

The 8-GPU launch command is a configuration, not evidence of a completed 8×4090
run. Memory/throughput must be measured on the target server. No 85%→90%, 25s→18s
or 96.2% result has been established by this code change. Raw rank-local JSON
preserves the observations, generated actions, rule/judge decisions and timings.
Checkpoints export adapters and value heads; exact optimizer/RNG resume is not
implemented. Choose a new output directory for each run.

## Local checks

```bash
python -m unittest discover -s step_rl/tests -v
```

Core data/rule/metric checks run without GPU dependencies. Tensor/PPO regression
checks require the training requirements. Full model training and multi-GPU
smoke tests require the target CUDA server and local WebShop assets.
