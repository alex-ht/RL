# On-policy Distillation

We provide an example on-policy distillation experiment using the [DeepScaler dataset](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview).

> [!NOTE]
> Distillation currently supports the DTensor and vLLM generation backend. Megatron generation/training paths are not supported yet.

## On-policy Distillation Single Node

To run on-policy distillation on a single GPU using `Qwen/Qwen3-1.7B-Base` as the student and `Qwen/Qwen3-4B` as the teacher:

```sh
uv run python examples/run_distillation.py
```

Customize parameters with command-line overrides. For example:

```sh
uv run python examples/run_distillation.py \
  policy.model_name="Qwen/Qwen3-1.7B-Base" \
  teacher.model_name="Qwen/Qwen3-4B" \
  cluster.gpus_per_node=8
```

### On-policy Distillation with NeMo Gym

On-policy distillation can use NeMo Gym for multi-step or multi-turn rollout collection. In this mode, NeMo RL exposes the student vLLM generation worker as an OpenAI-compatible HTTP server, NeMo Gym runs the environment interaction, and the resulting student samples are used for teacher-logit distillation.

Use the NeMo Gym distillation entrypoint with the example config. The checked-in config uses placeholder dataset paths, so override them for your local data:

```sh
uv run python examples/nemo_gym/run_distillation_nemo_gym.py \
  --config examples/nemo_gym/distillation_qwen3_0_6b.yaml \
  data.train.data_path=/path/to/train.jsonl \
  data.validation.data_path=/path/to/validation.jsonl
```

The config must enable the vLLM async HTTP server and NeMo Gym:

```yaml
policy:
  generation:
    backend: vllm
    vllm_cfg:
      async_engine: true
      expose_http_server: true

env:
  should_use_nemo_gym: true
```

NeMo Gym controls the rollout turn count from its environment and agent configuration. The standard distillation `distillation.max_rollout_turns` setting is not used by the NeMo Gym rollout path.

## On-policy Distillation Multi-node

```sh
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=2

COMMAND="uv run ./examples/run_distillation.py --config examples/configs/distillation_math.yaml cluster.num_nodes=2 cluster.gpus_per_node=8 checkpointing.checkpoint_dir='results/distill_2nodes' logger.wandb_enabled=True logger.wandb.name='distill-2nodes'" \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=4:0:0 \
    --gres=gpu:8 \
    ray.sub
```

> [!NOTE]
> For GB200 systems with 4 GPUs per node, use `--gres=gpu:4` instead.

## Union top-k reverse KL

`loss_fn.topk_mode` selects the support of the distillation KL. `k` is `distillation.topk_logits_k`.

- `teacher` (default) is the historical estimator: KL on the teacher's top-k only.
- `union` is the MiniCPM5-1B estimator: take the top-k token ids from the student and from the teacher, and compute reverse KL on the union of those two sets. It requires `kl_type: reverse` and `zero_outside_topk: false`.

Renormalization: both sides are `log_softmax`'d on the union only. Padded slots are `-inf` and drop out of the normalizer, so the KL is between two distributions supported on the union. That matches the historical teacher-top-k path with `zero_outside_topk: false`. It is not full-vocabulary probabilities restricted to the union. Full-vocabulary mass outside the union is discarded on both sides. `zero_outside_topk: true` is rejected in union mode because that tail correction is defined for the complement of the teacher top-k, not the union.

The student top-k pass runs first (eval, no grad). Those indices are sent to every teacher as `extra_indices`, and the teacher returns logits at its own top-k and at the student indices. Overlapping ids keep the teacher logit. The union is detached. The student training forward then gathers, with grad, at the union indices.

Union mode does not support context parallelism or sequence packing. The student top-k is computed in eval mode; if dropout is enabled, that set can differ from the training forward. Gemma attention dropout is typically 0.

## Gemma-4 final logit softcapping

Gemma-4 applies `softcap * tanh(logits / softcap)` with `softcap=30` inside the model forward (Automodel and Megatron). The map is strictly increasing, so top-k membership is unchanged, and applying it to the gathered union logits equals gathering after the full-vocabulary softcap.

Leave `loss_fn.final_logit_softcapping` unset (`null`) when the forward already softcaps. Set it to `30` only when the backend returns raw LM-head logits. Setting it while the forward also softcaps applies the cap twice.

## Resident multi-teacher distillation

`teacher:` remains the single teacher colocated on the training cluster. It is offloaded after each top-k pass. When vLLM is non-colocated, the student optimizer is offloaded before that teacher forward.

`teachers:` replaces it with up to four resident teachers:

```yaml
teacher: null
teachers:
  math:
    model_name: /path/to/math-expert
    task_names: ["DAPOMath17K"]
    resources:
      num_nodes: 1
      gpus_per_node: 4
    automodel_cfg:
      expert_parallel_size: 4
      tensor_parallel_size: 1
      context_parallel_size: 1
```

Each alias gets its own Ray cluster, subtracted from the training node budget the same way non-colocated vLLM nodes are. Teachers that fit on one node use a STRICT_PACK placement group so expert parallelism stays on one node. Partial-node teachers cannot be combined with `cluster.segment_size`. Samples are routed by `DatumSpec.task_name`. Teachers run concurrently (`submit_topk_logits` / `collect_topk_logits`); they cannot overlap the student top-k pass because union mode needs those indices first. Resident teachers are not offloaded between steps, and the student optimizer stays on the training GPUs.

Logged metrics include `kl/<alias>`, `union_size`, and `teacher_time/<alias>`.

The 8-node Gemma-4 26B-A4B recipe (student EP32 on 4 nodes, four EP4 teachers on 2 nodes, vLLM on 2 nodes) is `examples/configs/recipes/llm/distillation-gemma4-26ba4b-it-8n8g-automodel-ep32-multiteacher.yaml`. This distillation objective has no sampled-token importance ratio. IcePop remains the mismatch correction to use if a GRPO or DAPO term is added later; it is not a field of the distillation loss.
