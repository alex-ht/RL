# pinchbench-grpo image

`alexht/nemo-rl:pinchbench-grpo-<rl-shortsha>` = `alexht/nemo-rl:7ef8e83e` (NeMo-RL, torch/vLLM cu130,
CUDA 13.2 toolkit + forward-compat; host driver >= R580) plus:

- Node 22, OpenClaw 2026.5.28, @juppytt/fws 0.4.4 (+ /tmp -> $TMPDIR patch)
- PinchBench skill fork (alex-ht/skill, pinned commit) at `/opt/pinchbench` (read-only)
- system-python deps for PinchBench research tasks
- `gym-pinchbench-local.patch` applied to the Gym submodule (NVIDIA-NeMo/Gym @ da9f6714, not a pushable fork):
  the pinchbench agent gains `sandbox_provider: {local: {...}}` (each rollout = local subprocess in a private
  per-rollout dir, no apptainer/docker; never /tmp), judge-failure masking (`instance_config.mask_sample`),
  and knobs `judge_fallback_model`, `web_fetch_max_chars`, `sampling_params`, `model_reasoning`, `thinking_level`.
- `gym-lineage-grok.patch` on that same tree: lineage fingerprints drop `<think>...</think>`, and the
  pinchbench agent accepts web search provider `grok` (`xai_api_key` is forwarded as `XAI_API_KEY`).
- `automodel-gemma4-cp-logprob.patch` on Automodel @ 72daceff: SAC ignores `lift_fresh`, Gemma4 CP flex
  masks stay compiled across recompute, and lm_head logprobs are chunked.
- Owned files copied over the base image: `vllm_worker_async.py` (splice EOS from `NRL_SPLICE_EOS_TOKEN_IDS`),
  `automodel_policy_worker.py` (SingleController split train API), `nemo_rl/models/automodel/train.py`
  (consume the chunked logprobs).
- vLLM `chat_utils.py` tool-call ids drop the hyphen (`chatcmpltool...`). `rlpb_inductor_cache.py` is
  installed into the image venvs so each host and pid gets its own Inductor cache.

Files: `Dockerfile`, `build_and_push.sh [--push]`, `smoke_test.sh` (non-root, read-only rootfs check),
the patches above, and `rlpb_inductor_cache.py`. Build context is the repo root. Regenerate the Gym
local patch from a checked-out submodule with
`git -C 3rdparty/Gym-workspace/Gym diff --binary > docker/pinchbench-grpo/gym-pinchbench-local.patch`.
