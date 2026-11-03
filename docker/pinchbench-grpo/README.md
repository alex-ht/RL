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

Files: `Dockerfile`, `build_and_push.sh [--push]`, `smoke_test.sh` (non-root, read-only rootfs check),
`gym-pinchbench-local.patch`. Regenerate the patch from a checked-out submodule with
`git -C 3rdparty/Gym-workspace/Gym diff --binary > docker/pinchbench-grpo/gym-pinchbench-local.patch`.
