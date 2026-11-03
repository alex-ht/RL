#!/usr/bin/env bash
# Build + (optionally) push alexht/nemo-rl:pinchbench-grpo-<rl-shortsha> and :pinchbench-grpo-latest.
# Usage: docker/pinchbench-grpo/build_and_push.sh [--push]
# Long build? run with nohup and a log under ~/Workspace/ (see repo notes).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$here" rev-parse --show-toplevel)"
rl_commit="$(git -C "$repo" rev-parse HEAD)"
short="${rl_commit:0:8}"
image="${IMAGE_REPO:-alexht/nemo-rl}"
base="${BASE_IMAGE:-alexht/nemo-rl:7ef8e83e}"
if ! git -C "$repo" diff --quiet -- docker/pinchbench-grpo; then
  echo "WARN: docker/pinchbench-grpo has uncommitted changes; tag $short will not match the tree" >&2
fi
docker build \
  --build-arg BASE_IMAGE="$base" \
  --build-arg RL_COMMIT="$rl_commit" \
  ${PINCHBENCH_SKILL_COMMIT:+--build-arg PINCHBENCH_SKILL_COMMIT="$PINCHBENCH_SKILL_COMMIT"} \
  -t "$image:pinchbench-grpo-$short" -t "$image:pinchbench-grpo-latest" \
  "$here"
if [[ "${1:-}" == "--push" ]]; then
  docker push "$image:pinchbench-grpo-$short"
  docker push "$image:pinchbench-grpo-latest"
  docker buildx imagetools inspect "$image:pinchbench-grpo-$short" | sed -n '1,3p'
fi
