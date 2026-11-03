#!/usr/bin/env bash
# Build + (optionally) push alexht/nemo-rl:pinchbench-grpo-<rl-shortsha> and :pinchbench-grpo-latest.
# Usage: docker/pinchbench-grpo/build_and_push.sh [--push]
# Long build? run with nohup and a log under ~/Workspace/ (see repo notes).
#
# Memory knobs (env vars):
#   BUILD_MAX_JOBS   max parallel compile jobs inside the build (default 2)
#   BUILD_RAY_CPUS   CPUs given to the build-time Ray head for the Gym venv prefetch (default 2)
#   BUILDER          buildx builder to use, e.g. BUILDER=lowmem (docker-container driver
#                    with a memory cap); default: the docker daemon's own builder
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$here" rev-parse --show-toplevel)"
rl_commit="$(git -C "$repo" rev-parse HEAD)"
short="${rl_commit:0:8}"
image="${IMAGE_REPO:-alexht/nemo-rl}"
base="${BASE_IMAGE:-alexht/nemo-rl:7ef8e83e}"
build_max_jobs="${BUILD_MAX_JOBS:-2}"
build_ray_cpus="${BUILD_RAY_CPUS:-2}"
if ! git -C "$repo" diff --quiet -- docker/pinchbench-grpo; then
  echo "WARN: docker/pinchbench-grpo has uncommitted changes; tag $short will not match the tree" >&2
fi
build_cmd=(docker build)
if [[ -n "${BUILDER:-}" ]]; then
  # docker-container builders keep results in their own cache; --load brings the image back.
  build_cmd=(docker buildx build --builder "$BUILDER" --load)
fi
"${build_cmd[@]}" \
  --build-arg BASE_IMAGE="$base" \
  --build-arg RL_COMMIT="$rl_commit" \
  --build-arg BUILD_MAX_JOBS="$build_max_jobs" \
  --build-arg BUILD_RAY_CPUS="$build_ray_cpus" \
  ${PINCHBENCH_SKILL_COMMIT:+--build-arg PINCHBENCH_SKILL_COMMIT="$PINCHBENCH_SKILL_COMMIT"} \
  -t "$image:pinchbench-grpo-$short" -t "$image:pinchbench-grpo-latest" \
  "$here"
if [[ "${1:-}" == "--push" ]]; then
  docker push "$image:pinchbench-grpo-$short"
  docker push "$image:pinchbench-grpo-latest"
  docker buildx imagetools inspect "$image:pinchbench-grpo-$short" | sed -n '1,3p'
fi
