# syntax=docker/dockerfile:1
# PinchBench-Research GRPO image: NeMo-RL (alex-ht/RL fork) + everything a
# rollout needs to run a PinchBench task as a LOCAL subprocess inside the same
# enroot/Pyxis container (no apptainer/docker on the cluster):
#   Node 22, OpenClaw (pinned), fws (pinned, /tmp patch), PinchBench skill fork
#   (pinned commit, /opt/pinchbench), system-python deps for research tasks,
#   and the Gym pinchbench agent's `sandbox_provider: {local: {}}` run mode.
#
# Layered on the prebuilt NeMo-RL image (torch/vLLM cu130 wheels on a CUDA
# 13.2 cuda-dl-base; needs host driver >= R580 -- R580 runs 13.2 via the
# bundled forward-compat libs in /usr/local/cuda/compat). Nothing here compiles
# CUDA code, so a plain `docker build` is fine (no lowmem builder needed).
#
# Build (context = this directory; see build_and_push.sh):
#   docker build --build-arg RL_COMMIT=$(git rev-parse HEAD) \
#     -t alexht/nemo-rl:pinchbench-grpo-<sha> docker/pinchbench-grpo
ARG BASE_IMAGE=alexht/nemo-rl:7ef8e83e
ARG NODE_IMAGE=node:22-bookworm

FROM ${NODE_IMAGE} AS node

FROM ${BASE_IMAGE}

ARG OPENCLAW_VERSION=2026.5.28
ARG FWS_VERSION=0.4.4
ARG PINCHBENCH_SKILL_REPO=https://github.com/alex-ht/skill.git
ARG PINCHBENCH_SKILL_COMMIT=2d2bbdba8737b142fe1ed608711e68be594abbc6
ARG RL_COMMIT=unknown
ARG BASE_IMAGE

LABEL org.opencontainers.image.source="https://github.com/alex-ht/RL" \
      org.opencontainers.image.description="NeMo-RL + PinchBench/OpenClaw local-subprocess rollouts (GRPO on PinchBench research tasks)" \
      org.opencontainers.image.revision="${RL_COMMIT}" \
      pinchbench.skill.commit="${PINCHBENCH_SKILL_COMMIT}" \
      pinchbench.openclaw.version="${OPENCLAW_VERSION}" \
      pinchbench.fws.version="${FWS_VERSION}" \
      pinchbench.base.image="${BASE_IMAGE}"

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# System tools the agent's exec tool and the task graders use.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git jq procps openssl sqlite3 poppler-utils \
        unzip zip file python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Node.js 22 runtime (copied from the official image).
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=node /usr/local/include/node /usr/local/include/node
RUN ln -sf ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -sf ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && node --version && npm --version

# OpenClaw + fws, pinned. fws' MITM proxy hard-codes /tmp: patch it to $TMPDIR.
RUN npm install -g --no-fund --no-audit "openclaw@${OPENCLAW_VERSION}" "@juppytt/fws@${FWS_VERSION}" \
    && npm cache clean --force \
    && openclaw --version | grep -q "${OPENCLAW_VERSION}" \
    && fws --version

# PinchBench skill fork at a pinned commit (read-only at runtime; rollouts
# run benchmark.py in place, all writes go to the per-rollout dir).
RUN git clone --filter=blob:none "${PINCHBENCH_SKILL_REPO}" /opt/pinchbench \
    && git -C /opt/pinchbench checkout --detach "${PINCHBENCH_SKILL_COMMIT}" \
    && git -C /opt/pinchbench rev-parse HEAD > /opt/pinchbench/.pinchbench_commit \
    && rm -rf /opt/pinchbench/.git \
    && python3 /opt/pinchbench/scripts/docker/patch_fws_tmpdir.py \
    && install -m 0755 /opt/pinchbench/scripts/pinchbench_env.sh /usr/local/bin/pinchbench_env.sh \
    && chmod -R a+rX /opt/pinchbench

# System python (the one OpenClaw's exec tool and benchmark.py use) deps:
# benchmark.py needs pyyaml; research-task agents commonly use requests/bs4/
# lxml/pandas; graders may run pytest. Kept out of the NeMo-RL venvs.
RUN uv pip install --system --break-system-packages --python /usr/bin/python3 --no-cache \
        pyyaml requests beautifulsoup4 lxml pandas numpy matplotlib python-dateutil \
        tabulate openpyxl pypdf pdfplumber pytest \
    && /usr/bin/python3 -c "import yaml, requests, bs4, lxml, pandas, pypdf, pdfplumber, pytest; print('task deps OK')"

# Gym pinchbench agent with the local (container-less) run mode. The Gym
# submodule (NVIDIA-NeMo/Gym @ da9f6714) is not a fork we can push, so the
# change is carried in this directory as a patch and applied here.
COPY . /opt/pinchbench-grpo/
RUN cd /opt/nemo-rl/3rdparty/Gym-workspace/Gym \
    && git apply --verbose /opt/pinchbench-grpo/gym-pinchbench-local.patch
RUN chmod 0755 /opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/pinchbench/run_task_local.sh \
    && chmod -R a+rX /opt/pinchbench-grpo /opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/pinchbench \
    && printf 'rl_commit=%s\nskill_commit=%s\nopenclaw=%s\nfws=%s\nbase=%s\n' \
        "${RL_COMMIT}" "${PINCHBENCH_SKILL_COMMIT}" "${OPENCLAW_VERSION}" "${FWS_VERSION}" "${BASE_IMAGE}" \
        > /opt/pinchbench-grpo/BUILD_INFO

# Bake the Gym server venvs (vllm_model proxy + pinchbench agent) into
# /opt/gym_venvs: under enroot the rootfs is read-only, so Gym cannot create
# them at runtime. Same mechanism as docker/Dockerfile NEMO_GYM_PREFETCH_CONFIGS.
RUN cd /opt/nemo-rl \
    && UV_LINK_MODE=symlink uv run --frozen python examples/nemo_gym/prefetch_venvs.py \
        /opt/pinchbench-grpo/prefetch_pinchbench_envs.yaml 2>&1 | tee /opt/pinchbench-grpo/prefetch.log \
    && grep -q "Succeeded: 1" /opt/pinchbench-grpo/prefetch.log \
    && ls -la "${NEMO_GYM_VENV_DIR}" \
    && find /opt/gym_venvs -name ray_dist.jar -delete \
    && chmod -R a+rX /opt/gym_venvs

ENV PINCHBENCH_SKILL_DIR=/opt/pinchbench \
    PINCHBENCH_PYTHON=/usr/bin/python3 \
    PINCHBENCH_IMAGE=1 \
    NPM_CONFIG_UPDATE_NOTIFIER=false \
    NO_UPDATE_NOTIFIER=1 \
    OPENCLAW_NO_UPDATE_CHECK=1
