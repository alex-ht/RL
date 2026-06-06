# syntax=docker/dockerfile:1.7-labs
# Usage:
#   Self-contained build (default: builds from main):
#     docker buildx build -f docker/Dockerfile --tag <registry>/nemo-rl:latest --push .
#
#   Self-contained build (specific git ref):
#     docker buildx build -f docker/Dockerfile --build-arg NRL_GIT_REF=r0.6.0 --tag <registry>/nemo-rl:r0.6.0 --push .
#
#   Self-contained build (remote NeMo RL source; no need for a local clone of NeMo RL):
#     docker buildx build -f docker/Dockerfile --build-arg NRL_GIT_REF=r0.6.0 --tag <registry>/nemo-rl:r0.6.0 --push https://github.com/NVIDIA-NeMo/RL.git
#
#   Local NeMo RL source override:
#     docker buildx build --build-context nemo-rl=. -f docker/Dockerfile --tag <registry>/nemo-rl:latest --push .
#
# Optional build args to skip vLLM or SGLang dependencies:
#   --build-arg SKIP_VLLM_BUILD=1    # Skip vLLM dependencies
#   --build-arg SKIP_SGLANG_BUILD=1  # Skip SGLang dependencies

ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:26.03-cuda13.2-devel-ubuntu24.04

# === Global user configuration (for early non-root switch across stages) ===
# Defined here so hermetic and release stages can use ${USERNAME} in USER and --chown
# without re-declaring ARG in every stage (fixes "invalid user index: -1").
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=1000

ARG UV_VERSION=0.11.6
ARG PYTHON_VERSION=3.13.13

FROM scratch AS nemo-rl
ARG NRL_GIT_REF=main
ADD --keep-git-dir=true https://github.com/NVIDIA-NeMo/RL.git#${NRL_GIT_REF} /

# Empty default; CI overrides with --build-context to inject pre-compiled wheels
FROM scratch AS uv-cache-seed

FROM ${BASE_IMAGE} AS base
# An environment variable to indicate that we are in a container.
ENV NRL_CONTAINER=1

# It is more convenient for users to run as root
USER root

RUN <<EOF bash -exu -o pipefail
export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles

apt-get update
apt-get install -y --no-install-recommends \
    jq \
    curl \
    git \
    rsync \
    wget \
    less \
    vim \
    libz3-4

# Nsight
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/$(uname -m | sed 's/aarch64/sbsa/')/7fa2af80.pub
apt update
apt install -y nsight-systems-cli

# To fix CVE-2025-68973
apt install -y --only-upgrade gnupg

apt-get clean
cp /usr/lib/x86_64-linux-gnu/libz3.so.4 /usr/lib/x86_64-linux-gnu/libz3.so.4.15
rm /usr/lib/x86_64-linux-gnu/libz3.so.4
ln -s /usr/lib/x86_64-linux-gnu/libz3.so.4.15 /usr/lib/x86_64-linux-gnu/libz3.so.4
ldconfig
rm -rf /var/lib/apt/lists/*
EOF

# CMake (for sglang build)
RUN GITHUB_ARTIFACTORY=github.com \
    && CMAKE_VERSION=3.31.1 \
    && ARCH=$(uname -m) \
    && CMAKE_INSTALLER="cmake-${CMAKE_VERSION}-linux-${ARCH}" \
    && curl --retry 3 --retry-delay 2 -fsSL -o "${CMAKE_INSTALLER}.tar.gz" \
        "https://${GITHUB_ARTIFACTORY}/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_INSTALLER}.tar.gz" \
    && tar -xzf "${CMAKE_INSTALLER}.tar.gz" \
    && cp -r "${CMAKE_INSTALLER}/bin/"* /usr/local/bin/ \
    && cp -r "${CMAKE_INSTALLER}/share/"* /usr/local/share/ \
    && rm -rf "${CMAKE_INSTALLER}" "${CMAKE_INSTALLER}.tar.gz"

# ============================================================
# uv 安裝 - 使用官方推薦的 multi-stage 方式（最乾淨方案）
# ============================================================
# 這是解決 NVIDIA base image + /etc/bash.bashrc 污染的根本解法
# 不再使用 curl install.sh + 自訂 /opt/uv 路徑的 hack

# 從官方 uv 映像直接 COPY 二進位檔到標準 PATH
# 這是 astral.sh 官方推薦的 Docker 安裝方式
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-provider

FROM base AS base-with-uv

# 從 uv-provider 複製 uv 二進位（位於 /uv）
COPY --from=uv-provider /uv /usr/local/bin/uv

# 安裝指定 Python 版本（uv 現在已在標準 PATH）
RUN uv python install ${PYTHON_VERSION} && \
    mkdir -p /opt/uv-cache && chmod 777 /opt/uv-cache

# 重要：讓 base image 不要再污染 PATH 和 bashrc
ENV PS1=""

# ============================================================
# 以下為原本的 base 內容（已移除舊的 uv 安裝邏輯）
# ============================================================

# Disable usage stats by default for users who are sensitive to sharing usage.
# Users are encouraged to enable if the wish.
ENV RAY_USAGE_STATS_ENABLED=0
# After ray>=2.47, this feature is enabled by default which creates uv venvs for any py_executable starting with `uv run`.
# There is severe contention and performance issues with this enabled considering our dependencies are so large and occasionally
# need to be compiled, so NeMo RL has an implementation in nemo_rl/utils/venv.py that does it once per node as opposed to once per task.
ENV RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
ENV NEMO_RL_VENV_DIR=/opt/ray_venvs

# --- Phase 1 (user1000-migration-plan): Use existing ubuntu user from NVIDIA base image ---
# The base image (nvcr.io/nvidia/cuda-dl-base) already provides an "ubuntu" user at UID/GID 1000.
# We directly switch to it in hermetic/release stages instead of creating a new user.
# This avoids GID collision and keeps compatibility with common cloud Jupyter setups.
# All system operations in base remain root. uv/venv/compile work will run as ubuntu user later.
# Pre-create + chown directories that will be written by the ubuntu user.
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=1000

# Only prepare directories and set ownership. Do NOT create user (it already exists).
RUN mkdir -p /home/ubuntu/.cache/uv /opt/uv-cache /opt/nemo_rl_venv /opt/ray_venvs /opt/nemo-rl && \
    chown -R ubuntu:ubuntu /home/ubuntu /opt/uv-cache /opt/nemo_rl_venv /opt/ray_venvs /opt/nemo-rl && \
    chmod 755 /opt/nemo_rl_venv /opt/ray_venvs /opt/nemo-rl

# HOME will take effect after USER ubuntu in hermetic/release stages.
# This ensures uv, pip, ray, etc. write caches to /home/ubuntu instead of /root.
ENV HOME=/home/ubuntu


FROM base-with-uv AS hermetic

# Re-declare user ARGs in this stage so USER and --chown in COPY can resolve them.
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=1000

# Phase 2 (user1000-migration-plan): Switch to the existing "ubuntu" user immediately.
# The user already exists in the base image, so no creation is needed.
# All subsequent COPY use numeric chown for reliability.
USER ${USERNAME}

# Set HOME and UV_CACHE_DIR so uv, custom builds (vLLM/FlashInfer/TE),
# ray venvs, and all Python tooling write to /home/ubuntu instead of /root.
ENV HOME=/home/ubuntu
ENV UV_CACHE_DIR=/home/ubuntu/.cache/uv

# Ensure uv cache directory exists in this stage (base already creates it, but defensive).
# This RUN now executes as non-root (dir was pre-chowned 777 in base).
RUN mkdir -p /opt/uv-cache && chmod 777 /opt/uv-cache

WORKDIR /opt/nemo-rl

# Variables to control the build of TE. If there are issues with parallelization, consider
# setting these to 1.
ARG MAX_JOBS
ARG NVTE_BUILD_THREADS_PER_JOB
# Only use for custom vllm installs. Learn more at https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/use-custom-vllm.md
ARG BUILD_CUSTOM_VLLM
ARG BUILD_CUSTOM_VLLM_URL
ARG BUILD_CUSTOM_VLLM_REF
ARG BUILD_CUSTOM_VLLM_PRECOMPILED_WHEEL_LOCATION
# Only use for custom flashinfer installs.
ARG BUILD_CUSTOM_FLASHINFER
ARG BUILD_CUSTOM_FLASHINFER_URL
ARG BUILD_CUSTOM_FLASHINFER_REF
# Skip building vLLM or SGLang dependencies (set to any non-empty value to skip)
ARG SKIP_VLLM_BUILD
ARG SKIP_SGLANG_BUILD
ARG BASE_IMAGE
ARG UV_VERSION

ENV UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
ENV UV_LINK_MODE=copy

# Ensure DeepEP is built for H100 and B200 (also mcore inference unified memory API now invokes a torch API that requires these to be set)
ENV TORCH_CUDA_ARCH_LIST="9.0;10.0"
# NVTE_CUDA_ARCHS uses SM numbers (90 for 9.0) ; different format from TORCH_CUDA_ARCH_LIST.
# Required for Transformer Engine git build (see CMakeCUDAArchitecturesValidate).
ENV NVTE_CUDA_ARCHS="90;100"
ENV CMAKE_CUDA_ARCHITECTURES="90;100"
ENV SGL_KERNEL_COMPILE_THREADS=1

# Ensure HybridEP/DeepEP JIT compilation can find nvcc at runtime
ENV CUDA_HOME=/usr/local/cuda

## CUDA 13 moved standard library headers under include/cccl; expose that path so
## builds that include <cuda/std/...> (e.g., deep_gemm) can find them.
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl

# First copy only the dependency files (all must use --chown for Phase 2 non-root USER)
COPY --chown=${USER_UID}:${USER_GID} --from=nemo-rl pyproject.toml uv.lock ./
# Copy in the top level __init__.py/package_info.py since build-custom-vllm.sh needs the nemo_rl package to exist.
COPY --chown=${USER_UID}:${USER_GID} --from=nemo-rl nemo_rl/__init__.py nemo_rl/package_info.py ./nemo_rl/
COPY --chown=${USER_UID}:${USER_GID} --from=nemo-rl tools/build-custom-vllm.sh ./tools/build-custom-vllm.sh
COPY --chown=${USER_UID}:${USER_GID} --from=nemo-rl tools/build-custom-flashinfer.sh ./tools/build-custom-flashinfer.sh
COPY --chown=${USER_UID}:${USER_GID} --from=nemo-rl --link research/ ./research/
COPY --chown=${USER_UID}:${USER_GID} --from=nemo-rl --link 3rdparty/ ./3rdparty/

RUN --mount=type=ssh \
    --mount=type=bind,from=uv-cache-seed,source=.,target=/tmp/uv-cache-seed <<"EOF" bash -exu
# Phase 2 (user1000-migration-plan): Execute EVERYTHING as non-root.
# HOME/UV_CACHE_DIR exports ensure uv, custom build scripts, and all child processes
# (vLLM/FlashInfer/TE compiles, ray, etc.) write caches and artifacts under /home/ubuntu
# instead of /root. All paths below updated; rsync/find/echo now target $HOME cache.
export HOME=/home/${USERNAME}
export UV_CACHE_DIR=${HOME}/.cache/uv
export NVTE_CUDA_ARCHS="90;100"
export SGL_KERNEL_COMPILE_THREADS=1
export CMAKE_CUDA_ARCHITECTURES="90;100"

# Pre-create (defensive; already chowned in base) so seed rsync and uv have target.
mkdir -p ${HOME}/.cache/uv

CACHE_KEY=$(echo "${BASE_IMAGE}-${UV_VERSION}" | md5sum | cut -c1-12)
SEED_KEY=$(cat /tmp/uv-cache-seed/.cache-key 2>/dev/null || echo "")
if [ -n "$SEED_KEY" ] && [ "$SEED_KEY" != "$CACHE_KEY" ]; then
    echo "Cache key mismatch (base image or uv version changed) — skipping stale seed"
elif [ -n "$SEED_KEY" ]; then
    echo "Seeding uv cache (key=${CACHE_KEY})"
    rsync -a --exclude='.cache-key' /tmp/uv-cache-seed/ ${HOME}/.cache/uv/ 2>/dev/null || true
fi

# 使用 bare `uv`（現在已在 /usr/local/bin，標準 PATH，不再依賴 /opt/uv）
uv venv --seed
if [[ -n "${BUILD_CUSTOM_VLLM:-}" ]]; then
    bash tools/build-custom-vllm.sh ${BUILD_CUSTOM_VLLM_URL} ${BUILD_CUSTOM_VLLM_REF} ${BUILD_CUSTOM_VLLM_PRECOMPILED_WHEEL_LOCATION}
    source 3rdparty/vllm/nemo-rl.env
fi
if [[ -n "${BUILD_CUSTOM_FLASHINFER:-}" ]]; then
    bash tools/build-custom-flashinfer.sh ${BUILD_CUSTOM_FLASHINFER_URL} ${BUILD_CUSTOM_FLASHINFER_REF}
fi
# uv sync has a more reliable resolver than simple uv pip install which can fail

# Sync each training + inference backend one at a time (since they may conflict)
# to warm the uv cache, then at the end just sync the default dependencies.
# Do everything in one layer to prevent large layers.

# The venv is symlinked to avoid bloating the layer size
uv sync --link-mode symlink --locked --no-install-project
bash ./tools/build-apex.sh
if [[ -z "${SKIP_VLLM_BUILD:-}" ]]; then
    uv sync --link-mode symlink --locked --extra vllm --no-install-project
fi
if [[ -z "${SKIP_SGLANG_BUILD:-}" ]]; then
    uv sync --link-mode symlink --locked --extra sglang --no-install-project
fi
uv sync --link-mode symlink --locked --extra mcore --no-install-project
uv sync --link-mode symlink --locked --extra automodel --no-install-project
uv sync --link-mode symlink --locked --extra modelopt --no-install-project
uv sync --link-mode symlink --locked --extra nemo_gym --no-install-project
uv sync --link-mode symlink --locked --all-groups --no-install-project

# Remove the aiohttp in this uv cache dir to fully address CVE GHSA-mqqc-3gqh-h2x8
# The ray install will include the older aiohttp version in its cache
find ${HOME}/.cache/uv -type d -path "*ray/_private/runtime_env/agent/thirdparty_files/aiohttp*" -exec rm -rf {} +

echo "${CACHE_KEY}" > ${HOME}/.cache/uv/.cache-key
EOF

ENV PATH=/opt/nemo_rl_venv/bin:$PATH
ENV NEMO_RL_VENV_DIR=/opt/ray_venvs
# Point TE at the pip-installed cuDNN (nvidia-cudnn-cu12) instead of the system one.
# The container's system cuDNN (/lib/x86_64-linux-gnu/libcudnn*.so.9) may be a different
# version than what pip installed. TE prioritizes system libraries by default, causing a
# version mismatch crash (e.g. system 9.10.1 vs pip 9.19.0). CUDNN_HOME makes TE's Python
# code find the pip version first, and LD_LIBRARY_PATH makes the dynamic linker resolve
# cuDNN sub-libraries from pip when loading libtransformer_engine.so.
ENV CUDNN_HOME=/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn
ENV LD_LIBRARY_PATH="/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
# Verify with: python -c "import transformer_engine.pytorch as te; print(te.get_cudnn_version())"

WORKDIR /opt/nemo-rl

FROM hermetic AS release

# Re-declare user ARGs in this stage.
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=1000

# Phase 3 (user1000-migration-plan): Immediately switch to the existing "ubuntu" user after FROM.
# Root cleanup must run while still root (before USER) because /root is 700.
# Strengthened cleanup (more dirs) + early USER means prefetch/git/python/echo
# all run as ${USERNAME} and cannot pollute /root. Final COPY below gets --chown.

# Root-only cleanup while still root. (Original cleanup moved here and strengthened.)
RUN rm -rf /root/.cache /root/.local /root/.config 2>/dev/null || true; \
    chmod 700 /root || true

USER ${USERNAME}

# Re-declare build args for this stage
ARG SKIP_VLLM_BUILD
ARG SKIP_SGLANG_BUILD
ARG NEMO_RL_COMMIT
ARG NVIDIA_BUILD_ID
ARG NVIDIA_BUILD_REF
ARG RC_DATE=00.00
ARG TARGETARCH
ENV NEMO_RL_COMMIT=${NEMO_RL_COMMIT:-<unknown>}
ENV NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID:-<unknown>}
ENV NVIDIA_BUILD_REF=${NVIDIA_BUILD_REF:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"

ENV NEMO_RL_VENV_DIR=/opt/ray_venvs

# Copy in source from build context (defaults to cloned repo, can be overridden)
# Exclude pyproject.toml and uv.lock since those may be altered by build-custom-vllm.sh
# --chown required so that subsequent non-root RUNs (git fetch, uv run, python, echo) can write.
COPY --chown=${USER_UID}:${USER_GID} --from=nemo-rl --exclude=pyproject.toml --exclude=uv.lock . /opt/nemo-rl
# Unshallow the repo to get the full history (in the case it was from the scratch layer).
# Potentially not necessary if the repo is passed in as a complete repository (w/ full git history),
# so do a quick check before trying to unshallow.
RUN git rev-parse --is-shallow-repository | grep -q true && git fetch --unshallow || true
RUN <<EOF bash -exu
NEGATIVE_FILTERS=""
if [[ -n "${SKIP_VLLM_BUILD:-}" ]]; then
    NEGATIVE_FILTERS="$NEGATIVE_FILTERS vllm"
fi
if [[ -n "${SKIP_SGLANG_BUILD:-}" ]]; then
    NEGATIVE_FILTERS="$NEGATIVE_FILTERS sglang"
fi
if [[ -n "$NEGATIVE_FILTERS" ]]; then
    UV_LINK_MODE=symlink uv run nemo_rl/utils/prefetch_venvs.py --negative-filters $NEGATIVE_FILTERS
else
    UV_LINK_MODE=symlink uv run nemo_rl/utils/prefetch_venvs.py
fi
EOF

# Generate container fingerprint for frozen environment support
# Store outside /opt/nemo-rl to avoid being overwritten by user mounts
RUN python tools/generate_fingerprint.py > /opt/nemo_rl_container_fingerprint

# NOTICES.txt file points to where the OSS source code is archived
RUN echo "This distribution includes open source which is archived at the following URL: https://opensource.nvidia.com/oss/teams/nvidia/nemo-rl/${RC_DATE}:linux-${TARGETARCH}/index.html" > NOTICES.txt && \
    echo "For further inquiries or assistance, contact us at oss-requests@nvidia.com" >> NOTICES.txt

# NOTE: Root cleanup (rm -rf /root/.cache etc + chmod 700 /root) was moved to the
# top of the release stage (Phase 3) and executed while still root, before USER.
# It is intentionally omitted here; all post-USER RUNs run as non-root and do not
# pollute /root thanks to the HOME/UV_CACHE_DIR settings from hermetic.
