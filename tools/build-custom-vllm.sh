#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -eou pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath "$SCRIPT_DIR/..")"


# Parse command line arguments
GIT_URL=${1:-https://github.com/vllm-project/vllm.git}
GIT_REF=${2:-v0.20.0}
# If a third argument (explicit precompiled wheel URL) is provided, export VLLM_PRECOMPILED_WHEEL_LOCATION
# to instruct vLLM's build to download and use that wheel's prebuilt binaries instead of compiling
# C++/CUDA extensions from the cloned source. This is the fast path and is the historical default
# behavior when using published wheels for a commit (including the v0.20.0 release wheels).
#
# If the third argument is omitted/empty, do NOT set (or export) VLLM_PRECOMPILED_WHEEL_LOCATION.
# The subsequent `uv pip install -e .` inside the vllm tree will perform a full from-source
# compilation. Use this when your custom fork has C++/kernel changes, or no matching prebuilt
# wheel is available for the GIT_REF.
if [[ -n "${3:-}" ]]; then
  VLLM_PRECOMPILED_WHEEL_LOCATION="$3"
  export VLLM_PRECOMPILED_WHEEL_LOCATION
else
  unset VLLM_PRECOMPILED_WHEEL_LOCATION || true
fi

BUILD_DIR=$(realpath "$SCRIPT_DIR/../3rdparty/vllm")
if [[ -e "$BUILD_DIR" ]]; then
  echo "[ERROR] $BUILD_DIR already exists. Please remove or move it before running this script."
  exit 1 
fi

echo "Building vLLM from:"
echo "  Vllm Git URL: $GIT_URL"
echo "  Vllm Git ref: $GIT_REF"
if [[ -n "${VLLM_PRECOMPILED_WHEEL_LOCATION:-}" ]]; then
  echo "  Vllm precompiled wheel: $VLLM_PRECOMPILED_WHEEL_LOCATION"
else
  echo "  Vllm precompiled wheel: <none - will compile from source>"
fi

# Clone the repository
echo "Cloning repository..."
# When running inside Docker with --mount=type=ssh, the known_hosts file is empty.
# Skip host key verification for internal builds (only applies to SSH URLs).
GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" git clone "$GIT_URL" "$BUILD_DIR"
cd "$BUILD_DIR"
git checkout "$GIT_REF"

# Create a new Python environment using uv
echo "Creating Python environment..."
# Pop the project environment set by user to not interfere with the one we create for the vllm repo
OLD_UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT
unset UV_PROJECT_ENVIRONMENT
uv venv

# Remove all comments from requirements files to prevent use_existing_torch.py from incorrectly removing xformers
# (even though v0.20+ vllm reqs no longer list xformers, the comment stripping is still needed for any
# torch mentions in comments that would cause the stripper to drop unrelated lines).
echo "Removing comments from requirements files..."
find requirements/ -name "*.txt" -type f -exec sed -i 's/#.*$//' {} \; 2>/dev/null || true
find requirements/ -name "*.txt" -type f -exec sed -i '/^[[:space:]]*$/d' {} \; 2>/dev/null || true
# Replace xformers==.* (but preserve any platform markers at the end) — no-op on v0.20+ but kept for
# forks based on older vllm trees that still declare xformers in requirements.
# NOTE: xformers pin chosen for compatibility with torch 2.11 + cu13 (and TE / flash-attn used by NeMo-RL).
# Update the version here if a newer xformers is required for your custom vllm fork + torch combo.
find requirements/ -name "*.txt" -type f -exec sed -i -E 's/^(xformers)==[^;[:space:]]*/\1==0.0.32.post1/' {} \; 2>/dev/null || true

uv run --no-project use_existing_torch.py

# Install dependencies
echo "Installing dependencies..."
uv pip install --upgrade pip
uv pip install numpy setuptools setuptools_scm
# Use torch 2.11 + cu130 (CUDA 13) to match the rest of NeMo-RL (pyproject.toml, uv.lock, Docker base images
# cuda-dl-base:...-cuda13, transformer-engine core_cu13, flash-attn cu13torch wheels, etc.).
# Do NOT use cu129 / torch 2.10 here — that would produce a vllm built against CUDA 12 torch, causing
# runtime mismatches (libcudart.so.12 vs cu13, symbol errors, etc.) in the final image / venv.
uv pip install torch==2.11.0 --torch-backend=cu130

# Install vLLM: use precompiled wheel (fast, skips compilation) if provided; otherwise compile from the (custom) source.
if [[ -n "${VLLM_PRECOMPILED_WHEEL_LOCATION:-}" ]]; then
  echo "Installing vLLM using precompiled wheel (skips C++/CUDA compilation)..."
else
  echo "Installing vLLM by compiling from source (fresh build)..."
fi
uv pip install --no-build-isolation -e .

echo "Build completed successfully!"
echo "The built vLLM is available in: $BUILD_DIR"

echo "Updating repo pyproject.toml to point vLLM to local clone..."

PYPROJECT_TOML="$REPO_ROOT/pyproject.toml"
if [[ ! -f "$PYPROJECT_TOML" ]]; then
  echo "[ERROR] pyproject.toml not found at $PYPROJECT_TOML. This script must be run from the repo root and pyproject.toml must exist."
  exit 1
fi

cd "$REPO_ROOT"

export UV_PROJECT_ENVIRONMENT=$OLD_UV_PROJECT_ENVIRONMENT
if [[ -n "$UV_PROJECT_ENVIRONMENT" ]]; then
    # We optionally set this if the project environment is outside of the project directory.
    # If we do not set this then uv pip install commands will fail
    export VIRTUAL_ENV=$UV_PROJECT_ENVIRONMENT
fi
# Use tomlkit via uv to idempotently update pyproject.toml
uv run --no-project --with tomlkit python - <<'PY'
from pathlib import Path
from tomlkit import parse, dumps, inline_table

pyproject_path = Path("pyproject.toml")
text = pyproject_path.read_text()
doc = parse(text)

# 1) Ensure setuptools_scm in [project].dependencies
project = doc.get("project")
if project is None:
    raise SystemExit("[ERROR] Missing [project] in pyproject.toml")

deps = project.get("dependencies")

if not any(x.startswith("setuptools_scm") for x in deps):
    deps.append("setuptools_scm")

# 2) Update [project.optional-dependencies].vllm: unpin vllm==... -> vllm
opt = project.get("optional-dependencies")
vllm_list = opt["vllm"]
# Remove any pinned vllm==...
keep_items = []
has_unpinned_vllm = False
for item in vllm_list:
    s = str(item).strip()
    if s.startswith("vllm=="):
        continue
    if s == "vllm":
        has_unpinned_vllm = True
    keep_items.append(item)
if not has_unpinned_vllm:
    keep_items.append("vllm")
vllm_list.clear()
for it in keep_items:
    vllm_list.append(it)

# 3) Add [tool.uv.sources].vllm = { path = "3rdparty/vllm", editable = true }
tool = doc.setdefault("tool", {})
uv = tool.setdefault("uv", {})
sources = uv.setdefault("sources", {})
desired = inline_table()
desired.update({"path": "3rdparty/vllm", "editable": True})
sources["vllm"] = desired

# 4) Ensure [tool.uv].no-build-isolation-package includes "vllm"
nbip = uv.setdefault("no-build-isolation-package", [])
nbip_strs = [str(x) for x in nbip]
if "vllm" not in nbip_strs:
    nbip.append("vllm")

pyproject_path.write_text(dumps(doc))
print("[INFO] Updated pyproject.toml for local vLLM.")
PY

# Ensure build deps and re-lock
uv pip install setuptools_scm
uv lock

# Write to a file that a docker build will use to set the necessary env vars.
# Only include VLLM_PRECOMPILED_WHEEL_LOCATION when a precompiled wheel was used (i.e. 3rd arg provided).
cat >"$BUILD_DIR/nemo-rl.env" <<EOF
export VLLM_GIT_REF=$GIT_REF
EOF
if [[ -n "${VLLM_PRECOMPILED_WHEEL_LOCATION:-}" ]]; then
  cat >>"$BUILD_DIR/nemo-rl.env" <<EOF
export VLLM_PRECOMPILED_WHEEL_LOCATION=$VLLM_PRECOMPILED_WHEEL_LOCATION
EOF
fi

cat <<EOF
[INFO] pyproject.toml updated. NeMo RL is now configured to use the local vLLM at 3rdparty/vllm.
[INFO] Verify this new vllm version by running:
EOF

if [[ -n "${VLLM_PRECOMPILED_WHEEL_LOCATION:-}" ]]; then
  cat <<EOF
VLLM_PRECOMPILED_WHEEL_LOCATION=$VLLM_PRECOMPILED_WHEEL_LOCATION \\
  uv run --extra vllm vllm serve Qwen/Qwen3-0.6B
EOF
else
  cat <<'EOF'
# (no VLLM_PRECOMPILED_WHEEL_LOCATION set — this custom vLLM will compile from the local 3rdparty/vllm source)
uv run --extra vllm vllm serve Qwen/Qwen3-0.6B
EOF
fi

cat <<EOF
[INFO] For more information on this custom install, visit https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/use-custom-vllm.md
EOF

if [[ -n "${VLLM_PRECOMPILED_WHEEL_LOCATION:-}" ]]; then
  cat <<EOF
[IMPORTANT] Remember to set the shell variable 'VLLM_PRECOMPILED_WHEEL_LOCATION' when running NeMo RL apps with this custom vLLM to avoid re-compiling.
EOF
else
  cat <<EOF
[IMPORTANT] This vLLM was built from source with no precompiled wheel. Ray worker venvs and rebuilds (NRL_FORCE_REBUILD_VENVS) will compile C++/CUDA kernels from 3rdparty/vllm at install time. This is slower than the precompiled-wheel path; pass a wheel URL as the 3rd arg to build-custom-vllm.sh if a matching prebuilt is available.
EOF
fi
