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
GIT_URL=${1:-https://github.com/NVIDIA/apex.git}
GIT_REF=${2:-master}

BUILD_DIR=$(realpath "$SCRIPT_DIR/../3rdparty/apex")
if [[ -e "$BUILD_DIR" ]]; then
  echo "[ERROR] $BUILD_DIR already exists. Please remove or move it before running this script."
  exit 1
fi

echo "Building NVIDIA Apex from:"
echo "  Apex Git URL: $GIT_URL"
echo "  Apex Git ref: $GIT_REF"
echo "  Target: Full C++ + CUDA extensions (APEX_CPP_EXT=1 APEX_CUDA_EXT=1)"

# Clone the repository
echo "Cloning repository..."
# When running inside Docker with --mount=type=ssh, the known_hosts file is empty.
# Skip host key verification for internal builds (only applies to SSH URLs).
GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" git clone "$GIT_URL" "$BUILD_DIR"
cd "$BUILD_DIR"
git checkout "$GIT_REF"

# Ensure we have a Python environment (uv or system)
if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
    echo "Using existing UV project environment: $UV_PROJECT_ENVIRONMENT"
    export VIRTUAL_ENV=$UV_PROJECT_ENVIRONMENT
    export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
elif command -v uv >/dev/null 2>&1; then
    echo "Creating temporary uv venv for Apex build..."
    uv venv --python python3 .venv
    source .venv/bin/activate
else
    echo "No uv found — using system python / pip"
fi

# Install build dependencies
echo "Installing build dependencies (ninja, setuptools, wheel)..."
pip install --upgrade pip setuptools wheel ninja

# Recommended: Use environment variables for full C++/CUDA extensions
# This is the modern, recommended way per apex README (as of 2026)
export APEX_CPP_EXT=1
export APEX_CUDA_EXT=1

# Optional: Build many useful contrib extensions commonly needed by Megatron / NeMo-RL
# Comment out lines below if you want a leaner build
export APEX_FAST_MULTIHEAD_ATTN=1
export APEX_FUSED_CONV_BIAS_RELU=1
export APEX_FAST_LAYER_NORM=1
export APEX_DISTRIBUTED_ADAM=1
export APEX_DISTRIBUTED_LAMB=1

# Parallel build tuning (respect Docker MAX_JOBS if set)
if [[ -n "${MAX_JOBS:-}" ]]; then
    export APEX_PARALLEL_BUILD=$MAX_JOBS
    echo "Using APEX_PARALLEL_BUILD=$MAX_JOBS (from MAX_JOBS)"
else
    export APEX_PARALLEL_BUILD=8
fi

# NVCC threading (helps on high-core machines)
export NVCC_APPEND_FLAGS="--threads 4"

echo "Starting Apex build with full C++/CUDA extensions..."
echo "  APEX_CPP_EXT=$APEX_CPP_EXT"
echo "  APEX_CUDA_EXT=$APEX_CUDA_EXT"
echo "  APEX_PARALLEL_BUILD=$APEX_PARALLEL_BUILD"

# Build and install (no-build-isolation is important for custom CUDA/C++ extensions)
pip install -v --no-build-isolation --no-cache-dir .

echo ""
echo "[SUCCESS] NVIDIA Apex built and installed with C++ and CUDA extensions."
echo "Installed location: $(python -c 'import apex; print(apex.__file__)')"
echo ""
echo "To verify fused kernels (example):"
echo "  python -c \"from apex.normalization import FusedLayerNorm; print('FusedLayerNorm OK')\""
echo "  python -c \"from apex.optimizers import FusedAdam; print('FusedAdam OK')\""
echo ""
echo "If you need even more contrib modules later, re-run with APEX_ALL_CONTRIB_EXT=1"