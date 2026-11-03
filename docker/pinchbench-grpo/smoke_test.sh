#!/usr/bin/env bash
# Non-GPU smoke test for the pinchbench-grpo image. Meant to run as a NON-ROOT
# user on a READ-ONLY rootfs with one writable bind mount, mimicking
# enroot/Pyxis on SLURM, e.g.:
#   mkdir -p w && docker run --rm --read-only --user "$(id -u):$(id -g)" \
#     -v "$PWD/w:/work" -e SMOKE_WORKDIR=/work IMAGE bash /opt/pinchbench-grpo/smoke_test.sh
set -euo pipefail
W="${SMOKE_WORKDIR:-${PINCHBENCH_WORKDIR:-$PWD/.smoke}}"
export HOME="$W/home" TMPDIR="$W/tmp" PINCHBENCH_WORKDIR="$W/pb" XDG_CACHE_HOME="$W/cache"
export UV_CACHE_DIR="$W/cache/uv" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$HOME" "$TMPDIR" "$PINCHBENCH_WORKDIR" "$XDG_CACHE_HOME"
pass() { echo "PASS: $*"; }

echo "user: $(id)"
[[ "$(id -u)" != "0" ]] && pass "running as non-root" || echo "WARN: running as root"
if touch /opt/.rw_probe 2>/dev/null; then rm -f /opt/.rw_probe; echo "WARN: rootfs is writable"; else pass "rootfs read-only"; fi
if touch /tmp/.rw_probe 2>/dev/null; then rm -f /tmp/.rw_probe; echo "INFO: /tmp writable (not used by rollouts)"; else pass "/tmp not writable (rollouts must not need it)"; fi

cat /opt/pinchbench-grpo/BUILD_INFO
node --version | grep -q '^v22\.' && pass "node $(node --version)"
openclaw --version | tee "$W/openclaw.version" | grep -q '2026.5.28' && pass "openclaw $(cat "$W/openclaw.version")"
fws --version >/dev/null && pass "fws $(fws --version)"

/usr/bin/python3 - <<'PY'
import sys
sys.path.insert(0, "/opt/pinchbench/scripts")
import benchmark, lib_agent, lib_grading, lib_paths  # noqa: F401
assert "judge_failed" in lib_grading.GradeResult.__dataclass_fields__
assert hasattr(lib_agent, "_apply_web_tool_env_config")
print("PASS: benchmark.py importable; skill commit", open("/opt/pinchbench/.pinchbench_commit").read().strip())
PY
/usr/bin/python3 /opt/pinchbench/scripts/benchmark.py --help >/dev/null 2>&1 && pass "benchmark.py --help"

cd "$W"
/opt/nemo_rl_venv/bin/python -c "import nemo_rl, torch; print('PASS: import nemo_rl', nemo_rl.__file__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
GYM=/opt/nemo-rl/3rdparty/Gym-workspace/Gym
PYTHONPATH="$GYM" /opt/nemo_rl_venv/bin/python -c "
from responses_api_agents.pinchbench.app import resolve_local_work_root, LOCAL_PROVIDER
print('PASS: gym pinchbench agent (local provider) importable')"
if /opt/nemo_rl_venv/bin/python -c "import pytest, pytest_asyncio" 2>/dev/null; then
  PYTHONPATH="$GYM" /opt/nemo_rl_venv/bin/python -m pytest -q -p no:cacheprovider \
    --basetemp="$W/pytest" --rootdir="$GYM" \
    "$GYM/responses_api_agents/pinchbench/tests/test_local_provider.py" \
    && pass "gym local-provider unit tests"
else
  echo "SKIP: pytest/pytest-asyncio not in /opt/nemo_rl_venv"
fi
echo "SMOKE OK"
