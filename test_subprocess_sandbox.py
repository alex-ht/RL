"""
Standalone test for SubprocessSandboxServer core logic.
Tests _execute_code and _format_observation directly.
"""
import sys
import tempfile
import os
import subprocess
from typing import Any, Dict


class SubprocessSandboxConfig:
    timeout: int = 5
    max_output_lines: int = 20


class SubprocessSandboxServer:
    def __init__(self, config=None):
        self.config = config or SubprocessSandboxConfig()

    def _format_observation(self, result: Dict[str, Any]) -> str:
        if result.get("timed_out"):
            return "<timeout>"

        output = (result.get("stdout", "") + result.get("stderr", "")).strip()

        if not output:
            return "<empty output>"

        lines = output.split("\n")
        if len(lines) > self.config.max_output_lines:
            lines = lines[: self.config.max_output_lines] + ["... (output truncated)"]

        return "\n".join(lines)

    def _execute_code(self, code: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "stdout": "",
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            proc_result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )
            result["stdout"] = proc_result.stdout
            result["stderr"] = proc_result.stderr
            result["returncode"] = proc_result.returncode
        except subprocess.TimeoutExpired as e:
            result["timed_out"] = True
            result["stdout"] = e.stdout or ""
            result["stderr"] = (e.stderr or "") + "\n[Execution timed out]"
            result["returncode"] = -1
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

        return result


def run_test(name: str, code: str, check_fn=None):
    print(f"\n=== Test: {name} ===")
    server = SubprocessSandboxServer()

    result = server._execute_code(code)
    formatted = server._format_observation(result)

    print(f"returncode: {result['returncode']}")
    print(f"timed_out: {result['timed_out']}")
    print(f"formatted_output:\n{formatted}")

    if check_fn:
        try:
            check_fn(result, formatted)
            print("✓ PASS")
        except AssertionError as e:
            print(f"✗ FAIL: {e}")
    else:
        print("✓ Executed")


def main():
    # Test 1: Normal execution
    def check_normal(r, f):
        assert r["returncode"] == 0
        assert "Hello, sandbox!" in f
        assert "3" in f

    run_test("Normal execution", "print('Hello, sandbox!')\nx = 1 + 2\nprint(x)", check_normal)

    # Test 2: Syntax error
    def check_syntax(r, f):
        assert r["returncode"] != 0

    run_test("Syntax error", "print('Hello'\n  x = 1", check_syntax)

    # Test 3: Timeout
    def check_timeout(r, f):
        assert r["timed_out"] is True

    run_test("Timeout - infinite loop", "import time\nwhile True:\n    time.sleep(1)", check_timeout)

    # Test 4: Stderr
    def check_stderr(r, f):
        assert "stderr" in f

    run_test("Stderr output", "import sys\nprint('stdout')\nprint('stderr', file=sys.stderr)", check_stderr)

    # Test 5: Empty output
    def check_empty(r, f):
        assert f == "<empty output>"

    run_test("Empty output", "pass", check_empty)

    print("\n=== All standalone tests completed ===")


if __name__ == "__main__":
    main()
