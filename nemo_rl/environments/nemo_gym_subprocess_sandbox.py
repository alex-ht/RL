from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from typing import Any, Dict, Optional
import subprocess
import sys
import tempfile
import os


class SubprocessSandboxConfig(BaseResourcesServerConfig):
    timeout: int = 30
    max_output_lines: int = 50
    num_processes: int = 4


class SubprocessSandboxVerifyRequest(BaseVerifyRequest):
    code: str
    verifier_metadata: Optional[Dict[str, Any]] = None


class SubprocessSandboxVerifyResponse(BaseVerifyResponse):
    output: Optional[str] = None
    error: Optional[str] = None
    timed_out: bool = False


class SubprocessSandboxServer(SimpleResourcesServer):
    config: SubprocessSandboxConfig

    async def verify(self, body: SubprocessSandboxVerifyRequest) -> SubprocessSandboxVerifyResponse:
        result = self._execute_code(body.code)

        if result["timed_out"]:
            return SubprocessSandboxVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                output="",
                error="<timeout>",
                timed_out=True,
            )

        output = result.get("stdout", "") + result.get("stderr", "")

        return SubprocessSandboxVerifyResponse(
            **body.model_dump(),
            reward=1.0 if result["returncode"] == 0 else 0.0,
            output=output[: self.config.max_output_lines * 100],
            error=result.get("stderr"),
            timed_out=False,
        )

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
