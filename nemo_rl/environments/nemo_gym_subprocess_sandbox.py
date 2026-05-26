from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from typing import Any, Dict, Optional


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
        pass
