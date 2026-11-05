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

"""把 Inductor / Triton 快取分到「這台機器、這個 pid」。

190987：同機 8 個 rank 共用 work/home/<host>/.inductor，一起寫同一個 .py，
匯入時出現 source code string cannot contain null bytes。

191353：.pth 在父行程（pid 1965981）改了 TORCHINDUCTOR_CACHE_DIR，並設
*_RLPB=1。Ray / vLLM 的子行程繼承這組環境變數，旗標讓它們不再重分，
所有 rank（含別台機器）都寫 work/home/cnode2-001/.inductor/pid1965981。
cubin 被同時寫壞，get_logprobs 載入 flex kernel 時
CUDA driver error: device kernel image is invalid。

這裡每次 interpreter 啟動、以及 fork 之後，都依「目前」的 pid 重算。
不設會擋住子行程的旗標。繼承來的 pid 目錄會先剝掉，主機名稱改成本機，
避免跨節點 pid 撞在同一份共用盤目錄。編譯用的子行程稍後會用
extra_env 把目錄改回父行程的；同一個 pid 不重複覆寫，所以那次改寫留得住。
"""

from __future__ import annotations

import os
import re
import socket
import sys

_OWNED = re.compile(r"^(?:[A-Za-z0-9_.-]+-)?pid\d+$")
_NAMES = ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
_applied_pid: int | None = None


def _host() -> str:
    return socket.gethostname().split(".", 1)[0]


def _strip(path: str) -> str:
    path = path.rstrip("/")
    while True:
        parent, leaf = os.path.split(path)
        if parent and parent != path and _OWNED.match(leaf):
            path = parent
            continue
        return path


def _localize(path: str) -> str:
    parts = path.split("/")
    host = _host()
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "work" and parts[i + 1] == "home" and parts[i + 2]:
            parts[i + 2] = host
            break
    return "/".join(parts)


def apply() -> None:
    global _applied_pid
    pid = os.getpid()
    if _applied_pid == pid:
        return
    host = _host()
    for name in _NAMES:
        raw = os.environ.get(name)
        if not raw:
            continue
        path = os.path.join(_localize(_strip(raw)), f"{host}-pid{pid}")
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            pass
        os.environ[name] = path
    _applied_pid = pid


def _after_fork() -> None:
    global _applied_pid
    _applied_pid = None
    apply()


class _Finder:
    def find_spec(self, fullname, path, target=None):  # noqa: ANN001
        # Ray runtime_env 可能在 .pth 之後才覆寫環境變數。torch / triton
        # 第一次被 import 時再算一次，會蓋過那個繼承來的父 pid 目錄。
        if fullname == "torch" or fullname == "triton" or fullname.startswith(
            ("torch.", "triton.")
        ):
            apply()
        return None


def _install() -> None:
    if getattr(_install, "done", False):
        return
    sys.meta_path.insert(0, _Finder())
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=_after_fork)
    _install.done = True  # type: ignore[attr-defined]


_install()
apply()
