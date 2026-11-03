import asyncio
import re
import subprocess
import sys
from dataclasses import dataclass

MAX_OUTPUT = 2000
_BLOCKED = re.compile(
    r"\b(?:import|from)\s+(?:os|sys|subprocess|socket|shutil|pathlib|ctypes)\b"
    r"|__import__|\b(?:open|eval|exec|compile)\s*\("
)


@dataclass(frozen=True)
class ExecResult:
    ok: bool
    output: str
    error: str


def _run_sync(code: str, timeout: float) -> ExecResult:
    try:
        done = subprocess.run(
            [sys.executable, "-I", "-X", "utf8", "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ExecResult(False, "", f"Timed out after {timeout:g}s")
    if done.returncode != 0:
        tail = "\n".join(done.stderr.strip().splitlines()[-3:])
        error = tail or f"Exited with code {done.returncode}"
        return ExecResult(False, done.stdout[:MAX_OUTPUT], error[:MAX_OUTPUT])
    return ExecResult(True, done.stdout[:MAX_OUTPUT], "")


async def run_python(code: str, timeout: float = 5.0) -> ExecResult:
    """Run generated code in a separate isolated interpreter with a hard timeout.

    The denylist and `-I` are defence in depth, NOT a real sandbox. In production run this in a
    container or a service like nsjail/gVisor with no network and a read-only filesystem."""
    if _BLOCKED.search(code):
        return ExecResult(False, "", "Blocked: forbidden import or builtin (os, open, eval ...)")
    return await asyncio.to_thread(_run_sync, code, timeout)
