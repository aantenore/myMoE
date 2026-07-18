from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable
from urllib import error

from .http_boundary import open_model_endpoint


@dataclass(frozen=True)
class LlamaServerSpec:
    binary: str
    model: str
    host: str
    port: int
    ctx_size: int = 4096
    threads: int = 8
    gpu_layers: int = 99
    flash_attn: str = "auto"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"

    def command(self) -> list[str]:
        return [
            self.binary,
            "-m",
            self.model,
            "-ngl",
            str(self.gpu_layers),
            "-fa",
            self.flash_attn,
            "-c",
            str(self.ctx_size),
            "-t",
            str(self.threads),
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]


class ManagedLlamaServer:
    def __init__(self, spec: LlamaServerSpec, log_path: str | Path):
        self.spec = spec
        self.log_path = Path(log_path)
        self._process: subprocess.Popen[str] | None = None
        self._log_file = None

    def __enter__(self) -> "ManagedLlamaServer":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("w", encoding="utf-8")
        self._process = subprocess.Popen(
            self.spec.command(),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        wait_for_health(self.spec.health_url, timeout_seconds=90)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10)
        if self._log_file:
            self._log_file.close()


def wait_for_health(
    url: str,
    timeout_seconds: float,
    *,
    opener: Callable[..., Any] = open_model_endpoint,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with opener(url, timeout=2) as response:
                body = response.read().decode("utf-8")
            parsed = json.loads(body)
            if parsed.get("status") == "ok":
                return parsed
            last_error = body
        except (OSError, error.URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"Server did not become healthy: {last_error}")
