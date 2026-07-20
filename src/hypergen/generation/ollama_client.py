"""Narrow wrapper around the official Ollama Python client."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast

import httpx
from ollama import Client, RequestError, ResponseError

from hypergen.generation.errors import (
    ModelResponseError,
    ModelUnavailableError,
    ServiceUnavailableError,
)


class OllamaClientProtocol(Protocol):
    """Subset of the official client used by HyperGen."""

    def list(self) -> Any: ...

    def show(self, model: str) -> Any: ...

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        stream: bool,
        keep_alive: int,
    ) -> Any: ...

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        stream: bool,
        think: bool | str | None,
        format: dict[str, Any],
        options: dict[str, Any],
        keep_alive: str | float | None,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class OllamaSettings:
    """Machine-local Ollama connection and inference settings."""

    endpoint: str = "http://localhost:11434"
    model: str = "qwen3.5:9b"
    think: bool | str | None = False
    temperature: float = 0.0
    keep_alive: str | float | None = "10m"
    request_timeout_seconds: float = 300.0
    num_predict: int = 2048
    context_length: int = 8192

    def __post_init__(self) -> None:
        if isinstance(self.request_timeout_seconds, bool) or not isinstance(
            self.request_timeout_seconds, (int, float)
        ):
            raise ValueError("Ollama request timeout must be a built-in number")
        if not isfinite(self.request_timeout_seconds) or self.request_timeout_seconds <= 0:
            raise ValueError("Ollama request timeout must be positive and finite")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise ValueError("Ollama temperature must be a built-in number")
        if not isfinite(self.temperature):
            raise ValueError("Ollama temperature must be finite")
        if self.keep_alive is not None and not isinstance(self.keep_alive, str):
            if isinstance(self.keep_alive, bool) or not isinstance(self.keep_alive, (int, float)):
                raise ValueError("Ollama keep-alive must be a duration string or built-in number")
            if not isfinite(self.keep_alive) or self.keep_alive < 0:
                raise ValueError("Ollama keep-alive seconds must be nonnegative and finite")
        if not isinstance(self.num_predict, int) or isinstance(self.num_predict, bool):
            raise ValueError("Ollama output token limit must be an integer")
        if self.num_predict <= 0:
            raise ValueError("Ollama output token limit must be positive")
        if not isinstance(self.context_length, int) or isinstance(self.context_length, bool):
            raise ValueError("Ollama context length must be an integer")
        if self.context_length <= 0:
            raise ValueError("Ollama context length must be positive")


@dataclass(frozen=True, slots=True)
class OllamaCallResult:
    """Raw structured-call output and timing metadata."""

    content: str
    elapsed_seconds: float
    total_duration_ns: int | None
    load_duration_ns: int | None


class OllamaRuntime:
    """Official-client runtime shared by production prompt adapters."""

    def __init__(
        self,
        settings: OllamaSettings,
        *,
        client: OllamaClientProtocol | None = None,
    ) -> None:
        self.settings = settings
        self._client = client or Client(
            host=settings.endpoint,
            timeout=settings.request_timeout_seconds,
        )

    def require_model(self, *, capabilities: frozenset[str] = frozenset()) -> None:
        """Check model availability and required capabilities with actionable errors."""
        try:
            listed = self._client.list()
        except (httpx.HTTPError, RequestError, ResponseError) as error:
            raise ServiceUnavailableError(
                f"Cannot reach Ollama at {self.settings.endpoint}. "
                "Start the Ollama daemon and verify the configured endpoint."
            ) from error

        available_models = {
            model.model
            for model in getattr(listed, "models", ())
            if getattr(model, "model", None) is not None
        }
        if self.settings.model not in available_models:
            raise ModelUnavailableError(
                f"Ollama model {self.settings.model!r} is not installed. "
                f"Run `ollama pull {self.settings.model}`."
            )

        if not capabilities:
            return
        try:
            details = self._client.show(self.settings.model)
        except (httpx.HTTPError, RequestError, ResponseError) as error:
            raise ServiceUnavailableError(
                f"Ollama could not inspect model {self.settings.model!r}."
            ) from error
        available_capabilities = frozenset(getattr(details, "capabilities", ()) or ())
        missing = capabilities - available_capabilities
        if missing:
            names = ", ".join(sorted(missing))
            raise ModelUnavailableError(
                f"Ollama model {self.settings.model!r} lacks required capabilities: {names}."
            )

    def unload_model(self) -> None:
        """Unload the configured model so a subsequent call measures cold load."""
        try:
            self._client.generate(
                model=self.settings.model,
                prompt="",
                stream=False,
                keep_alive=0,
            )
        except (httpx.HTTPError, RequestError, ResponseError) as error:
            raise ServiceUnavailableError(
                f"Ollama could not unload model {self.settings.model!r}: {error}"
            ) from error

    def chat_structured(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        image_path: Path | None = None,
    ) -> OllamaCallResult:
        """Run one synchronous structured-output request."""
        message: dict[str, Any] = {"role": "user", "content": prompt}
        if image_path is not None:
            message["images"] = [image_path]
        started = perf_counter()
        try:
            response = self._client.chat(
                model=self.settings.model,
                messages=[message],
                stream=False,
                think=self.settings.think,
                format=schema,
                options={
                    "temperature": self.settings.temperature,
                    "num_predict": self.settings.num_predict,
                    "num_ctx": self.settings.context_length,
                },
                keep_alive=self.settings.keep_alive,
            )
        except (httpx.HTTPError, RequestError, ResponseError) as error:
            raise ServiceUnavailableError(
                f"Ollama request to {self.settings.endpoint} failed for "
                f"model {self.settings.model!r}: {error}"
            ) from error
        elapsed_seconds = perf_counter() - started
        response = cast(Any, response)
        content = getattr(getattr(response, "message", None), "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ModelResponseError(
                f"Ollama model {self.settings.model!r} returned an empty structured response."
            )
        return OllamaCallResult(
            content=content,
            elapsed_seconds=elapsed_seconds,
            total_duration_ns=getattr(response, "total_duration", None),
            load_duration_ns=getattr(response, "load_duration", None),
        )
