"""Tests for Ollama model discovery and capability requirements."""

from types import SimpleNamespace

from ollama import ResponseError

from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings


class FakeOllamaClient:
    def list(self) -> SimpleNamespace:
        return SimpleNamespace(
            models=(
                SimpleNamespace(model="retired:cloud"),
                SimpleNamespace(model="text-only:latest"),
                SimpleNamespace(model="vision-model:latest"),
            )
        )

    def show(self, model: str) -> SimpleNamespace:
        if model == "retired:cloud":
            raise ResponseError("model retired", 410)
        capabilities = (
            ("completion", "vision")
            if model == "vision-model:latest"
            else ("completion",)
        )
        return SimpleNamespace(capabilities=capabilities)


def test_installed_models_can_be_filtered_by_capability() -> None:
    runtime = OllamaRuntime(
        OllamaSettings(model="vision-model:latest"),
        client=FakeOllamaClient(),  # type: ignore[arg-type]
    )

    assert runtime.installed_models() == (
        "retired:cloud",
        "text-only:latest",
        "vision-model:latest",
    )
    assert runtime.installed_models(
        capabilities=frozenset({"vision"})
    ) == ("vision-model:latest",)
