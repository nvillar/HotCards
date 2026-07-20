"""Tests for Ollama-derived image prompts."""

import json
from types import SimpleNamespace

from hypergen.domain.models import ImageGenerationInputs
from hypergen.generation.image_prompts import (
    OllamaImagePromptDeriver,
    build_image_prompt_request,
)
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings


class FakeOllamaClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages: list[dict[str, object]] = []

    def list(self) -> SimpleNamespace:
        return SimpleNamespace(models=(SimpleNamespace(model="qwen3.5:9b"),))

    def show(self, model: str) -> SimpleNamespace:
        return SimpleNamespace(capabilities=("vision", "thinking"))

    def chat(self, **kwargs: object) -> SimpleNamespace:
        self.messages.extend(kwargs["messages"])  # type: ignore[arg-type]
        return SimpleNamespace(
            message=SimpleNamespace(content=self.content),
            total_duration=2_000_000,
            load_duration=500_000,
        )


def inputs() -> ImageGenerationInputs:
    return ImageGenerationInputs(
        scene_description="A stone courtyard at dusk",
        interaction_description="The gate leads to the garden",
        stack_art_direction="Ink and watercolor",
        card_style="Cool shadows",
    )


def test_prompt_request_contains_author_inputs_and_visual_rules() -> None:
    prompt = build_image_prompt_request(inputs())

    assert "A stone courtyard at dusk" in prompt
    assert "The gate leads to the garden" in prompt
    assert "spatially distinct" in prompt
    assert "Do not copy navigation instructions verbatim" in prompt


def test_ollama_deriver_parses_structured_output() -> None:
    content = json.dumps(
        {
            "render_prompt": "Ink courtyard with a prominent arched gate",
            "interactive_subjects": ["arched gate"],
        }
    )
    client = FakeOllamaClient(content)
    runtime = OllamaRuntime(OllamaSettings(), client=client)

    result = OllamaImagePromptDeriver(runtime).derive(inputs())

    assert result.prompt == "Ink courtyard with a prominent arched gate"
    assert result.interactive_subjects == ("arched gate",)
    assert result.raw_response == content
    assert result.total_duration_ns == 2_000_000
    assert client.messages[0]["role"] == "user"
