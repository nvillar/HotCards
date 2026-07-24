"""Tests for the opt-in Scene enrichment contract."""

from __future__ import annotations

import json
from types import SimpleNamespace

from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings
from hypergen.generation.scene_enrichment import (
    OllamaSceneEnricher,
    SceneEnrichmentRequest,
    build_scene_enrichment_prompt,
)


class FakeOllamaClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> SimpleNamespace:
        self.messages.extend(kwargs["messages"])  # type: ignore[arg-type]
        return SimpleNamespace(
            message=SimpleNamespace(content=self.content),
            total_duration=2_000_000,
            load_duration=500_000,
        )


def test_scene_enrichment_prompt_uses_scene_and_style_without_interactions() -> None:
    prompt = build_scene_enrichment_prompt(
        SceneEnrichmentRequest(
            scene="A mysterious wood",
            effective_style="Ink and watercolor",
        )
    )

    assert "A mysterious wood" in prompt
    assert "Ink and watercolor" in prompt
    assert "Do not add interactions" in prompt
    assert "interaction_description" not in prompt


def test_ollama_scene_enricher_parses_structured_output() -> None:
    content = json.dumps(
        {
            "scene": (
                "An ancient moonlit wood with silver mist winding between "
                "moss-covered trunks."
            )
        }
    )
    client = FakeOllamaClient(content)
    runtime = OllamaRuntime(OllamaSettings(), client=client)  # type: ignore[arg-type]

    result = OllamaSceneEnricher(runtime).enrich(
        SceneEnrichmentRequest(scene="A mysterious wood")
    )

    assert result.scene.startswith("An ancient moonlit wood")
    assert result.raw_response == content
    assert result.model_identifier == "qwen3.5:9b"
    assert result.total_duration_ns == 2_000_000
    assert client.messages[0]["role"] == "user"
