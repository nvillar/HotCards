"""Tests for the opt-in Description enrichment contract."""

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


def test_scene_enrichment_prompt_uses_description_and_style_without_interactions() -> None:
    prompt = build_scene_enrichment_prompt(
        SceneEnrichmentRequest(
            scene="A mysterious wood",
            effective_style="Ink and watercolor",
        )
    )

    assert "A mysterious wood" in prompt
    assert "Ink and watercolor" in prompt
    assert "Do not add interactions" in prompt
    assert "resolution or dimensions" in prompt
    assert "interaction_description" not in prompt


def test_scene_enrichment_prompt_merges_visible_image_details() -> None:
    prompt = build_scene_enrichment_prompt(
        SceneEnrichmentRequest(
            scene="A mysterious wood",
            effective_style="Ink and watercolor",
            image_description=(
                "Silver birches seen from below beneath a violet moon"
            ),
        )
    )

    assert '"authored_description": "A mysterious wood"' in prompt
    assert "Silver birches seen from below" in prompt
    assert "authoritative if it conflicts" in prompt
    assert "hidden story facts" in prompt


def test_scene_enrichment_accepts_image_context_without_authored_description() -> None:
    request = SceneEnrichmentRequest(
        image_description="A moonlit stone bridge over dark water"
    )

    assert request.scene == ""
    assert request.image_description


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
    assert result.model_identifier == "qwen3.5:9b-mlx"
    assert result.total_duration_ns == 2_000_000
    assert client.messages[0]["role"] == "user"


def test_ollama_scene_enricher_accepts_standalone_json_fence() -> None:
    client = FakeOllamaClient(
        '```json\n{"scene":"An ancient moonlit wood veiled in mist"}\n```'
    )
    runtime = OllamaRuntime(OllamaSettings(), client=client)  # type: ignore[arg-type]

    result = OllamaSceneEnricher(runtime).enrich(
        SceneEnrichmentRequest(scene="A mysterious wood")
    )

    assert result.scene == "An ancient moonlit wood veiled in mist"
