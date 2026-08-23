"""Tests for the opt-in Description enrichment contract."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from hypergen.domain.models import ReferenceRole
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings
from hypergen.generation.scene_enrichment import (
    OllamaSceneEnricher,
    SceneEnrichmentReference,
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


def test_ollama_runtime_lists_installed_models_in_stable_order() -> None:
    client = SimpleNamespace(
        list=lambda: SimpleNamespace(
            models=(
                SimpleNamespace(model="qwen3.5:9b-mlx"),
                SimpleNamespace(model="llama3.2:latest"),
                SimpleNamespace(model="qwen3.5:9b-mlx"),
            )
        )
    )
    runtime = OllamaRuntime(OllamaSettings(), client=client)  # type: ignore[arg-type]

    assert runtime.installed_models() == (
        "llama3.2:latest",
        "qwen3.5:9b-mlx",
    )


def test_scene_enrichment_prompt_expands_authored_visual_details() -> None:
    prompt = build_scene_enrichment_prompt(
        SceneEnrichmentRequest(scene='A mysterious wood with a sign saying "Enter"')
    )

    assert "A mysterious wood" in prompt
    assert "materials" in prompt
    assert "lighting quality and direction" in prompt
    assert "visible text in quotation marks" in prompt
    assert "Do not add interactions" in prompt
    assert "resolution or dimensions" in prompt
    assert "interaction_description" not in prompt


def test_scene_enrichment_prompt_scopes_grouped_reference_provenance() -> None:
    prompt = build_scene_enrichment_prompt(
        SceneEnrichmentRequest(
            scene="A guard approaches the outer gate",
            references=(
                SceneEnrichmentReference(
                    roles=(
                        ReferenceRole.IDENTITY,
                        ReferenceRole.SETTING,
                    ),
                    source_description=(
                        "A black basalt castle with copper roofs, seen from "
                        "its drawbridge"
                    ),
                ),
            ),
        )
    )

    assert '"roles": [' in prompt
    assert '"identity"' in prompt
    assert '"setting"' in prompt
    assert "black basalt castle" in prompt
    assert "source scenes" in prompt
    assert "mention references" in prompt


def test_scene_enrichment_rejects_empty_authored_description() -> None:
    with pytest.raises(ValidationError):
        SceneEnrichmentRequest(scene="")


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
