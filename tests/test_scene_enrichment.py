"""Tests for the text-only Description enrichment contract."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from hypergen.generation.errors import ModelResponseError
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings
from hypergen.generation.scene_enrichment import (
    OllamaSceneEnricher,
    SceneEnrichmentRequest,
    build_scene_enrichment_prompt,
)


class FakeOllamaClient:
    def __init__(self, content: str | list[str]) -> None:
        self.contents = [content] if isinstance(content, str) else content
        self.messages: list[dict[str, object]] = []
        self.call_count = 0

    def chat(self, **kwargs: object) -> SimpleNamespace:
        self.messages.extend(kwargs["messages"])  # type: ignore[arg-type]
        content = self.contents[min(self.call_count, len(self.contents) - 1)]
        self.call_count += 1
        return SimpleNamespace(
            message=SimpleNamespace(content=content),
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


def test_scene_enrichment_prompt_expands_only_authored_visual_details() -> None:
    prompt = build_scene_enrichment_prompt(
        SceneEnrichmentRequest(
            scene=(
                'A computer fills the frame with a sign saying "Enter"; '
                "only its screen and top row of keys are visible."
            )
        )
    )

    assert "A computer fills the frame" in prompt
    assert "materials" in prompt
    assert "lighting quality and direction" in prompt
    assert "Use 30 to 80 words by default" in prompt
    assert "Expand only as needed to preserve explicit input details" in prompt
    assert "main subject, key action or pose, critical visual" in prompt
    assert "Important elements come first" in prompt
    assert "direct, positive language" in prompt
    assert "viewpoint, crop, framing, composition" in prompt
    assert "limited fields of view" in prompt
    assert "widen the view" in prompt
    assert "Preserve exact" in prompt
    assert "authored color names and hex codes" in prompt
    assert "Preserve authored visible text exactly" in prompt
    assert "without inventing story facts, interactions" in prompt
    assert "dimensions, model settings" in prompt
    assert "reference_contexts" not in prompt
    assert "REFERENCE ROLE SCOPES" not in prompt
    assert "source Description" not in prompt
    assert "interaction_description" not in prompt


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


def test_enrichment_repairs_state_that_overrides_authored_intent() -> None:
    client = FakeOllamaClient(
        [
            json.dumps(
                {
                    "scene": (
                        "A closed circular space station hatch in a monochrome "
                        "industrial corridor."
                    )
                }
            ),
            json.dumps(
                {
                    "scene": (
                        "An open circular space station hatch reveals the passage "
                        "beyond in a monochrome industrial corridor."
                    )
                }
            ),
        ]
    )
    runtime = OllamaRuntime(OllamaSettings(), client=client)  # type: ignore[arg-type]

    result = OllamaSceneEnricher(runtime).enrich(
        SceneEnrichmentRequest(scene="The hatch is open.")
    )

    assert "open circular space station hatch" in result.scene
    assert client.call_count == 2
    assert "Invalid candidate:" in client.messages[-1]["content"]
    assert "authored 'hatch' is 'open'" in client.messages[-1]["content"]


def test_enrichment_rejects_persistent_authored_state_conflict() -> None:
    content = json.dumps({"scene": "A firmly closed circular hatch."})
    client = FakeOllamaClient([content, content])
    runtime = OllamaRuntime(OllamaSettings(), client=client)  # type: ignore[arg-type]

    with pytest.raises(ModelResponseError, match="authored intent"):
        OllamaSceneEnricher(runtime).enrich(
            SceneEnrichmentRequest(scene="The hatch is open.")
        )

    assert client.call_count == 2


def test_enrichment_allows_opposite_state_on_a_different_object() -> None:
    scene = (
        "The open circular hatch leads past a sealed bulkhead into the station."
    )
    client = FakeOllamaClient(json.dumps({"scene": scene}))
    runtime = OllamaRuntime(OllamaSettings(), client=client)  # type: ignore[arg-type]

    result = OllamaSceneEnricher(runtime).enrich(
        SceneEnrichmentRequest(scene="The hatch is open.")
    )

    assert result.scene == scene
    assert client.call_count == 1


def test_enrichment_detects_attributive_object_state_conflict() -> None:
    content = json.dumps(
        {"scene": "A closed circular space station hatch fills the wall."}
    )
    client = FakeOllamaClient([content, content])
    runtime = OllamaRuntime(OllamaSettings(), client=client)  # type: ignore[arg-type]

    with pytest.raises(ModelResponseError, match="authored intent"):
        OllamaSceneEnricher(runtime).enrich(
            SceneEnrichmentRequest(
                scene="An open circular space station hatch."
            )
        )


def test_scene_enrichment_rejects_invented_quoted_text() -> None:
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(
            json.dumps(
                {
                    "scene": (
                        'A misty castle gate with "CLOSED" carved into its '
                        "portcullis."
                    )
                }
            )
        ),
    )

    with pytest.raises(ModelResponseError, match="invented visible text"):
        OllamaSceneEnricher(runtime).enrich(
            SceneEnrichmentRequest(scene="A misty castle gate")
        )


@pytest.mark.parametrize(
    "invented",
    (
        'A gate marked "CLOSED.',
        "A gate marked “CLOSED“.",
        "A gate marked CLOSED”.",
        "A gate marked CLOSED».",
    ),
)
def test_scene_enrichment_rejects_unclosed_invented_text(
    invented: str,
) -> None:
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(
            json.dumps({"scene": invented}, ensure_ascii=False)
        ),
    )

    with pytest.raises(ModelResponseError, match="unclosed visible text"):
        OllamaSceneEnricher(runtime).enrich(
            SceneEnrichmentRequest(scene="A castle gate")
        )


@pytest.mark.parametrize(
    "invented",
    (
        "A gate marked «CLOSED».",
        "A gate marked „HALT“.",
        'A gate marked "CLOSED\\nTODAY".',
    ),
)
def test_scene_enrichment_rejects_unicode_or_multiline_invented_text(
    invented: str,
) -> None:
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(json.dumps({"scene": invented})),
    )

    with pytest.raises(ModelResponseError, match="invented visible text"):
        OllamaSceneEnricher(runtime).enrich(
            SceneEnrichmentRequest(scene="A castle gate")
        )


@pytest.mark.parametrize(
    "changed",
    (
        "A sign reads nothing.",
        'A sign reads "ENTER".',
        'A sign reads " Enter ".',
    ),
)
def test_scene_enrichment_requires_exact_authored_visible_text(
    changed: str,
) -> None:
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(json.dumps({"scene": changed})),
    )

    with pytest.raises(ModelResponseError, match="visible text"):
        OllamaSceneEnricher(runtime).enrich(
            SceneEnrichmentRequest(scene='A sign reads "Enter".')
        )


def test_scene_enrichment_disambiguates_german_and_curly_quotes() -> None:
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(
            json.dumps(
                {"scene": "Signs read „HALT“ beside “GO”."},
                ensure_ascii=False,
            )
        ),
    )

    result = OllamaSceneEnricher(runtime).enrich(
        SceneEnrichmentRequest(scene="Signs read „HALT“ and “GO”.")
    )

    assert result.scene == "Signs read „HALT“ beside “GO”."
