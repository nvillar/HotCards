"""Tests for the opt-in Description enrichment contract."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from hypergen.domain.models import ReferenceRole
from hypergen.generation.errors import ModelResponseError
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings
from hypergen.generation.scene_enrichment import (
    OllamaSceneEnricher,
    SceneEnrichmentReference,
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


def test_scene_enrichment_prompt_expands_authored_visual_details() -> None:
    prompt = build_scene_enrichment_prompt(
        SceneEnrichmentRequest(scene='A mysterious wood with a sign saying "Enter"')
    )

    assert "A mysterious wood" in prompt
    assert "materials" in prompt
    assert "lighting quality and direction" in prompt
    assert "Use 30 to 80 words by default" in prompt
    assert "Expand only as needed to preserve explicit input details" in prompt
    assert "main subject, key action or pose, critical visual" in prompt
    assert "Important elements come first" in prompt
    assert "direct, positive language" in prompt
    assert "with its specific object" in prompt
    assert "Preserve exact" in prompt
    assert "authored color names and hex codes" in prompt
    assert "authored" in prompt
    assert "or assigned Style is explicitly photographic" in prompt
    assert "Preserve authored visible text exactly" in prompt
    assert "without inventing story facts, interactions" in prompt
    assert "dimensions, model settings" in prompt
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
                        "A black basalt castle with copper roofs, seen from its drawbridge"
                    ),
                ),
            ),
        )
    )

    assert '"roles": [' in prompt
    assert '"subject"' in prompt
    assert '"setting"' in prompt
    assert "black basalt castle" in prompt
    assert "source Descriptions" in prompt
    assert "never mention sources, references, roles" in prompt
    assert "Assigned references are mandatory" in prompt
    assert "Resolve every conflict by replacement" in prompt
    assert "remove the losing detail completely" in prompt
    assert "authored Description is the scene skeleton" in prompt
    assert "References never override authored actions, poses, or object states" in prompt
    assert "Restate every explicit authored action, pose, and" in prompt
    assert "final hatch must be open" in prompt
    assert "Discard conflicting" in prompt
    assert "replace conflicting authored subject identity and appearance" in prompt
    assert "replace conflicting authored style" in prompt
    assert "replace a conflicting authored location or environment" in prompt
    assert "distinctive source phrase verbatim" in prompt
    assert "introduce no visible words" in prompt
    assert "one synthesis" in prompt
    assert "open versus closed" in prompt


def test_scene_enrichment_rejects_empty_authored_description() -> None:
    with pytest.raises(ValidationError):
        SceneEnrichmentRequest(scene="")


def test_scene_enrichment_rejects_duplicate_reference_roles() -> None:
    with pytest.raises(ValidationError, match="only once"):
        SceneEnrichmentRequest(
            scene="A castle gate",
            references=(
                SceneEnrichmentReference(
                    roles=(ReferenceRole.VISUAL_STYLE,),
                    source_description="Low-polygon rendering",
                ),
                SceneEnrichmentReference(
                    roles=(ReferenceRole.VISUAL_STYLE,),
                    source_description="Pixel art",
                ),
            ),
        )


def test_ollama_scene_enricher_parses_structured_output() -> None:
    content = json.dumps(
        {"scene": ("An ancient moonlit wood with silver mist winding between moss-covered trunks.")}
    )
    client = FakeOllamaClient(content)
    runtime = OllamaRuntime(OllamaSettings(), client=client)  # type: ignore[arg-type]

    result = OllamaSceneEnricher(runtime).enrich(SceneEnrichmentRequest(scene="A mysterious wood"))

    assert result.scene.startswith("An ancient moonlit wood")
    assert result.raw_response == content
    assert result.model_identifier == "qwen3.5:9b-mlx"
    assert result.total_duration_ns == 2_000_000
    assert client.messages[0]["role"] == "user"


def test_ollama_scene_enricher_accepts_standalone_json_fence() -> None:
    client = FakeOllamaClient('```json\n{"scene":"An ancient moonlit wood veiled in mist"}\n```')
    runtime = OllamaRuntime(OllamaSettings(), client=client)  # type: ignore[arg-type]

    result = OllamaSceneEnricher(runtime).enrich(SceneEnrichmentRequest(scene="A mysterious wood"))

    assert result.scene == "An ancient moonlit wood veiled in mist"


def test_enrichment_repairs_reference_state_that_overrides_authored_intent() -> None:
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
            SceneEnrichmentRequest(scene="An open circular space station hatch.")
        )


def test_reference_enrichment_requires_verbatim_role_fidelity() -> None:
    source_description = (
        "A castle rendered in a low-polygon aesthetic inspired by early 1990s "
        "video games, with pixelated texture and flat-shaded terrain."
    )
    scene = (
        "A low-polygon aesthetic shapes the outer gate in the manner of early "
        "1990s video games, with angular geometry and flat shading."
    )
    content = json.dumps(
        {
            "scene": scene,
        }
    )
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(content),
    )

    result = OllamaSceneEnricher(runtime).enrich(
        SceneEnrichmentRequest(
            scene="The castle's outer gate",
            references=(
                SceneEnrichmentReference(
                    roles=(ReferenceRole.VISUAL_STYLE,),
                    source_description=source_description,
                ),
            ),
        )
    )

    assert result.scene == scene


def test_reference_enrichment_rejects_ignored_style_context() -> None:
    content = json.dumps(
        {
            "scene": ("A photorealistic weathered castle with soft mist and cinematic lighting."),
        }
    )
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(content),
    )

    with pytest.raises(ModelResponseError, match="did not preserve"):
        OllamaSceneEnricher(runtime).enrich(
            SceneEnrichmentRequest(
                scene="The castle's outer gate",
                references=(
                    SceneEnrichmentReference(
                        roles=(ReferenceRole.VISUAL_STYLE,),
                        source_description=(
                            "A low-polygon aesthetic from early 1990s video "
                            "games with pixelated textures."
                        ),
                    ),
                ),
            )
        )


def test_reference_enrichment_rejects_content_only_overlap() -> None:
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(
            json.dumps(
                {
                    "scene": (
                        "A castle gate rendered beneath a warm sunset with weathered stone walls."
                    )
                }
            )
        ),
    )

    with pytest.raises(ModelResponseError, match="distinctive style"):
        OllamaSceneEnricher(runtime).enrich(
            SceneEnrichmentRequest(
                scene="A castle gate",
                references=(
                    SceneEnrichmentReference(
                        roles=(ReferenceRole.VISUAL_STYLE,),
                        source_description=(
                            "A castle gate rendered in a low-polygon aesthetic "
                            "with pixelated textures."
                        ),
                    ),
                ),
            )
        )


def test_reference_enrichment_rejects_retained_conflicting_authored_style() -> None:
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(
            json.dumps(
                {
                    "scene": (
                        "A sketchy charcoal castle rendered with photorealistic "
                        "stone textures and crisp directional lighting."
                    )
                }
            )
        ),
    )

    with pytest.raises(ModelResponseError, match="did not override"):
        OllamaSceneEnricher(runtime).enrich(
            SceneEnrichmentRequest(
                scene="A castle in a loose, sketchy charcoal style.",
                references=(
                    SceneEnrichmentReference(
                        roles=(ReferenceRole.VISUAL_STYLE,),
                        source_description=(
                            "A photorealistic image with physically accurate stone "
                            "textures and crisp directional lighting."
                        ),
                    ),
                ),
            )
        )


def test_reference_enrichment_does_not_treat_oil_lamp_as_a_style() -> None:
    scene = "A traveler carries an oil lamp through a photorealistic stone corridor."
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(json.dumps({"scene": scene})),
    )

    result = OllamaSceneEnricher(runtime).enrich(
        SceneEnrichmentRequest(
            scene="A traveler carrying an oil lamp.",
            references=(
                SceneEnrichmentReference(
                    roles=(ReferenceRole.VISUAL_STYLE,),
                    source_description=("A photorealistic image with natural stone textures."),
                ),
            ),
        )
    )

    assert result.scene == scene


def test_reference_enrichment_rejects_conflicting_painting_medium() -> None:
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(
            json.dumps(
                {
                    "scene": (
                        "An oil painting of a castle with thick brushwork and watercolor washes."
                    )
                }
            )
        ),
    )

    with pytest.raises(ModelResponseError, match="did not override"):
        OllamaSceneEnricher(runtime).enrich(
            SceneEnrichmentRequest(
                scene="A castle in watercolor style.",
                references=(
                    SceneEnrichmentReference(
                        roles=(ReferenceRole.VISUAL_STYLE,),
                        source_description="An oil painting with thick brushwork.",
                    ),
                ),
            )
        )


def test_reference_enrichment_accepts_source_approved_mixed_media() -> None:
    scene = "An ink illustration of a castle with watercolor washes."
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(json.dumps({"scene": scene})),
    )

    result = OllamaSceneEnricher(runtime).enrich(
        SceneEnrichmentRequest(
            scene="A castle in watercolor style.",
            references=(
                SceneEnrichmentReference(
                    roles=(ReferenceRole.VISUAL_STYLE,),
                    source_description=("An ink illustration with watercolor washes."),
                ),
            ),
        )
    )

    assert result.scene == scene


def test_reference_enrichment_requires_all_coordinated_source_media() -> None:
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(
            json.dumps({"scene": ("A watercolor illustration of a castle with textured washes.")})
        ),
    )

    with pytest.raises(ModelResponseError, match="distinctive style"):
        OllamaSceneEnricher(runtime).enrich(
            SceneEnrichmentRequest(
                scene="A castle.",
                references=(
                    SceneEnrichmentReference(
                        roles=(ReferenceRole.VISUAL_STYLE,),
                        source_description=(
                            "An ink and watercolor illustration with textured washes."
                        ),
                    ),
                ),
            )
        )


@pytest.mark.parametrize(
    ("source", "scene"),
    (
        (
            "An illustration in ink and watercolor.",
            "A watercolor illustration of a castle.",
        ),
        (
            "A drawing in charcoal and pastel.",
            "A pastel drawing of a castle.",
        ),
    ),
)
def test_reference_enrichment_requires_all_postpositive_source_media(
    source: str,
    scene: str,
) -> None:
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(json.dumps({"scene": scene})),
    )

    with pytest.raises(ModelResponseError, match="distinctive style"):
        OllamaSceneEnricher(runtime).enrich(
            SceneEnrichmentRequest(
                scene="A castle.",
                references=(
                    SceneEnrichmentReference(
                        roles=(ReferenceRole.VISUAL_STYLE,),
                        source_description=source,
                    ),
                ),
            )
        )


def test_reference_enrichment_does_not_treat_oil_lamp_near_art_as_style() -> None:
    scene = "A traveler carries an oil lamp beside a framed painting in a photorealistic room."
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(json.dumps({"scene": scene})),
    )

    result = OllamaSceneEnricher(runtime).enrich(
        SceneEnrichmentRequest(
            scene="A traveler carries an oil lamp beside a framed painting.",
            references=(
                SceneEnrichmentReference(
                    roles=(ReferenceRole.VISUAL_STYLE,),
                    source_description=("A photorealistic image with natural room lighting."),
                ),
            ),
        )
    )

    assert result.scene == scene


def test_reference_enrichment_rejects_substituted_drawing_medium() -> None:
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(
            json.dumps({"scene": ("A pencil sketch of a castle with delicate graphite shading.")})
        ),
    )

    with pytest.raises(ModelResponseError, match="distinctive style"):
        OllamaSceneEnricher(runtime).enrich(
            SceneEnrichmentRequest(
                scene="A castle in a pencil sketch.",
                references=(
                    SceneEnrichmentReference(
                        roles=(ReferenceRole.VISUAL_STYLE,),
                        source_description=("An ink illustration with dense cross-hatching."),
                    ),
                ),
            )
        )


@pytest.mark.parametrize(
    "source",
    (
        "Pixel art with a limited palette.",
        "A low-polygon render with angular geometry.",
    ),
)
def test_reference_enrichment_rejects_photorealism_with_digital_style(
    source: str,
) -> None:
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(
            json.dumps({"scene": (f"A photorealistic castle in {source.rstrip('.').lower()}.")})
        ),
    )

    with pytest.raises(ModelResponseError, match="did not override"):
        OllamaSceneEnricher(runtime).enrich(
            SceneEnrichmentRequest(
                scene="A photorealistic castle.",
                references=(
                    SceneEnrichmentReference(
                        roles=(ReferenceRole.VISUAL_STYLE,),
                        source_description=source,
                    ),
                ),
            )
        )


def test_reference_enrichment_rejects_an_unknown_omitted_style() -> None:
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(json.dumps({"scene": "A castle rendered photorealistically."})),
    )

    with pytest.raises(ModelResponseError, match="distinctive style"):
        OllamaSceneEnricher(runtime).enrich(
            SceneEnrichmentRequest(
                scene="A castle gate",
                references=(
                    SceneEnrichmentReference(
                        roles=(ReferenceRole.VISUAL_STYLE,),
                        source_description="A castle in a woodcut treatment.",
                    ),
                ),
            )
        )


def test_reference_enrichment_accepts_an_unknown_style_phrase() -> None:
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(json.dumps({"scene": "A castle in a stark woodcut treatment."})),
    )

    result = OllamaSceneEnricher(runtime).enrich(
        SceneEnrichmentRequest(
            scene="A castle gate",
            references=(
                SceneEnrichmentReference(
                    roles=(ReferenceRole.VISUAL_STYLE,),
                    source_description="A castle in a woodcut treatment.",
                ),
            ),
        )
    )

    assert "woodcut treatment" in result.scene


@pytest.mark.parametrize(
    ("role", "source", "scene"),
    (
        (
            ReferenceRole.IDENTITY,
            "A red-haired pilot in brass armor.",
            "A blue tower.",
        ),
        (
            ReferenceRole.SETTING,
            "A citadel above a stormy coastline.",
            "A quiet forest with a stream.",
        ),
    ),
)
def test_reference_enrichment_rejects_trivial_role_overlap(
    role: ReferenceRole,
    source: str,
    scene: str,
) -> None:
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(json.dumps({"scene": scene})),
    )

    with pytest.raises(ModelResponseError, match="distinctive source phrase"):
        OllamaSceneEnricher(runtime).enrich(
            SceneEnrichmentRequest(
                scene="A new scene",
                references=(
                    SceneEnrichmentReference(
                        roles=(role,),
                        source_description=source,
                    ),
                ),
            )
        )


def test_reference_enrichment_accepts_a_strong_single_word_style_anchor() -> None:
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(
            json.dumps(
                {
                    "scene": (
                        "A low-polygon treatment gives the gate angular forms "
                        "and deliberately simple surfaces."
                    )
                }
            )
        ),
    )

    result = OllamaSceneEnricher(runtime).enrich(
        SceneEnrichmentRequest(
            scene="A castle gate",
            references=(
                SceneEnrichmentReference(
                    roles=(ReferenceRole.VISUAL_STYLE,),
                    source_description=("A low-polygon aesthetic with pixelated textures."),
                ),
            ),
        )
    )

    assert "low-polygon" in result.scene


def test_reference_enrichment_accepts_unicode_source_language() -> None:
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(
            json.dumps(
                {"scene": ("浮世絵 水彩画の質感で描かれた静かな城門。")},
                ensure_ascii=False,
            )
        ),
    )

    result = OllamaSceneEnricher(runtime).enrich(
        SceneEnrichmentRequest(
            scene="静かな城門。",
            references=(
                SceneEnrichmentReference(
                    roles=(ReferenceRole.VISUAL_STYLE,),
                    source_description="浮世絵水彩画の質感。",
                ),
            ),
        )
    )

    assert "浮世絵" in result.scene


def test_reference_enrichment_accepts_korean_source_language() -> None:
    scene = "고요한 성문을 목판화풍으로 묘사한다."
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(json.dumps({"scene": scene}, ensure_ascii=False)),
    )

    result = OllamaSceneEnricher(runtime).enrich(
        SceneEnrichmentRequest(
            scene="고요한 성문.",
            references=(
                SceneEnrichmentReference(
                    roles=(ReferenceRole.VISUAL_STYLE,),
                    source_description="목판화풍의 고요한 성문.",
                ),
            ),
        )
    )

    assert result.scene == scene


def test_reference_enrichment_rejects_unretained_source_language() -> None:
    content = json.dumps(
        {
            "scene": "An angular castle gate.",
        }
    )
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(content),
    )

    with pytest.raises(ModelResponseError, match="not visibly preserved"):
        OllamaSceneEnricher(runtime).enrich(
            SceneEnrichmentRequest(
                scene="The castle's outer gate",
                references=(
                    SceneEnrichmentReference(
                        roles=(ReferenceRole.VISUAL_STYLE,),
                        source_description="A low-polygon aesthetic.",
                    ),
                ),
            )
        )


def test_scene_enrichment_rejects_invented_quoted_text() -> None:
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(
            json.dumps({"scene": ('A misty castle gate with "CLOSED" carved into its portcullis.')})
        ),
    )

    with pytest.raises(ModelResponseError, match="invented visible text"):
        OllamaSceneEnricher(runtime).enrich(SceneEnrichmentRequest(scene="A misty castle gate"))


@pytest.mark.parametrize(
    "invented",
    (
        'A gate marked "CLOSED.',
        "A gate marked “CLOSED“.",
        "A gate marked CLOSED”.",
        "A gate marked CLOSED».",
    ),
)
def test_scene_enrichment_rejects_unclosed_invented_text(invented: str) -> None:
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(json.dumps({"scene": invented}, ensure_ascii=False)),
    )

    with pytest.raises(ModelResponseError, match="unclosed visible text"):
        OllamaSceneEnricher(runtime).enrich(SceneEnrichmentRequest(scene="A castle gate"))


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
        OllamaSceneEnricher(runtime).enrich(SceneEnrichmentRequest(scene="A castle gate"))


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
        OllamaSceneEnricher(runtime).enrich(SceneEnrichmentRequest(scene='A sign reads "Enter".'))


def test_scene_enrichment_disambiguates_german_and_curly_quotes() -> None:
    runtime = OllamaRuntime(  # type: ignore[arg-type]
        OllamaSettings(),
        client=FakeOllamaClient(
            json.dumps({"scene": "Signs read „HALT“ beside “GO”."}, ensure_ascii=False)
        ),
    )

    result = OllamaSceneEnricher(runtime).enrich(
        SceneEnrichmentRequest(scene="Signs read „HALT“ and “GO”.")
    )

    assert result.scene == "Signs read „HALT“ beside “GO”."
