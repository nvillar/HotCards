"""Tests for structured Ollama hotspot generation."""

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from hypergen.generation.errors import ModelResponseError, ModelUnavailableError
from hypergen.generation.hotspot_prompts import (
    CardCatalogueEntry,
    HotspotGenerationRequest,
    OllamaHotspotGenerator,
    UnresolvedCandidateTarget,
    build_hotspot_prompt,
    build_hotspot_response_schema,
)
from hypergen.generation.ollama_client import (
    DEFAULT_OLLAMA_MODEL,
    OllamaRuntime,
    OllamaSettings,
)


class FakeOllamaClient:
    def __init__(
        self,
        responses: list[str],
        *,
        models: tuple[str, ...] = (DEFAULT_OLLAMA_MODEL,),
        capabilities: tuple[str, ...] = ("vision", "thinking"),
    ) -> None:
        self.responses = responses
        self.models = models
        self.capabilities = capabilities
        self.chat_calls: list[dict[str, object]] = []

    def list(self) -> SimpleNamespace:
        return SimpleNamespace(models=tuple(SimpleNamespace(model=model) for model in self.models))

    def show(self, model: str) -> SimpleNamespace:
        return SimpleNamespace(capabilities=self.capabilities)

    def chat(self, **kwargs: object) -> SimpleNamespace:
        self.chat_calls.append(kwargs)
        return SimpleNamespace(
            message=SimpleNamespace(content=self.responses.pop(0)),
            total_duration=3_000_000,
            load_duration=750_000,
        )


def request(image_path: Path) -> HotspotGenerationRequest:
    return HotspotGenerationRequest(
        image_path=image_path,
        interaction_description="The gate leads to the garden",
        card_catalogue=(
            CardCatalogueEntry(
                token="C1",
                name="Garden",
                description="A moonlit garden",
            ),
        ),
    )


def hotspot_response(*, card_token: str = "C1") -> str:
    return json.dumps(
        {
            "interactions": [
                {
                    "source_interaction_index": 1,
                    "label": "Gate",
                    "destination_token": card_token,
                    "polygons": [
                        {
                            "points": [
                                {"x": -10, "y": 100},
                                {"x": 1010, "y": 100},
                                {"x": 500, "y": 900},
                            ]
                        }
                    ],
                }
            ]
        }
    )


def test_hotspot_prompt_uses_tokens_without_uuids(tmp_path: Path) -> None:
    prompt = build_hotspot_prompt(request(tmp_path / "fixture.png"))

    assert '"token": "C1"' in prompt
    assert "Garden" in prompt
    assert "UUID" in prompt
    assert "UNRESOLVED" in prompt
    assert "at most\n  4 interactions" in prompt
    assert "00000000-0000-0000-0000-000000000000" not in prompt


def test_hotspot_request_rejects_duplicate_catalogue_tokens(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="tokens must be unique"):
        HotspotGenerationRequest(
            image_path=tmp_path / "fixture.png",
            interaction_description="Choose a garden",
            card_catalogue=(
                CardCatalogueEntry(token="C1", name="North Garden"),
                CardCatalogueEntry(token="C1", name="South Garden"),
            ),
        )


def test_hotspot_schema_constrains_destination_to_request_tokens(tmp_path: Path) -> None:
    schema = build_hotspot_response_schema(request(tmp_path / "fixture.png"))

    destination = schema["$defs"]["ModelInteractionOutput"]["properties"]["destination_token"]
    assert destination["enum"] == ["C1", "UNRESOLVED"]


def test_hotspot_adapter_clamps_coordinates_and_preserves_warnings(tmp_path: Path) -> None:
    image_path = tmp_path / "fixture.png"
    image_path.write_bytes(b"fixture")
    client = FakeOllamaClient([hotspot_response()])
    runtime = OllamaRuntime(OllamaSettings(), client=client)

    result = OllamaHotspotGenerator(runtime).generate(request(image_path))

    polygon = result.proposals[0].polygons[0]
    assert polygon.points[0].x == 0.0
    assert polygon.points[1].x == 1.0
    assert "model coordinates were clamped to the canvas" in polygon.warnings
    assert result.proposals[0].target.type == "existing"
    message = client.chat_calls[0]["messages"][0]  # type: ignore[index]
    assert message["images"] == [image_path]  # type: ignore[index]
    destination = client.chat_calls[0]["format"]["$defs"]["ModelInteractionOutput"][  # type: ignore[index]
        "properties"
    ]["destination_token"]
    assert destination["enum"] == ["C1", "UNRESOLVED"]


def test_unknown_request_token_becomes_unresolved_warning(tmp_path: Path) -> None:
    image_path = tmp_path / "fixture.png"
    image_path.write_bytes(b"fixture")
    runtime = OllamaRuntime(
        OllamaSettings(),
        client=FakeOllamaClient([hotspot_response(card_token="C2")]),
    )

    result = OllamaHotspotGenerator(runtime).generate(request(image_path))

    assert isinstance(result.proposals[0].target, UnresolvedCandidateTarget)
    assert "unknown card token" in result.warnings[0]


def test_unresolved_destination_remains_reviewable(tmp_path: Path) -> None:
    image_path = tmp_path / "fixture.png"
    image_path.write_bytes(b"fixture")
    runtime = OllamaRuntime(
        OllamaSettings(),
        client=FakeOllamaClient([hotspot_response(card_token="UNRESOLVED")]),
    )

    result = OllamaHotspotGenerator(runtime).generate(request(image_path))

    assert isinstance(result.proposals[0].target, UnresolvedCandidateTarget)
    assert not any("unknown card token" in warning for warning in result.warnings)


def test_repeated_hotspot_proposals_are_flagged(tmp_path: Path) -> None:
    image_path = tmp_path / "fixture.png"
    image_path.write_bytes(b"fixture")
    repeated = json.loads(hotspot_response())["interactions"][0]
    response = json.dumps({"interactions": [repeated, repeated]})
    runtime = OllamaRuntime(
        OllamaSettings(),
        client=FakeOllamaClient([response]),
    )

    result = OllamaHotspotGenerator(runtime).generate(request(image_path))

    assert any("repeated source interaction" in warning for warning in result.warnings)


def test_unlocated_subject_returns_actionable_reconciliation_warning(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "fixture.png"
    image_path.write_bytes(b"fixture")
    response = json.dumps(
        {
            "interactions": [],
            "unlocated_interactions": [
                {
                    "source_interaction_index": 1,
                    "label": "Hidden gate",
                    "reason": "no gate is visible",
                }
            ],
        }
    )
    runtime = OllamaRuntime(
        OllamaSettings(),
        client=FakeOllamaClient([response]),
    )

    result = OllamaHotspotGenerator(runtime).generate(request(image_path))

    warning = result.reconciliation_warnings[0]
    assert warning.label == "Hidden gate"
    assert "Edit Description and regenerate" in warning.message
    assert "draw a manual hotspot" in warning.message
    assert "adjust/remove the interaction" in warning.message


def test_contradictory_unlocated_subject_is_ignored(tmp_path: Path) -> None:
    image_path = tmp_path / "fixture.png"
    image_path.write_bytes(b"fixture")
    response = json.loads(hotspot_response())
    response["unlocated_interactions"] = [
        {
            "source_interaction_index": 1,
            "label": "Gate",
            "reason": "no gate is visible",
        }
    ]
    runtime = OllamaRuntime(
        OllamaSettings(),
        client=FakeOllamaClient([json.dumps(response)]),
    )

    result = OllamaHotspotGenerator(runtime).generate(request(image_path))

    assert result.reconciliation_warnings == ()
    assert any("both located and marked" in warning for warning in result.warnings)


def test_invalid_structured_response_is_rejected(tmp_path: Path) -> None:
    image_path = tmp_path / "fixture.png"
    image_path.write_bytes(b"fixture")
    runtime = OllamaRuntime(
        OllamaSettings(),
        client=FakeOllamaClient(['{"interactions": "not-a-list"}']),
    )

    with pytest.raises(ModelResponseError, match="invalid hotspot response") as caught:
        OllamaHotspotGenerator(runtime).generate(request(image_path))

    assert caught.value.raw_response == '{"interactions": "not-a-list"}'
    assert "eval_count" in caught.value.response_metadata


def test_empty_response_preserves_call_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "fixture.png"
    image_path.write_bytes(b"fixture")
    runtime = OllamaRuntime(
        OllamaSettings(),
        client=FakeOllamaClient([""]),
    )

    with pytest.raises(ModelResponseError, match="empty structured response") as caught:
        OllamaHotspotGenerator(runtime).generate(request(image_path))

    assert caught.value.raw_response == ""
    assert caught.value.response_metadata["total_duration_ns"] == 3_000_000


def test_runtime_reports_missing_model_before_generation() -> None:
    runtime = OllamaRuntime(
        OllamaSettings(),
        client=FakeOllamaClient([], models=()),
    )

    with pytest.raises(ModelUnavailableError, match="ollama pull"):
        runtime.require_model(capabilities=frozenset({"vision"}))


def test_ollama_settings_reject_nonfinite_numbers() -> None:
    with pytest.raises(ValueError, match="timeout"):
        OllamaSettings(request_timeout_seconds=float("inf"))
    with pytest.raises(ValueError, match="temperature"):
        OllamaSettings(temperature=float("nan"))
    with pytest.raises(ValueError, match="keep-alive"):
        OllamaSettings(keep_alive=Decimal("Infinity"))  # type: ignore[arg-type]
