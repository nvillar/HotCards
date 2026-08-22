"""Tests for the strict geometry-only Ollama hotspot remap contract."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import ValidationError

from hypergen.generation.errors import ModelResponseError
from hypergen.generation.hotspot_prompts import (
    HOTSPOT_REMAP_PROMPT_VERSION,
    HOTSPOT_REMAP_SCHEMA_VERSION,
    HotspotRemapRequest,
    OllamaHotspotRemapper,
    RemapHotspotInput,
    build_hotspot_remap_prompt,
    build_hotspot_remap_schema,
)
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        content = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        return SimpleNamespace(
            message=SimpleNamespace(content=content),
            total_duration=2_000_000,
            load_duration=500_000,
            prompt_eval_count=20,
            eval_count=10,
            done_reason="stop",
        )


def _request(
    tmp_path: Path,
    *,
    count: int = 2,
) -> HotspotRemapRequest:
    image = tmp_path / "image.png"
    Image.new("RGB", (64, 48), "navy").save(image)
    return HotspotRemapRequest(
        image_path=image,
        hotspots=tuple(
            RemapHotspotInput(token=f"H{index}", label=f"Subject {index}")
            for index in range(1, count + 1)
        ),
    )


def _runtime(client: FakeClient) -> OllamaRuntime:
    return OllamaRuntime(
        OllamaSettings(model="test-model"),
        client=client,
    )


def _response(
    *,
    token: str = "H1",
    points: list[dict[str, int]] | None = None,
    unlocated: list[dict[str, str]] | None = None,
) -> str:
    return json.dumps(
        {
            "mapped": [
                {
                    "token": token,
                    "polygons": [
                        {
                            "points": points
                            or [
                                {"x": 100, "y": 100},
                                {"x": 500, "y": 100},
                                {"x": 300, "y": 500},
                            ]
                        }
                    ],
                }
            ],
            "unlocated": unlocated or [],
        }
    )


def test_prompt_uses_only_opaque_tokens_labels_and_geometry(tmp_path: Path) -> None:
    request = _request(tmp_path)
    prompt = build_hotspot_remap_prompt(request, request.hotspots)

    assert HOTSPOT_REMAP_PROMPT_VERSION in prompt
    assert HOTSPOT_REMAP_SCHEMA_VERSION in prompt
    assert '"token": "H1"' in prompt
    assert '"label": "Subject 1"' in prompt
    assert "Do not return labels, destinations, actions" in prompt
    assert "UUID" not in prompt


def test_single_hotspot_prompt_example_returns_token_once(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, count=1)

    prompt = build_hotspot_remap_prompt(request, request.hotspots)

    assert '"unlocated": []' in prompt


def test_grid_prompt_explains_normalized_overlay(tmp_path: Path) -> None:
    request = _request(tmp_path).model_copy(
        update={"coordinate_grid_divisions": 10}
    )

    prompt = build_hotspot_remap_prompt(request, request.hotspots)

    assert "temporary 10 by 10 measurement grid" in prompt
    assert "Cyan vertical lines mark x coordinates" in prompt
    assert "minimal surrounding padding" in prompt


def test_no_grid_request_preserves_production_prompt_spacing(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)

    prompt = build_hotspot_remap_prompt(request, request.hotspots)

    assert "inventing geometry.\n\nPrompt contract:" in prompt
    assert "measurement grid" not in prompt


def test_schema_constrains_output_to_current_batch_tokens(tmp_path: Path) -> None:
    request = _request(tmp_path)
    schema = build_hotspot_remap_schema(request.hotspots)

    assert schema["$defs"]["ModelMappedHotspot"]["properties"]["token"]["enum"] == [
        "H1",
        "H2",
    ]
    assert schema["$defs"]["ModelUnlocatedHotspot"]["properties"]["token"]["enum"] == [
        "H1",
        "H2",
    ]
    assert not {
        "label",
        "destination",
        "action",
        "order",
    } & set(schema["$defs"]["ModelMappedHotspot"]["properties"])


def test_request_rejects_duplicate_tokens(tmp_path: Path) -> None:
    request = _request(tmp_path)
    with pytest.raises(ValidationError, match="tokens must be unique"):
        HotspotRemapRequest(
            image_path=request.image_path,
            hotspots=(request.hotspots[0], request.hotspots[0]),
        )


def test_adapter_maps_coordinates_and_retains_unlocated_results(
    tmp_path: Path,
) -> None:
    response = _response(
        points=[
            {"x": -10, "y": 100},
            {"x": 500, "y": 100},
            {"x": 300, "y": 500},
        ],
        unlocated=[{"token": "H2", "reason": "not visible"}],
    )
    client = FakeClient([f"```json\n{response}\n```"])

    result = OllamaHotspotRemapper(_runtime(client)).remap(_request(tmp_path))

    polygon = result.polygons_by_token["H1"][0]
    assert polygon.points[0].x == 0
    assert result.unlocated[0].token == "H2"
    assert result.warnings == ("H1 coordinates were clamped to the canvas",)
    assert len(result.raw_responses) == 1
    assert result.provenance.model_identifier == "test-model"


@pytest.mark.parametrize(
    "response, message",
    [
        (
            json.dumps(
                {
                    "mapped": [
                        {
                            "token": "H9",
                            "polygons": [
                                {
                                    "points": [
                                        {"x": 100, "y": 100},
                                        {"x": 500, "y": 100},
                                        {"x": 300, "y": 500},
                                    ]
                                }
                            ],
                        }
                    ],
                    "unlocated": [],
                }
            ),
            "unknown hotspot tokens",
        ),
        (
            json.dumps(
                {
                    "mapped": [
                        {
                            "token": "H1",
                            "polygons": [
                                {
                                    "points": [
                                        {"x": 100, "y": 100},
                                        {"x": 500, "y": 100},
                                        {"x": 300, "y": 500},
                                    ]
                                }
                            ],
                        }
                    ],
                    "unlocated": [{"token": "H1", "reason": "also missing"}],
                }
            ),
            "more than once",
        ),
    ],
)
def test_adapter_rejects_unknown_or_duplicate_output_tokens(
    tmp_path: Path,
    response: str,
    message: str,
) -> None:
    with pytest.raises(ModelResponseError, match=message):
        OllamaHotspotRemapper(_runtime(FakeClient([response]))).remap(
            _request(tmp_path)
        )


def test_invalid_geometry_and_omitted_tokens_become_unlocated(
    tmp_path: Path,
) -> None:
    response = _response(
        points=[
            {"x": 100, "y": 100},
            {"x": 200, "y": 200},
            {"x": 300, "y": 300},
        ]
    )
    result = OllamaHotspotRemapper(_runtime(FakeClient([response]))).remap(
        _request(tmp_path)
    )

    assert result.polygons_by_token == {}
    assert {item.token for item in result.unlocated} == {"H1", "H2"}
    assert any("invalid" in warning for warning in result.warnings)


def test_adapter_batches_every_existing_hotspot(tmp_path: Path) -> None:
    first = json.dumps(
        {
            "mapped": [],
            "unlocated": [
                {"token": f"H{index}", "reason": "not visible"}
                for index in range(1, 5)
            ],
        }
    )
    second = json.dumps(
        {
            "mapped": [],
            "unlocated": [{"token": "H5", "reason": "not visible"}],
        }
    )
    client = FakeClient([first, second])

    result = OllamaHotspotRemapper(_runtime(client)).remap(
        _request(tmp_path, count=5)
    )

    assert len(client.calls) == 2
    assert {item.token for item in result.unlocated} == {
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
    }


def test_adapter_requires_existing_image(tmp_path: Path) -> None:
    request = HotspotRemapRequest(
        image_path=tmp_path / "missing.png",
        hotspots=(RemapHotspotInput(token="H1", label="Door"),),
    )
    with pytest.raises(ModelResponseError, match="does not exist"):
        OllamaHotspotRemapper(_runtime(FakeClient([_response()]))).remap(request)
