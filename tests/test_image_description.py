"""Tests for the vision image-to-Scene contract."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from hypergen.generation.image_description import (
    ImageDescriptionRequest,
    OllamaImageDescriber,
    build_image_description_prompt,
)
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings


class FakeOllamaClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            message=SimpleNamespace(content=self.content),
            total_duration=2_000_000,
            load_duration=500_000,
        )


def test_image_description_prompt_excludes_interaction_inference(
    tmp_path: Path,
) -> None:
    request = ImageDescriptionRequest(image_path=tmp_path / "image.png")

    prompt = build_image_description_prompt(request)

    assert "text-to-image Scene prompt" in prompt
    assert "Do not infer interactions" in prompt
    assert "hotspots" in prompt


def test_ollama_image_describer_uses_image_and_parses_scene(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image")
    content = json.dumps(
        {
            "scene": (
                "A moonlit stone castle viewed from below, framed by dark pines "
                "and silver mist."
            )
        }
    )
    client = FakeOllamaClient(content)
    runtime = OllamaRuntime(OllamaSettings(), client=client)  # type: ignore[arg-type]

    result = OllamaImageDescriber(runtime).describe(
        ImageDescriptionRequest(image_path=image_path)
    )

    assert result.scene.startswith("A moonlit stone castle")
    assert result.raw_response == content
    message = client.calls[0]["messages"][0]  # type: ignore[index]
    assert message["images"] == [image_path]  # type: ignore[index]


def test_ollama_image_describer_accepts_standalone_json_fence(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image")
    client = FakeOllamaClient(
        '```json\n{"scene":"A moonlit stone castle framed by pines"}\n```'
    )
    runtime = OllamaRuntime(OllamaSettings(), client=client)  # type: ignore[arg-type]

    result = OllamaImageDescriber(runtime).describe(
        ImageDescriptionRequest(image_path=image_path)
    )

    assert result.scene == "A moonlit stone castle framed by pines"
