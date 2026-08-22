"""Tests for generated background revision naming."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hypergen.generation.errors import ModelResponseError
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings
from hypergen.generation.revision_naming import (
    OllamaRevisionNamer,
    RevisionNamingRequest,
    build_revision_naming_prompt,
    unique_revision_name,
)


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


def request(image_path: Path) -> RevisionNamingRequest:
    return RevisionNamingRequest(
        image_path=image_path,
        render_prompt="A moonlit castle moat with a stone bridge",
        existing_names=("Castle Gate",),
    )


def test_revision_naming_uses_image_and_validates_structured_title(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "generated.png"
    image_path.write_bytes(b"image")
    client = FakeOllamaClient(json.dumps({"name": "Moonlit Moat"}))
    runtime = OllamaRuntime(
        OllamaSettings(model="test-model"),
        client=client,  # type: ignore[arg-type]
    )

    result = OllamaRevisionNamer(runtime).name(request(image_path))

    assert result.name == "Moonlit Moat"
    assert result.model_identifier == "test-model"
    assert client.calls[0]["messages"][0]["images"] == [image_path]  # type: ignore[index]
    prompt = client.calls[0]["messages"][0]["content"]  # type: ignore[index]
    assert "A moonlit castle moat" in prompt
    assert "Castle Gate" in prompt


def test_revision_naming_accepts_valid_plain_title_from_mlx_model(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "generated.png"
    image_path.write_bytes(b"image")
    client = FakeOllamaClient("Castle Moat Bridge Scene")
    runtime = OllamaRuntime(
        OllamaSettings(model="qwen3.5:9b-mlx"),
        client=client,  # type: ignore[arg-type]
    )

    result = OllamaRevisionNamer(runtime).name(request(image_path))

    assert result.name == "Castle Moat Bridge Scene"


def test_revision_naming_accepts_standalone_json_fence(tmp_path: Path) -> None:
    image_path = tmp_path / "generated.png"
    image_path.write_bytes(b"image")
    client = FakeOllamaClient('```json\n{"name":"Moonlit Moat"}\n```')
    runtime = OllamaRuntime(
        OllamaSettings(model="qwen3.5:9b-mlx"),
        client=client,  # type: ignore[arg-type]
    )

    result = OllamaRevisionNamer(runtime).name(request(image_path))

    assert result.name == "Moonlit Moat"


@pytest.mark.parametrize("name", ["Generated", "x" * 49])
def test_revision_naming_rejects_invalid_name(
    tmp_path: Path,
    name: str,
) -> None:
    image_path = tmp_path / "generated.png"
    image_path.write_bytes(b"image")
    client = FakeOllamaClient(json.dumps({"name": name}))
    runtime = OllamaRuntime(
        OllamaSettings(),
        client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(ModelResponseError, match="invalid revision naming response"):
        OllamaRevisionNamer(runtime).name(request(image_path))


def test_revision_naming_requires_existing_image(tmp_path: Path) -> None:
    client = FakeOllamaClient(json.dumps({"name": "Moonlit Moat"}))
    runtime = OllamaRuntime(
        OllamaSettings(),
        client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(ModelResponseError, match="does not exist"):
        OllamaRevisionNamer(runtime).name(request(tmp_path / "missing.png"))

    assert client.calls == []


def test_unique_revision_name_uses_case_insensitive_first_available_index() -> None:
    assert (
        unique_revision_name(
            "Moonlit Moat",
            ("moonlit moat", "Moonlit Moat 2", "Other"),
        )
        == "Moonlit Moat 3"
    )
    assert unique_revision_name("Castle Gate", ("Other",)) == "Castle Gate"


def test_revision_naming_prompt_forbids_generic_or_numbered_names(
    tmp_path: Path,
) -> None:
    prompt = build_revision_naming_prompt(request(tmp_path / "generated.png"))

    assert "Do not use generic names" in prompt
    assert "Do not add numbering" in prompt
