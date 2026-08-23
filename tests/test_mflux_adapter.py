"""Tests for the in-process MFLUX adapter behind fakes."""

from pathlib import Path

import pytest
from PIL import Image

from hypergen.domain.models import ImageGenerationInputs
from hypergen.generation.errors import ImageGenerationError, ModelLoadError
from hypergen.generation.mflux_generator import (
    MfluxGenerationRequest,
    MfluxGenerator,
)


class FakeGeneratedImage:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def save(self, path: Path, *, overwrite: bool) -> None:
        assert not overwrite
        Image.new("RGB", (self.width, self.height), "navy").save(path, format="PNG")


class FakeMfluxModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_image(self, **kwargs: object) -> FakeGeneratedImage:
        self.calls.append(kwargs)
        return FakeGeneratedImage(
            width=kwargs["width"],  # type: ignore[arg-type]
            height=kwargs["height"],  # type: ignore[arg-type]
        )


class CorruptGeneratedImage:
    def save(self, path: Path, *, overwrite: bool) -> None:
        path.write_bytes(b"not-a-png")


def request(output_path: Path) -> MfluxGenerationRequest:
    return MfluxGenerationRequest(
        inputs=ImageGenerationInputs(
            description="A storybook watercolor courtyard"
        ),
        render_prompt="A storybook watercolor courtyard",
        output_path=output_path,
        seed=42,
    )


def test_mflux_adapter_loads_once_and_records_effective_metadata(tmp_path: Path) -> None:
    model = FakeMfluxModel()
    factory_calls: list[tuple[str, int | None]] = []

    def factory(model_identifier: str, quantization: int | None) -> FakeMfluxModel:
        factory_calls.append((model_identifier, quantization))
        return model

    generator = MfluxGenerator(model_factory=factory)
    first = generator.generate(request(tmp_path / "first.png"))
    second = generator.generate(request(tmp_path / "second.png"))

    assert factory_calls == [("flux2-klein-4b", None)]
    assert len(model.calls) == 2
    with Image.open(first.output_path) as image:
        assert image.size == (1024, 768)
    assert second.output_path.is_file()
    assert first.metadata.model_identifier == "flux2-klein-4b"
    assert first.metadata.width == 1024
    assert first.metadata.height == 768
    assert first.metadata.step_count == 4
    assert first.metadata.render_prompt == "A storybook watercolor courtyard"
    assert first.load_duration_seconds >= 0
    assert first.generation_duration_seconds >= 0
    assert first.serialization_duration_seconds >= 0


def test_mflux_adapter_refuses_overwrite(tmp_path: Path) -> None:
    output_path = tmp_path / "existing.png"
    output_path.write_bytes(b"existing")
    generator = MfluxGenerator(model_factory=lambda *_: FakeMfluxModel())

    with pytest.raises(ImageGenerationError, match="Refusing to overwrite"):
        generator.generate(request(output_path))


def test_mflux_adapter_surfaces_model_load_failure(tmp_path: Path) -> None:
    def failing_factory(model_identifier: str, quantization: int | None) -> FakeMfluxModel:
        raise RuntimeError("weights unavailable")

    generator = MfluxGenerator(model_factory=failing_factory)

    with pytest.raises(ModelLoadError, match="weights unavailable"):
        generator.generate(request(tmp_path / "image.png"))


def test_mflux_adapter_rejects_undecodable_output(tmp_path: Path) -> None:
    class CorruptModel:
        def generate_image(self, **kwargs: object) -> CorruptGeneratedImage:
            return CorruptGeneratedImage()

    output_path = tmp_path / "corrupt.png"
    generator = MfluxGenerator(model_factory=lambda *_: CorruptModel())

    with pytest.raises(ImageGenerationError, match="unreadable image"):
        generator.generate(request(output_path))

    assert not output_path.exists()
