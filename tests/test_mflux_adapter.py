"""Tests for the in-process MFLUX adapter behind fakes."""

from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image

from hypergen.domain.models import ImageGenerationInputs, ImageReferenceSnapshot
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


def test_reference_generation_uses_edit_model_and_kv_cache(
    tmp_path: Path,
) -> None:
    source_model = FakeMfluxModel()
    edit_model = FakeMfluxModel()
    source_calls: list[tuple[str, int | None]] = []
    edit_calls: list[tuple[str, int | None]] = []
    snapshot = ImageReferenceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )
    generation_request = request(tmp_path / "referenced.png").model_copy(
        update={
            "inputs": ImageGenerationInputs(
                description="A referenced portrait",
                identity_reference=snapshot,
            ),
            "render_prompt": "REFERENCE IMAGE 1\nIDENTITY\nPreserve identity",
            "model_identifier": "flux2-klein-9b-kv",
            "reference_image_paths": (tmp_path / "identity.png",),
        }
    )
    generator = MfluxGenerator(
        model_factory=lambda model, quantization: (
            source_calls.append((model, quantization)) or source_model
        ),
        edit_model_factory=lambda model, quantization: (
            edit_calls.append((model, quantization)) or edit_model
        ),
    )

    result = generator.generate(generation_request)

    assert source_calls == []
    assert edit_calls == [("flux2-klein-9b-kv", None)]
    assert edit_model.calls[0]["image_paths"] == [tmp_path / "identity.png"]
    assert edit_model.calls[0]["use_kv_cache"] is True
    assert result.metadata.inputs.identity_reference == snapshot
    assert result.metadata.effective_settings["reference_count"] == 1
    assert result.metadata.effective_settings["use_kv_cache"] is True


def test_reference_paths_must_match_snapshot_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must match"):
        MfluxGenerationRequest(
            inputs=ImageGenerationInputs(description="Missing snapshot"),
            render_prompt="Missing snapshot",
            output_path=tmp_path / "invalid.png",
            reference_image_paths=(tmp_path / "reference.png",),
            seed=1,
        )


def test_multi_role_snapshot_requires_one_unique_image_path(
    tmp_path: Path,
) -> None:
    snapshot = ImageReferenceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )

    generation_request = MfluxGenerationRequest(
        inputs=ImageGenerationInputs(
            description="Same castle",
            identity_reference=snapshot,
            setting_reference=snapshot,
        ),
        render_prompt="Same castle",
        output_path=tmp_path / "multi-role.png",
        reference_image_paths=(tmp_path / "castle.png",),
        seed=1,
    )

    assert generation_request.reference_image_paths == (
        tmp_path / "castle.png",
    )


def test_switching_generation_modes_evicts_the_previous_model(
    tmp_path: Path,
) -> None:
    source_calls = 0
    edit_calls = 0

    def source_factory(
        _model_identifier: str,
        _quantization: int | None,
    ) -> FakeMfluxModel:
        nonlocal source_calls
        source_calls += 1
        return FakeMfluxModel()

    def edit_factory(
        _model_identifier: str,
        _quantization: int | None,
    ) -> FakeMfluxModel:
        nonlocal edit_calls
        edit_calls += 1
        return FakeMfluxModel()

    snapshot = ImageReferenceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )
    generator = MfluxGenerator(
        model_factory=source_factory,
        edit_model_factory=edit_factory,
    )
    generator.generate(request(tmp_path / "source-1.png"))
    generator.generate(
        MfluxGenerationRequest(
            inputs=ImageGenerationInputs(
                description="Referenced scene",
                identity_reference=snapshot,
            ),
            render_prompt="Referenced scene",
            output_path=tmp_path / "edit.png",
            reference_image_paths=(tmp_path / "reference.png",),
            seed=1,
        )
    )
    generator.generate(request(tmp_path / "source-2.png"))

    assert source_calls == 2
    assert edit_calls == 1


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
