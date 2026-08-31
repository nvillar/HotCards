"""Tests for the serialized typed MFLUX adapter behind fakes."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from uuid import uuid4
from weakref import finalize, ref

import pytest
from PIL import Image

import hotcards.generation.mflux_generator as mflux_module
from hotcards.domain.image_dimensions import (
    AspectRatio,
    GenerateResolution,
    output_dimensions,
)
from hotcards.domain.models import (
    AcceptedEdit,
    EditPreserveOptions,
    GenerateInputs,
    ImageReferenceSnapshot,
    ImageSourceSnapshot,
    RefineTransformation,
)
from hotcards.generation.errors import (
    ImageGenerationCancelled,
    ImageGenerationError,
    ModelLoadError,
)
from hotcards.generation.mflux_generator import (
    MfluxCancellationToken,
    MfluxEditRequest,
    MfluxEditResult,
    MfluxGenerateRequest,
    MfluxGenerateResult,
    MfluxGenerator,
    MfluxRefineRequest,
    MfluxRefineResult,
)


@pytest.fixture(autouse=True)
def reset_process_model_cache() -> Iterator[None]:
    mflux_module._CACHED_MODEL = None
    yield
    mflux_module._CACHED_MODEL = None


class FakeGeneratedImage:
    def __init__(
        self,
        width: int,
        height: int,
        *,
        corrupt: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.corrupt = corrupt

    def save(self, path: Path, *, overwrite: bool) -> None:
        assert not overwrite
        if self.corrupt:
            path.write_bytes(b"not-a-png")
            return
        Image.new("RGB", (self.width, self.height), "navy").save(
            path,
            format="PNG",
        )


class SuffixingGeneratedImage(FakeGeneratedImage):
    def save(self, path: Path, *, overwrite: bool) -> None:
        assert not overwrite
        suffixed = path.with_name(f"{path.stem}_1{path.suffix}")
        Image.new("RGB", (self.width, self.height), "navy").save(
            suffixed,
            format="PNG",
        )


class CancellingGeneratedImage(FakeGeneratedImage):
    def __init__(
        self,
        width: int,
        height: int,
        *,
        cancellation: MfluxCancellationToken,
    ) -> None:
        super().__init__(width, height)
        self.cancellation = cancellation

    def save(self, path: Path, *, overwrite: bool) -> None:
        super().save(path, overwrite=overwrite)
        self.cancellation.cancel()


class FakeCallbackRegistry:
    def __init__(self) -> None:
        self.registered: list[object] = []

    def register(self, callback: object) -> None:
        self.registered.append(callback)


class FakeMfluxModel:
    def __init__(
        self,
        *,
        fail: Exception | None = None,
        corrupt: bool = False,
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self.callbacks = FakeCallbackRegistry()
        self.fail = fail
        self.corrupt = corrupt

    def generate_image(self, **kwargs: object) -> FakeGeneratedImage:
        self.calls.append(kwargs)
        if self.fail is not None:
            raise self.fail
        config = SimpleNamespace(
            num_inference_steps=kwargs["num_inference_steps"]
        )
        for callback in self.callbacks.registered:
            callback.call_before_loop(config=config)
        for _step in range(kwargs["num_inference_steps"]):  # type: ignore[arg-type]
            for callback in self.callbacks.registered:
                callback.call_in_loop()
        for callback in self.callbacks.registered:
            callback.call_after_loop()
        return FakeGeneratedImage(
            width=kwargs["width"],  # type: ignore[arg-type]
            height=kwargs["height"],  # type: ignore[arg-type]
            corrupt=self.corrupt,
        )


class GeneratedImageModel(FakeMfluxModel):
    def __init__(self, generated_image: FakeGeneratedImage) -> None:
        super().__init__()
        self.generated_image = generated_image

    def generate_image(self, **kwargs: object) -> FakeGeneratedImage:
        self.calls.append(kwargs)
        return self.generated_image


class RacingMfluxModel(FakeMfluxModel):
    def __init__(
        self,
        *,
        target: Path,
        foreign_bytes: bytes,
    ) -> None:
        super().__init__()
        self.target = target
        self.foreign_bytes = foreign_bytes

    def generate_image(self, **kwargs: object) -> FakeGeneratedImage:
        self.calls.append(kwargs)
        self.target.write_bytes(self.foreign_bytes)
        return FakeGeneratedImage(
            width=kwargs["width"],  # type: ignore[arg-type]
            height=kwargs["height"],  # type: ignore[arg-type]
        )


class BlockingMfluxModel(FakeMfluxModel):
    def __init__(
        self,
        *,
        entered: Event,
        release: Event,
    ) -> None:
        super().__init__()
        self.entered = entered
        self.release = release

    def generate_image(self, **kwargs: object) -> FakeGeneratedImage:
        self.calls.append(kwargs)
        config = SimpleNamespace(
            num_inference_steps=kwargs["num_inference_steps"]
        )
        for callback in self.callbacks.registered:
            callback.call_before_loop(config=config)
        self.entered.set()
        assert self.release.wait(2)
        for callback in self.callbacks.registered:
            callback.call_in_loop()
            callback.call_after_loop()
        return FakeGeneratedImage(
            width=kwargs["width"],  # type: ignore[arg-type]
            height=kwargs["height"],  # type: ignore[arg-type]
        )


def write_source(path: Path) -> Path:
    Image.new("RGB", (32, 32), "green").save(path, format="PNG")
    return path


def owned_output_scopes(directory: Path) -> list[Path]:
    return list(directory.glob(".hotcards-mflux-*"))


def generate_request(
    output_path: Path,
    *,
    references: tuple[ImageReferenceSnapshot, ...] = (),
    reference_paths: tuple[Path, ...] = (),
    model_identifier: str = "flux2-klein-4b",
    quantization: int | None = None,
) -> MfluxGenerateRequest:
    return MfluxGenerateRequest(
        inputs=GenerateInputs(
            description="A storybook watercolor courtyard",
            references=references,
        ),
        render_prompt="A storybook watercolor courtyard",
        output_path=output_path,
        model_identifier=model_identifier,
        seed=42,
        quantization=quantization,
        reference_image_paths=reference_paths,
    )


def source_snapshot() -> ImageSourceSnapshot:
    return ImageSourceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )


def accepted_edit() -> AcceptedEdit:
    return AcceptedEdit(
        instruction="Open the gate.",
        preserve=EditPreserveOptions(
            subject_identity=True,
            existing_text_and_logos=True,
        ),
        expanded_prompt=(
            "Open the gate. Preserve subject identity and existing text and logos."
        ),
    )


def refine_request(
    output_path: Path,
    source_path: Path,
    *,
    model_identifier: str = "flux2-klein-4b",
    quantization: int | None = None,
) -> MfluxRefineRequest:
    return MfluxRefineRequest(
        output_path=output_path,
        model_identifier=model_identifier,
        quantization=quantization,
        source=source_snapshot(),
        source_image_path=source_path,
        source_seed=73,
        description="A refined courtyard",
        render_prompt="A refined courtyard",
        resolution=GenerateResolution.RESOLUTION_512,
        transformation=RefineTransformation.BALANCED,
        image_strength=0.50,
    )


def edit_request(
    output_path: Path,
    source_path: Path,
    *,
    model_identifier: str = "flux2-klein-4b",
    quantization: int | None = None,
) -> MfluxEditRequest:
    edit = accepted_edit()
    return MfluxEditRequest(
        output_path=output_path,
        model_identifier=model_identifier,
        quantization=quantization,
        source=source_snapshot(),
        source_image_path=source_path,
        instruction=edit.instruction,
        preserve=edit.preserve,
        expanded_prompt=edit.expanded_prompt,
        resolution=GenerateResolution.RESOLUTION_512,
        edit_lineage=(edit,),
        seed=991,
    )


def run_in_thread(operation: object) -> tuple[Thread, list[object]]:
    outcomes: list[object] = []

    def invoke() -> None:
        try:
            outcomes.append(operation())  # type: ignore[operator]
        except Exception as error:
            outcomes.append(error)

    thread = Thread(target=invoke)
    thread.start()
    return thread, outcomes


def test_plain_generate_routes_to_regular_model_without_image_input(
    tmp_path: Path,
) -> None:
    regular = FakeMfluxModel()
    edit = FakeMfluxModel()
    generator = MfluxGenerator(
        model_factory=lambda *_: regular,
        edit_model_factory=lambda *_: edit,
    )
    progress: list[tuple[int, int]] = []

    result = generator.generate(
        generate_request(tmp_path / "plain.png"),
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert isinstance(result, MfluxGenerateResult)
    assert len(regular.calls) == 1
    assert edit.calls == []
    call = regular.calls[0]
    assert call == {
        "seed": 42,
        "prompt": "A storybook watercolor courtyard",
        "num_inference_steps": 4,
        "height": 448,
        "width": 592,
        "guidance": 1.0,
        "scheduler": "flow_match_euler_discrete",
    }
    assert progress == [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (0, 0)]
    assert result.provenance.operation == "generate"
    assert result.provenance.inputs.references == ()
    assert result.provenance.settings.mflux_version == "0.19.1"
    assert result.provenance.settings.dependency_versions["mlx"] == "0.32.2"
    assert not result.provenance.settings.use_kv_cache
    assert result.queue_duration_seconds >= 0
    assert result.load_duration_seconds >= 0
    assert result.generation_duration_seconds >= 0
    assert result.serialization_duration_seconds >= 0


def test_reference_generate_routes_ordered_images_once_to_edit_model(
    tmp_path: Path,
) -> None:
    paths = (
        write_source(tmp_path / "first.png"),
        write_source(tmp_path / "second.png"),
    )
    snapshots = tuple(
        ImageReferenceSnapshot(
            card_id=uuid4(),
            revision_id=uuid4(),
            background_id=uuid4(),
        )
        for _ in paths
    )
    regular = FakeMfluxModel()
    edit = FakeMfluxModel()
    generator = MfluxGenerator(
        model_factory=lambda *_: regular,
        edit_model_factory=lambda *_: edit,
    )

    result = generator.generate(
        generate_request(
            tmp_path / "referenced.png",
            references=snapshots,
            reference_paths=paths,
            model_identifier="flux2-klein-9b-kv",
        )
    )

    assert regular.calls == []
    assert edit.calls[0]["image_paths"] == list(paths)
    assert edit.calls[0]["use_kv_cache"] is True
    assert edit.calls[0]["image_paths"].count(paths[0]) == 1  # type: ignore[union-attr]
    assert edit.calls[0]["image_paths"].count(paths[1]) == 1  # type: ignore[union-attr]
    assert result.provenance.inputs.references == snapshots
    assert result.provenance.settings.use_kv_cache


def test_refine_routes_true_img2img_with_source_seed_and_strength(
    tmp_path: Path,
) -> None:
    source_path = write_source(tmp_path / "source.png")
    regular = FakeMfluxModel()
    edit = FakeMfluxModel()
    generator = MfluxGenerator(
        model_factory=lambda *_: regular,
        edit_model_factory=lambda *_: edit,
    )
    request = refine_request(tmp_path / "refined.png", source_path)

    result = generator.refine(request)

    assert isinstance(result, MfluxRefineResult)
    assert edit.calls == []
    assert regular.calls[0]["seed"] == 73
    assert regular.calls[0]["image_path"] == source_path
    assert regular.calls[0]["image_strength"] == 0.50
    assert "image_paths" not in regular.calls[0]
    assert result.provenance.source == request.source
    assert result.provenance.transformation is RefineTransformation.BALANCED
    assert result.provenance.strength == 0.50
    assert result.provenance.settings.seed == 73
    assert source_path.is_file()


def test_edit_routes_one_source_and_expanded_prompt_with_fresh_seed(
    tmp_path: Path,
) -> None:
    source_path = write_source(tmp_path / "source.png")
    regular = FakeMfluxModel()
    edit = FakeMfluxModel()
    generator = MfluxGenerator(
        model_factory=lambda *_: regular,
        edit_model_factory=lambda *_: edit,
    )
    request = edit_request(
        tmp_path / "edited.png",
        source_path,
        model_identifier="flux2-klein-9b-kv",
    )

    result = generator.edit(request)

    assert isinstance(result, MfluxEditResult)
    assert regular.calls == []
    assert edit.calls[0]["seed"] == 991
    assert edit.calls[0]["prompt"] == request.expanded_prompt
    assert edit.calls[0]["image_paths"] == [source_path]
    assert edit.calls[0]["use_kv_cache"] is True
    assert "image_strength" not in edit.calls[0]
    assert result.provenance.instruction == request.instruction
    assert result.provenance.preserve == request.preserve
    assert result.provenance.edit_lineage == request.edit_lineage
    assert result.provenance.settings.seed == 991
    assert source_path.is_file()


@pytest.mark.parametrize("aspect_ratio", tuple(AspectRatio))
@pytest.mark.parametrize("resolution", tuple(GenerateResolution))
def test_generate_request_accepts_every_supported_dimension_combination(
    tmp_path: Path,
    aspect_ratio: AspectRatio,
    resolution: GenerateResolution,
) -> None:
    width, height = output_dimensions(resolution, aspect_ratio)

    request = MfluxGenerateRequest(
        inputs=GenerateInputs(
            description="Valid dimensions",
            resolution=resolution,
        ),
        render_prompt="Valid dimensions",
        output_path=tmp_path / "valid.png",
        seed=1,
        aspect_ratio=aspect_ratio,
        width=width,
        height=height,
    )

    assert (request.width, request.height) == (width, height)


def test_requests_reject_mismatched_dimensions_strength_and_lineage(
    tmp_path: Path,
) -> None:
    source_path = write_source(tmp_path / "source.png")
    with pytest.raises(ValueError, match="resolution and aspect ratio"):
        generate_request(tmp_path / "invalid.png").model_copy(
            update={"width": 608}
        ).__class__.model_validate(
            {
                **generate_request(tmp_path / "invalid.png").model_dump(),
                "width": 608,
            }
        )
    with pytest.raises(ValueError, match="image strength"):
        MfluxRefineRequest.model_validate(
            {
                **refine_request(
                    tmp_path / "invalid-refine.png",
                    source_path,
                ).model_dump(),
                "image_strength": 0.25,
            }
        )
    request = edit_request(tmp_path / "invalid-edit.png", source_path)
    with pytest.raises(ValueError, match="lineage"):
        MfluxEditRequest.model_validate(
            {
                **request.model_dump(),
                "edit_lineage": (
                    AcceptedEdit(
                        instruction="Different.",
                        preserve=request.preserve,
                        expanded_prompt="Different.",
                    ),
                ),
            }
        )


def test_requests_require_exact_existing_source_paths(tmp_path: Path) -> None:
    snapshot = ImageReferenceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )
    generator = MfluxGenerator(
        model_factory=lambda *_: FakeMfluxModel(),
        edit_model_factory=lambda *_: FakeMfluxModel(),
    )
    with pytest.raises(ImageGenerationError, match="source image 1"):
        generator.generate(
            generate_request(
                tmp_path / "missing-reference-output.png",
                references=(snapshot,),
                reference_paths=(tmp_path / "missing-reference.png",),
            )
        )
    with pytest.raises(ImageGenerationError, match="source image 1"):
        generator.refine(
            refine_request(
                tmp_path / "missing-refine-output.png",
                tmp_path / "missing-source.png",
            )
        )


def test_regular_family_reuses_plain_generate_and_refine_model(
    tmp_path: Path,
) -> None:
    source_path = write_source(tmp_path / "source.png")
    regular = FakeMfluxModel()
    regular_loads: list[tuple[str, int | None]] = []
    generator = MfluxGenerator(
        model_factory=lambda model, quantization: (
            regular_loads.append((model, quantization)) or regular
        ),
    )

    generator.generate(
        generate_request(tmp_path / "plain.png"),
        progress=lambda *_: None,
    )
    generator.refine(
        refine_request(tmp_path / "refined.png", source_path),
        progress=lambda *_: None,
    )

    assert regular_loads == [("flux2-klein-4b", None)]
    assert len(regular.calls) == 2
    assert len(regular.callbacks.registered) == 1


def test_edit_family_reuses_reference_generate_and_edit_model(
    tmp_path: Path,
) -> None:
    source_path = write_source(tmp_path / "source.png")
    snapshot = ImageReferenceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )
    edit = FakeMfluxModel()
    edit_loads: list[tuple[str, int | None]] = []
    generator = MfluxGenerator(
        edit_model_factory=lambda model, quantization: (
            edit_loads.append((model, quantization)) or edit
        ),
    )

    generator.generate(
        generate_request(
            tmp_path / "referenced.png",
            references=(snapshot,),
            reference_paths=(source_path,),
        ),
        progress=lambda *_: None,
    )
    generator.edit(
        edit_request(tmp_path / "edited.png", source_path),
        progress=lambda *_: None,
    )

    assert edit_loads == [("flux2-klein-4b", None)]
    assert len(edit.calls) == 2
    assert len(edit.callbacks.registered) == 1


def test_incompatible_family_model_or_quantization_evicts_cached_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = write_source(tmp_path / "source.png")
    snapshot = ImageReferenceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )
    regular_loads: list[tuple[str, int | None]] = []
    edit_loads: list[tuple[str, int | None]] = []
    releases: list[None] = []
    monkeypatch.setattr(
        mflux_module,
        "_release_model_cache",
        lambda: releases.append(None),
    )
    generator = MfluxGenerator(
        model_factory=lambda model, quantization: (
            regular_loads.append((model, quantization)) or FakeMfluxModel()
        ),
        edit_model_factory=lambda model, quantization: (
            edit_loads.append((model, quantization)) or FakeMfluxModel()
        ),
    )

    generator.generate(generate_request(tmp_path / "regular.png"))
    generator.generate(
        generate_request(
            tmp_path / "reference.png",
            references=(snapshot,),
            reference_paths=(source_path,),
        )
    )
    generator.edit(edit_request(tmp_path / "edit.png", source_path))
    generator.generate(
        generate_request(
            tmp_path / "quantized.png",
            model_identifier="flux2-klein-9b",
            quantization=8,
        )
    )

    assert regular_loads == [
        ("flux2-klein-4b", None),
        ("flux2-klein-9b", 8),
    ]
    assert edit_loads == [("flux2-klein-4b", None)]
    assert len(releases) == 2


def test_family_switches_destroy_old_model_before_cache_clear_and_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = write_source(tmp_path / "source.png")
    snapshot = ImageReferenceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )
    events: list[str] = []
    current_ref: ref[FakeMfluxModel] | None = None
    current_label = ""

    def make_model(label: str) -> FakeMfluxModel:
        nonlocal current_ref, current_label
        model = FakeMfluxModel()
        current_ref = ref(model)
        current_label = label
        finalize(model, events.append, f"destroy:{label}")
        return model

    def clear_cache() -> None:
        assert current_ref is not None
        assert current_ref() is None
        events.append(f"clear:{current_label}")

    def regular_factory(
        _model_identifier: str,
        _quantization: int | None,
    ) -> FakeMfluxModel:
        if current_ref is not None:
            assert current_ref() is None
            assert events[-1] == f"clear:{current_label}"
        events.append("factory:regular")
        return make_model("regular")

    def edit_factory(
        _model_identifier: str,
        _quantization: int | None,
    ) -> FakeMfluxModel:
        assert current_ref is not None
        assert current_ref() is None
        assert events[-1] == "clear:regular"
        events.append("factory:edit")
        return make_model("edit")

    monkeypatch.setattr(mflux_module, "_release_model_cache", clear_cache)
    generator = MfluxGenerator(
        model_factory=regular_factory,
        edit_model_factory=edit_factory,
    )

    generator.generate(
        generate_request(tmp_path / "regular.png"),
        progress=lambda *_: None,
    )
    generator.generate(
        generate_request(
            tmp_path / "edit.png",
            references=(snapshot,),
            reference_paths=(source_path,),
        )
    )
    generator.generate(generate_request(tmp_path / "regular-again.png"))

    assert events[:7] == [
        "factory:regular",
        "destroy:regular",
        "clear:regular",
        "factory:edit",
        "destroy:edit",
        "clear:edit",
        "factory:regular",
    ]


def test_config_switch_destroys_old_model_before_cache_clear_and_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    old_ref: ref[FakeMfluxModel] | None = None

    def factory(
        model_identifier: str,
        _quantization: int | None,
    ) -> FakeMfluxModel:
        nonlocal old_ref
        if model_identifier == "flux2-klein-9b":
            assert old_ref is not None
            assert old_ref() is None
            assert events == ["destroy", "clear"]
            events.append("replacement")
            return FakeMfluxModel()
        model = FakeMfluxModel()
        old_ref = ref(model)
        finalize(model, events.append, "destroy")
        return model

    def clear_cache() -> None:
        assert old_ref is not None
        assert old_ref() is None
        events.append("clear")

    monkeypatch.setattr(mflux_module, "_release_model_cache", clear_cache)
    generator = MfluxGenerator(model_factory=factory)
    generator.generate(generate_request(tmp_path / "first.png"))

    generator.generate(
        generate_request(
            tmp_path / "second.png",
            model_identifier="flux2-klein-9b",
            quantization=8,
        )
    )

    assert events == ["destroy", "clear", "replacement"]


def test_release_discards_this_adapters_cached_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    releases: list[None] = []
    monkeypatch.setattr(
        mflux_module,
        "_release_model_cache",
        lambda: releases.append(None),
    )
    generator = MfluxGenerator(model_factory=lambda *_: FakeMfluxModel())
    generator.generate(generate_request(tmp_path / "generated.png"))

    generator.release()

    assert releases == [None]
    assert mflux_module._CACHED_MODEL is None


def test_release_destroys_model_before_clearing_mlx_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    model_ref: ref[FakeMfluxModel] | None = None

    def factory(*_args: object) -> FakeMfluxModel:
        nonlocal model_ref
        model = FakeMfluxModel()
        model_ref = ref(model)
        finalize(model, events.append, "destroy")
        return model

    def clear_cache() -> None:
        assert model_ref is not None
        assert model_ref() is None
        events.append("clear")

    monkeypatch.setattr(mflux_module, "_release_model_cache", clear_cache)
    generator = MfluxGenerator(model_factory=factory)
    generator.generate(
        generate_request(tmp_path / "generated.png"),
        progress=lambda *_: None,
    )

    generator.release()

    assert events == ["destroy", "clear"]


def test_separate_adapter_instances_share_one_process_execution_boundary(
    tmp_path: Path,
) -> None:
    first_entered = Event()
    release_first = Event()
    second_entered = Event()
    first = BlockingMfluxModel(
        entered=first_entered,
        release=release_first,
    )

    class SecondModel(FakeMfluxModel):
        def generate_image(self, **kwargs: object) -> FakeGeneratedImage:
            second_entered.set()
            return super().generate_image(**kwargs)

    first_generator = MfluxGenerator(model_factory=lambda *_: first)
    second_generator = MfluxGenerator(model_factory=lambda *_: SecondModel())
    first_thread, first_outcomes = run_in_thread(
        lambda: first_generator.generate(
            generate_request(tmp_path / "first.png")
        )
    )
    assert first_entered.wait(1)
    second_thread, second_outcomes = run_in_thread(
        lambda: second_generator.generate(
            generate_request(tmp_path / "second.png")
        )
    )

    assert not second_entered.wait(0.1)
    release_first.set()
    first_thread.join(2)
    second_thread.join(2)

    assert isinstance(first_outcomes[0], MfluxGenerateResult)
    assert isinstance(second_outcomes[0], MfluxGenerateResult)
    assert second_entered.is_set()


def test_cancellation_while_waiting_for_process_boundary_loads_no_model(
    tmp_path: Path,
) -> None:
    first_entered = Event()
    release_first = Event()
    first = BlockingMfluxModel(
        entered=first_entered,
        release=release_first,
    )
    second_loads: list[None] = []
    first_generator = MfluxGenerator(model_factory=lambda *_: first)
    second_generator = MfluxGenerator(
        model_factory=lambda *_: second_loads.append(None) or FakeMfluxModel()
    )
    first_thread, _first_outcomes = run_in_thread(
        lambda: first_generator.generate(
            generate_request(tmp_path / "first.png")
        )
    )
    assert first_entered.wait(1)
    cancellation = MfluxCancellationToken()
    second_thread, second_outcomes = run_in_thread(
        lambda: second_generator.generate(
            generate_request(tmp_path / "cancelled.png"),
            cancellation=cancellation,
        )
    )

    cancellation.cancel()
    second_thread.join(2)
    release_first.set()
    first_thread.join(2)

    assert isinstance(second_outcomes[0], ImageGenerationCancelled)
    assert second_loads == []
    assert not (tmp_path / "cancelled.png").exists()


def test_cancellation_during_generation_publishes_no_output_or_source_cleanup(
    tmp_path: Path,
) -> None:
    source_path = write_source(tmp_path / "source.png")
    entered = Event()
    release = Event()
    model = BlockingMfluxModel(entered=entered, release=release)
    generator = MfluxGenerator(model_factory=lambda *_: model)
    cancellation = MfluxCancellationToken()
    output_path = tmp_path / "cancelled-refine.png"
    thread, outcomes = run_in_thread(
        lambda: generator.refine(
            refine_request(output_path, source_path),
            cancellation=cancellation,
        )
    )
    assert entered.wait(1)

    cancellation.cancel()
    release.set()
    thread.join(2)

    assert isinstance(outcomes[0], ImageGenerationCancelled)
    assert not output_path.exists()
    assert source_path.is_file()
    assert mflux_module._CACHED_MODEL is None


def test_cancellation_destroys_active_model_before_clearing_mlx_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = Event()
    release = Event()
    cancellation = MfluxCancellationToken()
    events: list[str] = []
    model_ref: ref[FakeMfluxModel] | None = None

    def factory(*_args: object) -> FakeMfluxModel:
        nonlocal model_ref
        model = BlockingMfluxModel(entered=entered, release=release)
        model_ref = ref(model)
        finalize(model, events.append, "destroy")
        return model

    def clear_cache() -> None:
        assert model_ref is not None
        assert model_ref() is None
        events.append("clear")

    monkeypatch.setattr(mflux_module, "_release_model_cache", clear_cache)
    generator = MfluxGenerator(model_factory=factory)
    thread, outcomes = run_in_thread(
        lambda: generator.generate(
            generate_request(tmp_path / "cancelled.png"),
            cancellation=cancellation,
        )
    )
    assert entered.wait(1)

    cancellation.cancel()
    release.set()
    thread.join(2)

    assert isinstance(outcomes[0], ImageGenerationCancelled)
    assert events == ["destroy", "clear"]


def test_success_atomically_publishes_exact_target_and_cleans_owned_scope(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "published.png"
    generator = MfluxGenerator(model_factory=lambda *_: FakeMfluxModel())

    result = generator.generate(generate_request(output_path))

    assert result.output_path == output_path
    assert output_path.is_file()
    with Image.open(output_path) as generated:
        assert generated.size == (592, 448)
    assert owned_output_scopes(tmp_path) == []


def test_result_disposal_is_idempotent_and_identity_checked(tmp_path: Path) -> None:
    owned_path = tmp_path / "owned.png"
    generator = MfluxGenerator(model_factory=lambda *_: FakeMfluxModel())
    owned_result = generator.generate(generate_request(owned_path))

    owned_result.dispose_output()
    owned_result.dispose_output()

    assert not owned_path.exists()

    replaced_path = tmp_path / "replaced.png"
    replaced_result = generator.generate(generate_request(replaced_path))
    replaced_path.unlink()
    replaced_path.write_bytes(b"foreign replacement")

    replaced_result.dispose_output()

    assert replaced_path.read_bytes() == b"foreign replacement"


def test_target_created_during_inference_survives_failed_publication(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "raced.png"
    foreign_bytes = b"foreign-writer"
    model = RacingMfluxModel(
        target=output_path,
        foreign_bytes=foreign_bytes,
    )
    generator = MfluxGenerator(model_factory=lambda *_: model)

    with pytest.raises(ImageGenerationError, match="appeared before publication"):
        generator.generate(generate_request(output_path))

    assert output_path.read_bytes() == foreign_bytes
    assert owned_output_scopes(tmp_path) == []


def test_mflux_suffix_output_is_rejected_and_owned_scope_is_cleaned(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "requested.png"
    generator = MfluxGenerator(
        model_factory=lambda *_: GeneratedImageModel(
            SuffixingGeneratedImage(592, 448)
        )
    )

    with pytest.raises(ImageGenerationError, match="reserved candidate"):
        generator.generate(generate_request(output_path))

    assert not output_path.exists()
    assert owned_output_scopes(tmp_path) == []


def test_cancellation_after_temp_save_publishes_nothing_and_cleans_scope(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "cancelled-after-save.png"
    cancellation = MfluxCancellationToken()
    generator = MfluxGenerator(
        model_factory=lambda *_: GeneratedImageModel(
            CancellingGeneratedImage(
                592,
                448,
                cancellation=cancellation,
            )
        )
    )

    with pytest.raises(ImageGenerationCancelled, match="cancelled"):
        generator.generate(
            generate_request(output_path),
            cancellation=cancellation,
        )

    assert not output_path.exists()
    assert owned_output_scopes(tmp_path) == []


def test_operation_errors_are_actionable_and_leave_no_output(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "existing.png"
    output_path.write_bytes(b"existing")
    generator = MfluxGenerator(model_factory=lambda *_: FakeMfluxModel())
    with pytest.raises(ImageGenerationError, match="overwrite.*generate"):
        generator.generate(generate_request(output_path))
    assert output_path.read_bytes() == b"existing"
    assert owned_output_scopes(tmp_path) == []

    failing = MfluxGenerator(
        model_factory=lambda *_: FakeMfluxModel(
            fail=RuntimeError("inference exploded")
        )
    )
    failed_path = tmp_path / "failed.png"
    with pytest.raises(
        ImageGenerationError,
        match="MFLUX generate failed.*inference exploded",
    ):
        failing.generate(generate_request(failed_path))
    assert not failed_path.exists()
    assert owned_output_scopes(tmp_path) == []

    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("file")
    with pytest.raises(ImageGenerationError, match="prepare output directory"):
        generator.generate(generate_request(blocked_parent / "candidate.png"))


def test_model_load_and_output_validation_errors_are_typed(
    tmp_path: Path,
) -> None:
    def fail_load(
        _model_identifier: str,
        _quantization: int | None,
    ) -> FakeMfluxModel:
        raise RuntimeError("weights unavailable")

    generator = MfluxGenerator(model_factory=fail_load)
    with pytest.raises(ModelLoadError, match="regular model.*weights unavailable"):
        generator.generate(generate_request(tmp_path / "load-failed.png"))

    corrupt = MfluxGenerator(
        model_factory=lambda *_: FakeMfluxModel(corrupt=True)
    )
    output_path = tmp_path / "corrupt.png"
    with pytest.raises(ImageGenerationError, match="unreadable image"):
        corrupt.generate(generate_request(output_path))
    assert not output_path.exists()
    assert owned_output_scopes(tmp_path) == []
