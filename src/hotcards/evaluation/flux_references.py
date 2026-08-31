"""FLUX.2 Klein ordered-Reference behavior suite."""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from gc import collect
from pathlib import Path
from time import perf_counter
from typing import Protocol

from PIL import Image

from hotcards.domain.models import GenerateInputs
from hotcards.evaluation.manifest import (
    EnvironmentProvider,
    RunLifecycle,
    atomic_write_json,
    classify_failure,
    default_environment,
)
from hotcards.evaluation.reports import create_contact_sheet
from hotcards.generation.image_generation import compose_generation_prompt
from hotcards.storage.stack_store import StackStore

REFERENCE_RESULT_VERSION = "flux-reference-result-v1"


class GeneratedImageProtocol(Protocol):
    def save(self, path: Path, *, overwrite: bool) -> None: ...


class ReferenceModelProtocol(Protocol):
    def generate_image(
        self,
        *,
        seed: int,
        prompt: str,
        num_inference_steps: int,
        height: int,
        width: int,
        guidance: float,
        image_paths: list[Path],
        scheduler: str,
        use_kv_cache: bool | None,
    ) -> GeneratedImageProtocol: ...


class SourceModelProtocol(Protocol):
    def generate_image(
        self,
        *,
        seed: int,
        prompt: str,
        num_inference_steps: int,
        height: int,
        width: int,
        guidance: float,
        scheduler: str,
    ) -> GeneratedImageProtocol: ...


ReferenceModelFactory = Callable[[str, int | None], ReferenceModelProtocol]
SourceModelFactory = Callable[[str, int | None], SourceModelProtocol]


def default_reference_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return Path("evals/runs") / f"flux-references-{timestamp}"


def _model_config(model_identifier: str, *, edit: bool) -> object:
    from mflux.models.common.config import ModelConfig

    configurations = {
        "flux2-klein-4b": ModelConfig.flux2_klein_4b,
        "flux2-klein-9b": ModelConfig.flux2_klein_9b,
        "flux2-klein-9b-kv": (
            ModelConfig.flux2_klein_9b_kv if edit else ModelConfig.flux2_klein_9b
        ),
    }
    configuration_factory = configurations.get(model_identifier)
    if configuration_factory is None:
        supported = ", ".join(sorted(configurations))
        raise ValueError(
            f"unsupported reference model {model_identifier!r}; choose one of: {supported}"
        )
    return configuration_factory()


def _default_model_factory(
    model_identifier: str,
    quantization: int | None,
) -> ReferenceModelProtocol:
    from mflux.models.flux2.variants import Flux2KleinEdit

    return Flux2KleinEdit(
        quantize=quantization,
        model_config=_model_config(model_identifier, edit=True),
    )


def _default_source_model_factory(
    model_identifier: str,
    quantization: int | None,
) -> SourceModelProtocol:
    from mflux.models.flux2.variants import Flux2Klein

    return Flux2Klein(
        quantize=quantization,
        model_config=_model_config(model_identifier, edit=False),
    )


def _release_model_cache() -> None:
    collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except (AttributeError, ImportError):
        return


def _resident_bytes() -> int | None:
    try:
        completed = subprocess.run(  # noqa: S603
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return int(completed.stdout.strip()) * 1024
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _peak_resident_bytes() -> int | None:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if value <= 0:
        return None
    return int(value if os.uname().sysname == "Darwin" else value * 1024)


def _validate_png(path: Path, size: tuple[int, int]) -> None:
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.size != size:
            raise ValueError(
                f"generated image must be PNG at {size}, got {image.format} at {image.size}"
            )


def _copy_reference(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        source_size = image.size
    shutil.copyfile(source, destination)
    _validate_png(destination, source_size)
    return destination


def _stack_references(
    bundle_path: Path,
    inputs_dir: Path,
) -> dict[str, dict[str, object]]:
    store = StackStore(bundle_path)
    stack = store.load()
    cards = {card.name.casefold(): card for card in stack.cards}
    try:
        map_card = cards["map"]
        castle_card = cards["castle"]
    except KeyError as error:
        raise ValueError("reference stack must contain Map and Castle cards") from error
    candidates = {
        "map_style": map_card,
        "castle_identity": castle_card,
    }
    references: dict[str, dict[str, object]] = {}
    for key, card in candidates.items():
        revision = card.active_revision
        if revision.background is None:
            raise ValueError(f"{card.name} active revision has no background")
        source = store.asset_path(revision.background.image_path)
        suffix = source.suffix.casefold() or ".png"
        copied = _copy_reference(source, inputs_dir / f"{key}{suffix}")
        references[key] = {
            "path": copied,
            "source_path": source,
            "card_id": str(card.id),
            "revision_id": str(revision.id),
            "background_id": str(revision.background.id),
            "description": revision.description,
        }
    return references


def _case_specs() -> tuple[dict[str, object], ...]:
    return (
        {
            "case_id": "character-identity",
            "reference_keys": ("character_identity",),
            "scene": (
                "Use image 1 for the clockwork knight's helmet silhouette, "
                "blackened armor, teal enamel, brass trim, and red sash. "
                "Show a full-body view of the same knight standing before "
                "the gates of a majestic medieval castle beneath a moody "
                "overcast sky, cinematic storybook illustration."
            ),
            "rubric": (
                "recognizable knight identity",
                "new castle environment",
                "reference background not copied",
            ),
        },
        {
            "case_id": "style-only",
            "reference_keys": ("map_style",),
            "scene": (
                "An intimate clockmaker's workshop interior with a tall blue "
                "cabinet, brass gears on a workbench, and a round window showing "
                "rain. Use image 1 for its hand-drawn ink linework, restrained "
                "watercolor washes, pale blue accents, white paper, and "
                "cross-hatched shading, without copying its subjects."
            ),
            "rubric": (
                "map rendering treatment transferred",
                "workshop subjects present",
                "map subjects absent",
            ),
        },
        {
            "case_id": "character-plus-style",
            "reference_keys": ("character_identity", "map_style"),
            "scene": (
                "Use image 1 for the clockwork knight's helmet silhouette, "
                "blackened armor, teal enamel, brass trim, and red sash. The "
                "same knight explores a moonlit stone courtyard with an arched "
                "wooden gate and a leafy tree, full-body view. Use image 2 for "
                "hand-drawn ink linework, restrained watercolor washes, pale "
                "blue accents, white paper, and cross-hatched shading, without "
                "copying its subjects."
            ),
            "rubric": (
                "knight identity preserved",
                "map treatment transferred",
                "map content absent",
            ),
        },
        {
            "case_id": "building-new-view",
            "reference_keys": ("castle_identity",),
            "scene": (
                "Use image 1 for the castle's recognizable cylindrical corner "
                "towers, conical roofs, central gable, crenellated walls, "
                "proportions, and pencil-sketch appearance. Show the same castle "
                "from its rear garden at ground level, looking toward a servants' "
                "entrance and the backs of the towers."
            ),
            "rubric": (
                "recognizable castle identity",
                "rear viewpoint differs from source",
                "same architectural vocabulary",
            ),
        },
        {
            "case_id": "building-two-views",
            "reference_keys": ("castle_identity", "building_new_view"),
            "scene": (
                "Use image 1 as the authoritative frontal view of the castle and "
                "image 2 as a second view of the same castle. Reconcile their "
                "shared towers, roofs, gable, crenellated walls, and rear layout "
                "in an elevated three-quarter side view at dawn that shows both "
                "the front bridge and part of the rear garden."
            ),
            "rubric": (
                "both views reconciled",
                "castle identity preserved",
                "third viewpoint is distinct",
            ),
        },
        {
            "case_id": "reference-count-1",
            "reference_keys": ("character_identity",),
            "scene": (
                "A storybook market square with the clockwork knight from image 1 "
                "in the foreground and original stalls, townspeople, and architecture."
            ),
            "rubric": ("one reference used",),
        },
        {
            "case_id": "reference-count-2",
            "reference_keys": ("character_identity", "map_style"),
            "scene": (
                "A storybook market square with the clockwork knight from image 1 "
                "in the foreground and original stalls, townspeople, and "
                "architecture. Use image 2 for ink-and-watercolor rendering "
                "treatment without copying its subjects."
            ),
            "rubric": ("two references used",),
        },
    )


def _generate(
    *,
    model: ReferenceModelProtocol,
    output_path: Path,
    prompt: str,
    image_paths: Sequence[Path],
    seed: int,
    width: int,
    height: int,
    step_count: int,
    use_kv_cache: bool | None,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rss_before = _resident_bytes()
    peak_before = _peak_resident_bytes()
    started = perf_counter()
    generated = model.generate_image(
        seed=seed,
        prompt=prompt,
        num_inference_steps=step_count,
        height=height,
        width=width,
        guidance=1.0,
        image_paths=list(image_paths),
        scheduler="flow_match_euler_discrete",
        use_kv_cache=use_kv_cache,
    )
    inference_seconds = perf_counter() - started
    serialization_started = perf_counter()
    generated.save(output_path, overwrite=False)
    serialization_seconds = perf_counter() - serialization_started
    _validate_png(output_path, (width, height))
    rss_after = _resident_bytes()
    peak_after = _peak_resident_bytes()
    return {
        "status": "success",
        "artifact_path": output_path,
        "inference_seconds": inference_seconds,
        "serialization_seconds": serialization_seconds,
        "resident_bytes_before": rss_before,
        "resident_bytes_after": rss_after,
        "resident_bytes_delta": (
            rss_after - rss_before if rss_before is not None and rss_after is not None else None
        ),
        "peak_resident_bytes_before": peak_before,
        "peak_resident_bytes_after": peak_after,
        "peak_resident_bytes_delta": (
            peak_after - peak_before if peak_before is not None and peak_after is not None else None
        ),
    }


def _generate_source(
    *,
    model: SourceModelProtocol,
    output_path: Path,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    step_count: int,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rss_before = _resident_bytes()
    peak_before = _peak_resident_bytes()
    started = perf_counter()
    generated = model.generate_image(
        seed=seed,
        prompt=prompt,
        num_inference_steps=step_count,
        height=height,
        width=width,
        guidance=1.0,
        scheduler="flow_match_euler_discrete",
    )
    inference_seconds = perf_counter() - started
    serialization_started = perf_counter()
    generated.save(output_path, overwrite=False)
    serialization_seconds = perf_counter() - serialization_started
    _validate_png(output_path, (width, height))
    rss_after = _resident_bytes()
    peak_after = _peak_resident_bytes()
    return {
        "status": "success",
        "artifact_path": output_path,
        "inference_seconds": inference_seconds,
        "serialization_seconds": serialization_seconds,
        "resident_bytes_before": rss_before,
        "resident_bytes_after": rss_after,
        "resident_bytes_delta": (
            rss_after - rss_before if rss_before is not None and rss_after is not None else None
        ),
        "peak_resident_bytes_before": peak_before,
        "peak_resident_bytes_after": peak_after,
        "peak_resident_bytes_delta": (
            peak_after - peak_before if peak_before is not None and peak_after is not None else None
        ),
    }


def _model_load_record(
    *,
    started: float,
    rss_before: int | None,
    peak_before: int | None,
) -> dict[str, object]:
    rss_after = _resident_bytes()
    peak_after = _peak_resident_bytes()
    return {
        "duration_seconds": perf_counter() - started,
        "resident_bytes_before": rss_before,
        "resident_bytes_after": rss_after,
        "resident_bytes_delta": (
            rss_after - rss_before if rss_before is not None and rss_after is not None else None
        ),
        "peak_resident_bytes_before": peak_before,
        "peak_resident_bytes_after": peak_after,
        "peak_resident_bytes_delta": (
            peak_after - peak_before if peak_before is not None and peak_after is not None else None
        ),
    }


def run_flux_reference_evaluation(
    *,
    output_dir: Path,
    stack_path: Path,
    model_identifier: str = "flux2-klein-4b",
    quantization: int | None = None,
    seed: int = 42,
    width: int = 1024,
    height: int = 768,
    step_count: int = 4,
    use_kv_cache: bool | None = None,
    model_factory: ReferenceModelFactory = _default_model_factory,
    source_model_factory: SourceModelFactory = _default_source_model_factory,
    environment_provider: EnvironmentProvider = default_environment,
) -> Path:
    """Run supported one- and two-Reference experiments with auditable inputs."""
    settings = {
        "stack_path": str(stack_path),
        "model_identifier": model_identifier,
        "quantization": quantization,
        "seed": seed,
        "width": width,
        "height": height,
        "step_count": step_count,
        "use_kv_cache": use_kv_cache,
    }
    lifecycle = RunLifecycle.create(
        run_dir=output_dir,
        suite="flux-references",
        settings=settings,
        models={
            "mflux_source": model_identifier,
            "mflux_edit": model_identifier,
        },
        contracts={
            "result_version": REFERENCE_RESULT_VERSION,
            "case_ids": [case["case_id"] for case in _case_specs()],
        },
        environment_provider=environment_provider,
    )
    result_path = output_dir / "flux-reference-results.json"
    inputs_dir = output_dir / "inputs"
    outputs_dir = output_dir / "outputs"
    result: dict[str, object] = {
        "result_version": REFERENCE_RESULT_VERSION,
        "suite": "flux-references",
        "status": "running",
        "settings": settings,
        "sources": {},
        "cases": [],
        "warnings": [],
    }
    atomic_write_json(result_path, result)
    try:
        lifecycle.set_stage("resolve-inputs")
        references = _stack_references(stack_path, inputs_dir)
        lifecycle.complete_stage("resolve-inputs")

        character_prompt = (
            "Full-body character reference on a plain warm-gray studio background: "
            "a distinctive clockwork knight with a narrow beaked helmet, blackened "
            "plate armor, teal enamel shoulder plates, brass filigree trim, a red "
            "sash tied at the waist, and a round brass shield. Centered neutral pose, "
            "entire silhouette visible, detailed restrained storybook illustration."
        )
        lifecycle.set_stage("load-source-model")
        source_load_rss = _resident_bytes()
        source_load_peak = _peak_resident_bytes()
        source_load_started = perf_counter()
        source_model = source_model_factory(model_identifier, quantization)
        source_model_load = _model_load_record(
            started=source_load_started,
            rss_before=source_load_rss,
            peak_before=source_load_peak,
        )
        lifecycle.complete_stage("load-source-model")
        lifecycle.set_stage("source:character-identity")
        character_output = inputs_dir / "character_identity.png"
        character_generation = _generate_source(
            model=source_model,
            output_path=character_output,
            prompt=character_prompt,
            seed=seed,
            width=width,
            height=height,
            step_count=step_count,
        )
        del source_model
        _release_model_cache()
        references["character_identity"] = {
            "path": character_output,
            "description": character_prompt,
            "generation": character_generation,
        }
        lifecycle.complete_stage("source:character-identity")

        lifecycle.set_stage("load-edit-model")
        edit_load_rss = _resident_bytes()
        edit_load_peak = _peak_resident_bytes()
        edit_load_started = perf_counter()
        model = model_factory(model_identifier, quantization)
        edit_model_load = _model_load_record(
            started=edit_load_started,
            rss_before=edit_load_rss,
            peak_before=edit_load_peak,
        )
        lifecycle.complete_stage("load-edit-model")

        result["sources"] = {
            key: {
                **{
                    item_key: (
                        value.relative_to(output_dir).as_posix()
                        if isinstance(value, Path) and value.is_relative_to(output_dir)
                        else str(value)
                    )
                    for item_key, value in source.items()
                    if item_key != "generation"
                },
                **(
                    {
                        "generation": {
                            **source["generation"],
                            "artifact_path": character_output.relative_to(output_dir).as_posix(),
                        }
                    }
                    if "generation" in source
                    else {}
                ),
            }
            for key, source in references.items()
        }
        result["model_load"] = {
            "source": source_model_load,
            "edit": edit_model_load,
        }
        atomic_write_json(result_path, result)

        cases: list[dict[str, object]] = result["cases"]  # type: ignore[assignment]
        output_by_case: dict[str, Path] = {}
        for case in _case_specs():
            case_id = str(case["case_id"])
            reference_keys = tuple(case["reference_keys"])
            prompt = compose_generation_prompt(
                GenerateInputs(description=str(case["scene"]))
            )
            stage = f"case:{case_id}"
            lifecycle.set_stage(stage)
            record: dict[str, object] = {
                "case_id": case_id,
                "reference_count": len(reference_keys),
                "reference_keys": list(reference_keys),
                "reference_paths": [],
                "prompt": prompt,
                "scene": case["scene"],
                "rubric": case["rubric"],
            }
            output_path = outputs_dir / f"{case_id}.png"
            try:
                image_paths = [
                    (
                        output_by_case["building-new-view"]
                        if key == "building_new_view"
                        else references[key]["path"]
                    )
                    for key in reference_keys
                ]
                if not all(isinstance(path, Path) for path in image_paths):
                    raise TypeError(f"{case_id} resolved a non-path reference")
                record["reference_paths"] = [
                    path.relative_to(output_dir).as_posix() for path in image_paths
                ]
                generation = _generate(
                    model=model,
                    output_path=output_path,
                    prompt=prompt,
                    image_paths=image_paths,  # type: ignore[arg-type]
                    seed=seed,
                    width=width,
                    height=height,
                    step_count=step_count,
                    use_kv_cache=use_kv_cache,
                )
                generation["artifact_path"] = output_path.relative_to(output_dir).as_posix()
                record["generation"] = generation
                output_by_case[case_id] = output_path
                lifecycle.complete_stage(stage)
            except Exception as error:
                record["generation"] = {
                    "status": "failed",
                    "failure": {
                        "classification": classify_failure(error),
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                }
            cases.append(record)
            atomic_write_json(result_path, result)

        contact_entries = [
            (
                path,
                f"{record['case_id']} · {record['reference_count']} refs",
            )
            for record in cases
            if (
                isinstance(record.get("generation"), dict)
                and record["generation"].get("status") == "success"  # type: ignore[union-attr]
                and (path := output_by_case.get(str(record["case_id"]))) is not None
            )
        ]
        lifecycle.set_stage("contact-sheet")
        create_contact_sheet(
            contact_entries,
            output_dir / "contact-sheet.png",
            columns=3,
        )
        comparisons_dir = output_dir / "comparisons"
        for record in cases:
            output_path = output_by_case.get(str(record["case_id"]))
            if output_path is None:
                continue
            reference_entries = [
                (
                    output_dir / relative_path,
                    f"Reference {index} · {key}",
                )
                for index, (key, relative_path) in enumerate(
                    zip(
                        record["reference_keys"],  # type: ignore[arg-type]
                        record["reference_paths"],  # type: ignore[arg-type]
                        strict=True,
                    ),
                    start=1,
                )
            ]
            create_contact_sheet(
                [
                    *reference_entries,
                    (output_path, f"Output · {record['case_id']}"),
                ],
                comparisons_dir / f"{record['case_id']}.png",
                cell_size=(320, 260),
                columns=len(reference_entries) + 1,
            )
        lifecycle.complete_stage("contact-sheet")
        failures = [
            case
            for case in cases
            if case["generation"]["status"] == "failed"  # type: ignore[index]
        ]
        result["status"] = "completed_with_failures" if failures else "success"
        atomic_write_json(result_path, result)
        lifecycle.finalize(status=str(result["status"]))
        return result_path
    except Exception as error:
        result["status"] = "failed"
        result["failure"] = {
            "classification": classify_failure(error),
            "error_type": type(error).__name__,
            "message": str(error),
        }
        atomic_write_json(result_path, result)
        lifecycle.finalize(
            status="failed",
            failure=result["failure"],  # type: ignore[arg-type]
        )
        raise


__all__ = [
    "REFERENCE_RESULT_VERSION",
    "default_reference_output_dir",
    "run_flux_reference_evaluation",
]
