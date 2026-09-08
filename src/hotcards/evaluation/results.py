"""Current image-generation records shared by evaluation suites."""

from pathlib import Path

from pydantic import ValidationError

from hotcards.domain.models import image_operation_settings
from hotcards.generation.mflux_generator import (
    MfluxGenerateRequest,
    MfluxGenerateResult,
    MfluxGenerator,
)


def generate_cold(
    generator: MfluxGenerator,
    request: MfluxGenerateRequest,
) -> MfluxGenerateResult:
    """Request a cold load atomically inside the adapter's native invocation."""
    return generator.generate(request.model_copy(update={"force_reload": True}))


def generation_record(
    generated: MfluxGenerateResult,
    output_dir: Path,
) -> dict[str, object]:
    """Retain portable output links, exact provenance, and adapter timings."""
    try:
        payload = generated.model_dump(mode="json")
    except ValidationError:
        generated.dispose_output()
        raise
    return {
        "status": "success",
        "cache_reused": generated.cache_reused,
        "artifact_path": generated.output_path.relative_to(output_dir).as_posix(),
        "queue_duration_seconds": generated.queue_duration_seconds,
        "load_duration_seconds": generated.load_duration_seconds,
        "inference_duration_seconds": generated.generation_duration_seconds,
        "serialization_duration_seconds": generated.serialization_duration_seconds,
        "total_duration_seconds": image_operation_settings(generated.provenance).duration_seconds,
        "metadata": payload["provenance"],
    }
