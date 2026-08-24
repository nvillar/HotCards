"""Local model prompt builders and adapters."""

from hypergen.generation.errors import (
    GenerationError,
    ImageGenerationError,
    ModelLoadError,
    ModelResponseError,
    ModelUnavailableError,
    ServiceUnavailableError,
)
from hypergen.generation.scene_enrichment import (
    OllamaSceneEnricher,
    SceneEnrichmentRequest,
    SceneEnrichmentResult,
)

__all__ = [
    "GenerationError",
    "ImageGenerationError",
    "ModelLoadError",
    "ModelResponseError",
    "ModelUnavailableError",
    "OllamaSceneEnricher",
    "SceneEnrichmentRequest",
    "SceneEnrichmentResult",
    "ServiceUnavailableError",
]
