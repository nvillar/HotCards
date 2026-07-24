"""Local model prompt builders and adapters."""

from hypergen.generation.errors import (
    GenerationError,
    ImageGenerationError,
    ModelLoadError,
    ModelResponseError,
    ModelUnavailableError,
    ServiceUnavailableError,
)
from hypergen.generation.image_description import (
    ImageDescriptionRequest,
    ImageDescriptionResult,
    OllamaImageDescriber,
)
from hypergen.generation.scene_enrichment import (
    OllamaSceneEnricher,
    SceneEnrichmentRequest,
    SceneEnrichmentResult,
)

__all__ = [
    "GenerationError",
    "ImageDescriptionRequest",
    "ImageDescriptionResult",
    "ImageGenerationError",
    "ModelLoadError",
    "ModelResponseError",
    "ModelUnavailableError",
    "OllamaImageDescriber",
    "OllamaSceneEnricher",
    "SceneEnrichmentRequest",
    "SceneEnrichmentResult",
    "ServiceUnavailableError",
]
