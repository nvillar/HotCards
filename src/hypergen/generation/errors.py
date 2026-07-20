"""Actionable failures raised by local generation adapters."""


class GenerationError(RuntimeError):
    """Base class for expected local-generation failures."""


class ServiceUnavailableError(GenerationError):
    """A separately managed local service could not be reached."""


class ModelUnavailableError(GenerationError):
    """A configured local model is not available or lacks a required capability."""


class ModelResponseError(GenerationError):
    """A model response failed the production structured-output contract."""


class ModelLoadError(GenerationError):
    """An in-process model could not be loaded."""


class ImageGenerationError(GenerationError):
    """MFLUX failed while generating or saving an image."""
