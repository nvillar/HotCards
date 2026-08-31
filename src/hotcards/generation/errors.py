"""Actionable failures raised by local generation adapters."""


class GenerationError(RuntimeError):
    """Base class for expected local-generation failures."""


class ModelUnavailableError(GenerationError):
    """A configured local model is not available or lacks a required capability."""


class ModelLoadError(GenerationError):
    """An in-process model could not be loaded."""


class ImageGenerationError(GenerationError):
    """MFLUX failed while generating or saving an image."""


class ImageGenerationCancelled(GenerationError):
    """A queued or active MFLUX image operation was explicitly cancelled."""
