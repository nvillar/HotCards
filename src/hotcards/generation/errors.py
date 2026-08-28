"""Actionable failures raised by local generation adapters."""


class GenerationError(RuntimeError):
    """Base class for expected local-generation failures."""


class ServiceUnavailableError(GenerationError):
    """A separately managed local service could not be reached."""


class ModelUnavailableError(GenerationError):
    """A configured local model is not available or lacks a required capability."""


class ModelResponseError(GenerationError):
    """A model response failed the production structured-output contract."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: str | None = None,
        response_metadata: dict[str, int | str | float | None] | None = None,
        response_attempts: tuple[dict[str, object], ...] = (),
    ) -> None:
        self.raw_response = raw_response
        self.response_metadata = response_metadata or {}
        self.response_attempts = response_attempts
        super().__init__(message)


class ModelLoadError(GenerationError):
    """An in-process model could not be loaded."""


class ImageGenerationError(GenerationError):
    """MFLUX failed while generating or saving an image."""
