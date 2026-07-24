"""Document commands, controllers, and workers."""

from hypergen.application.background_workflow import (
    BackgroundDraft,
    BackgroundGenerationSettings,
    BackgroundWorkflow,
    BackgroundWorkflowError,
)
from hypergen.application.commands import (
    ActivateRevisionCommand,
    AddImageRevisionCommand,
    ChangeHotspotDestinationCommand,
    CommandError,
    CreateCardAndResolveCommand,
    CreateCardCommand,
    DeleteCardCommand,
    DeleteImageRevisionCommand,
    DocumentCommand,
    EditCardTextCommand,
    RenameCardCommand,
    ReorderCardCommand,
    ReorderHotspotCommand,
    ReplaceHotspotSetCommand,
    ReplaceInteractionPolygonsCommand,
    ReplacePolygonCommand,
    SetRunOverlayModeCommand,
    SetStartCardCommand,
)
from hypergen.application.document_controller import AutosaveHook, DocumentController
from hypergen.application.document_session import (
    DocumentSession,
    DocumentSessionError,
    DocumentSessionState,
)
from hypergen.application.hotspot_generation_workflow import (
    HotspotGenerationDraft,
    HotspotGenerationWorkflow,
    HotspotGenerationWorkflowError,
)
from hypergen.application.image_description_workflow import (
    ImageDescriptionWorkflow,
    ImageDescriptionWorkflowError,
)
from hypergen.application.scene_enrichment_workflow import (
    SceneEnrichmentDraft,
    SceneEnrichmentWorkflow,
    SceneEnrichmentWorkflowError,
)
from hypergen.application.workers import (
    AdapterKind,
    AdapterWorkers,
    AvailabilityDiagnostic,
    OperationStatus,
    WorkerFailure,
    WorkerFailureKind,
    WorkerOperation,
)

__all__ = [
    "ActivateRevisionCommand",
    "AddImageRevisionCommand",
    "AdapterKind",
    "AdapterWorkers",
    "AutosaveHook",
    "AvailabilityDiagnostic",
    "BackgroundDraft",
    "BackgroundGenerationSettings",
    "BackgroundWorkflow",
    "BackgroundWorkflowError",
    "ChangeHotspotDestinationCommand",
    "CommandError",
    "CreateCardAndResolveCommand",
    "CreateCardCommand",
    "DeleteCardCommand",
    "DeleteImageRevisionCommand",
    "DocumentCommand",
    "DocumentController",
    "DocumentSession",
    "DocumentSessionError",
    "DocumentSessionState",
    "EditCardTextCommand",
    "HotspotGenerationDraft",
    "HotspotGenerationWorkflow",
    "HotspotGenerationWorkflowError",
    "ImageDescriptionWorkflow",
    "ImageDescriptionWorkflowError",
    "OperationStatus",
    "RenameCardCommand",
    "ReorderCardCommand",
    "ReorderHotspotCommand",
    "ReplaceHotspotSetCommand",
    "ReplaceInteractionPolygonsCommand",
    "ReplacePolygonCommand",
    "SetRunOverlayModeCommand",
    "SetStartCardCommand",
    "SceneEnrichmentDraft",
    "SceneEnrichmentWorkflow",
    "SceneEnrichmentWorkflowError",
    "WorkerFailure",
    "WorkerFailureKind",
    "WorkerOperation",
]
