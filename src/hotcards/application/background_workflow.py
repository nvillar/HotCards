"""Direct, revision-local background image workflow."""

from __future__ import annotations

import logging
import secrets
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from threading import Event, Lock
from typing import Literal
from uuid import UUID, uuid4

from pydantic import ValidationError
from PySide6.QtCore import QObject, Signal

from hotcards.application.commands import (
    ActivateRevisionCommand,
    CommandError,
    CreateEditedRevisionCommand,
    CreateRefinedRevisionCommand,
    DeleteRevisionCommand,
    DuplicateRevisionCommand,
    ReplaceRevisionBackgroundCommand,
)
from hotcards.application.document_controller import (
    DocumentController,
    DocumentMutationBlockedError,
    OwnedImageAsset,
    UndoToken,
)
from hotcards.application.document_session import (
    DocumentSession,
    DocumentSessionError,
    DocumentSessionState,
)
from hotcards.application.generated_revision_change import GeneratedRevisionChange
from hotcards.application.image_files import (
    UnreadableImageError,
    require_readable_image,
)
from hotcards.application.workers import AdapterWorkers, WorkerOperation
from hotcards.domain.image_dependencies import image_source_dependencies
from hotcards.domain.image_dimensions import (
    AspectRatio,
    GenerateResolution,
    higher_output_resolutions,
    output_dimensions,
)
from hotcards.domain.models import (
    AcceptedEdit,
    Card,
    CardRevision,
    CurrentSourceSize,
    EditOutputSize,
    EditPreserveOptions,
    GeneratedBackground,
    GenerateInputs,
    ImageReferenceSnapshot,
    ImageSourceSnapshot,
    PresetOutputSize,
    RefineTransformation,
    ResolvedCardReference,
    Stack,
    StyleSnapshot,
    UnresolvedCardReference,
    image_edit_lineage,
    image_operation_settings,
)
from hotcards.generation.image_generation import (
    compose_edit_prompt,
    compose_generation_prompt,
    compose_refine_prompt,
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
    dispose_mflux_result,
)
from hotcards.storage.stack_store import (
    StackStore,
    StackStoreError,
    StackStoreTransactionError,
    StoredImageAsset,
    StoredImageSnapshot,
)

logger = logging.getLogger(__name__)


class BackgroundWorkflowError(ValueError):
    """A background operation cannot proceed without losing user intent."""


@dataclass(frozen=True, slots=True)
class BackgroundGenerationSettings:
    """Machine-local effective settings captured before worker submission."""

    mflux_model: str
    step_count: int
    quantization: int | None
    random_seed: bool
    fixed_seed: int


@dataclass(frozen=True, slots=True)
class _GenerationTarget:
    stack_id: UUID
    card_id: UUID
    revision_id: UUID
    description: str
    background_id: UUID | None
    bundle_path: Path
    references: tuple[_GenerationReferenceTarget, ...]
    style: StyleSnapshot | None
    generate_resolution: GenerateResolution


@dataclass(frozen=True, slots=True)
class _GenerationReferenceTarget:
    card_id: UUID
    card_name: str = field(compare=False)
    revision_id: UUID
    background_id: UUID
    image_path: str


@dataclass(frozen=True, slots=True)
class _RefineTarget:
    stack_id: UUID
    card_id: UUID
    revision: CardRevision
    bundle_path: Path
    style: StyleSnapshot | None
    edit_lineage: tuple[AcceptedEdit, ...]
    aspect_ratio: AspectRatio
    resolution: GenerateResolution
    transformation: RefineTransformation
    settings: BackgroundGenerationSettings
    history_token: UndoToken | None
    source_snapshot: StoredImageSnapshot
    source_background_id: UUID


@dataclass(frozen=True, slots=True)
class _EditTarget:
    stack_id: UUID
    card_id: UUID
    revision: CardRevision
    bundle_path: Path
    instruction: str
    preserve: EditPreserveOptions
    expanded_prompt: str
    style: StyleSnapshot | None
    edit_lineage: tuple[AcceptedEdit, ...]
    aspect_ratio: AspectRatio
    output_size: EditOutputSize
    settings: BackgroundGenerationSettings
    history_token: UndoToken | None
    source_snapshot: StoredImageSnapshot
    source_background_id: UUID


@dataclass(frozen=True, slots=True)
class _PendingEditCompletion:
    document: Stack
    previous_token: UndoToken | None


GenerationSettingsProvider = Callable[[], BackgroundGenerationSettings]


class BackgroundWorkflow(QObject):
    """Generate, clear, and manage complete card revisions."""

    busy_changed = Signal(bool)
    invocation_active_changed = Signal(bool)
    progress_changed = Signal(str)
    generation_progress_changed = Signal(int, int)
    failed = Signal(object)
    document_changed = Signal(object)
    change_applied = Signal(str, object)
    generation_applied = Signal(object)
    edit_instruction_clear_requested = Signal()

    def __init__(
        self,
        controller: DocumentController,
        session: DocumentSession,
        workers: AdapterWorkers,
        settings_provider: GenerationSettingsProvider,
        *,
        mflux_generator: MfluxGenerator | None = None,
        temporary_directory: Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.session = session
        self.workers = workers
        self._settings_provider = settings_provider
        self._mflux_generator = mflux_generator or MfluxGenerator()
        self._owned_temporary_directory = (
            tempfile.TemporaryDirectory(prefix="hotcards-background-")
            if temporary_directory is None
            else None
        )
        self._temporary_directory = (
            Path(self._owned_temporary_directory.name)
            if self._owned_temporary_directory is not None
            else temporary_directory
        )
        assert self._temporary_directory is not None
        self._temporary_directory.mkdir(parents=True, exist_ok=True)
        self._operation: WorkerOperation | None = None
        self._request_id: UUID | None = None
        self._request_target: _GenerationTarget | _RefineTarget | _EditTarget | None = None
        self._pending_result: (
            MfluxGenerateResult | MfluxRefineResult | MfluxEditResult | None
        ) = None
        self._close_requested = Event()
        self._invocation_active = Event()
        self._source_snapshot_lock = Lock()
        self._source_snapshot: StoredImageSnapshot | None = None
        self._snapshot_cleanup_in_progress = False
        self._temporary_cleanup_blocked = False
        self._busy = False
        self._active_operation: Literal["generate", "refine", "edit"] | None = None
        self._pending_edit_completion: _PendingEditCompletion | None = None
        self.session.state_changed.connect(self._session_state_changed)

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def active_operation(self) -> Literal["generate", "refine", "edit"] | None:
        return self._active_operation

    @property
    def invocation_active(self) -> bool:
        with self._source_snapshot_lock:
            return (
                self._invocation_active.is_set()
                or self._snapshot_cleanup_in_progress
                or (
                    self._source_snapshot is not None
                    and self._temporary_cleanup_blocked
                )
            )

    def generate(self, card_id: UUID) -> WorkerOperation:
        """Generate and directly apply a background to the active revision."""
        self._require_ready(card_id)
        document = self.controller.document
        card = self._card(document, card_id)
        revision = card.active_revision
        if not revision.description.strip():
            raise BackgroundWorkflowError("enter a Description before generating")
        if revision.background is not None:
            dependencies = image_source_dependencies(document, (revision.id,))
            if dependencies:
                dependent = dependencies[0]
                raise BackgroundWorkflowError(
                    "cannot replace this source background because "
                    f"{dependent.operation.title()} revision "
                    f"{dependent.dependent_revision_number} on card "
                    f'"{dependent.dependent_card_name}" derives from it'
                )
        references = self._resolve_references(document, card)
        reference_snapshots = tuple(
            ImageReferenceSnapshot(
                card_id=reference.card_id,
                revision_id=reference.revision_id,
                background_id=reference.background_id,
            )
            for reference in references
        )
        settings = self._settings_provider()
        inputs = GenerateInputs(
            description=revision.description,
            references=reference_snapshots,
            style=self._style_snapshot(document, revision),
            resolution=revision.generate_resolution,
        )
        render_prompt = compose_generation_prompt(inputs)
        reference_image_paths = tuple(
            self._reference_asset_path(reference, position)
            for position, reference in enumerate(references, start=1)
        )
        request_id = uuid4()
        asset_id = uuid4()
        target = self._target(document, card, references)
        self._request_id = request_id
        self._request_target = target
        output_path = self._temporary_directory / f"generated-{asset_id}.png"
        width, height = output_dimensions(
            revision.generate_resolution,
            document.aspect_ratio,
        )
        request = MfluxGenerateRequest(
            inputs=inputs,
            render_prompt=render_prompt,
            output_path=output_path,
            model_identifier=settings.mflux_model,
            seed=(
                secrets.randbelow(2_147_483_648) if settings.random_seed else settings.fixed_seed
            ),
            aspect_ratio=document.aspect_ratio,
            width=width,
            height=height,
            step_count=settings.step_count,
            quantization=settings.quantization,
            reference_image_paths=reference_image_paths,
        )
        cancellation = MfluxCancellationToken()
        self._active_operation = "generate"
        self._set_busy(True, "Generating image...")
        operation = self.workers.run_mflux(
            lambda: self._mflux_generator.generate(
                request,
                progress=partial(
                    self._generation_progress,
                    request_id,
                ),
                cancellation=cancellation,
            ),
            stage="generating background image",
            request_cancel=cancellation.cancel,
            dispose_result=dispose_mflux_result,
            invocation_started=self._invocation_started,
            invocation_finished=self._invocation_finished,
        )
        self._operation = operation
        operation.succeeded.connect(
            partial(self._generation_succeeded, request_id, target, asset_id)
        )
        operation.failed.connect(partial(self._operation_failed, request_id))
        return operation

    def available_refine_resolutions(
        self,
        card_id: UUID,
    ) -> tuple[GenerateResolution, ...]:
        """Return output presets larger than the current decoded background."""
        card = self._card(self.controller.document, card_id)
        revision = card.active_revision
        background = revision.background
        if background is None:
            return ()
        try:
            width, height = self._require_store().image_asset_dimensions(
                background.image_path,
                card_id=card.id,
                asset_id=background.id,
            )
        except StackStoreError as error:
            raise BackgroundWorkflowError(
                "the current image is unavailable or unreadable"
            ) from error
        return higher_output_resolutions(
            width,
            height,
            self.controller.document.aspect_ratio,
        )

    def refine(
        self,
        card_id: UUID,
        *,
        transformation: RefineTransformation,
        resolution: GenerateResolution,
    ) -> WorkerOperation:
        """Refine the current image into one automatic complete revision."""
        self._require_ready(card_id)
        if not self.session.flush():
            raise BackgroundWorkflowError(
                self.session.state.error or "the current stack could not be saved"
            )
        document = self.controller.document
        card = self._card(document, card_id)
        revision = card.active_revision
        if not revision.description.strip():
            raise BackgroundWorkflowError("enter a Description before refining")
        background = revision.background
        if background is None:
            raise BackgroundWorkflowError("generate an image before refining")
        store = self._require_store()
        try:
            source_snapshot = store.snapshot_image_asset(
                background.image_path,
                card_id=card.id,
                asset_id=background.id,
                destination_directory=self._temporary_directory,
            )
        except StackStoreError as error:
            raise BackgroundWorkflowError(
                "the current image is unavailable or unreadable"
            ) from error
        try:
            self._track_source_snapshot(source_snapshot)
        except Exception:
            if not source_snapshot.dispose():
                logger.warning(
                    "Refine source snapshot cleanup preserved a changed file: %s",
                    source_snapshot.snapshot_path,
                )
            raise
        available_resolutions = higher_output_resolutions(
            source_snapshot.width,
            source_snapshot.height,
            document.aspect_ratio,
        )
        if not available_resolutions:
            self._cleanup_source_snapshot_if_idle()
            raise BackgroundWorkflowError(
                "the current image is already at the maximum Refine resolution"
            )
        if resolution not in available_resolutions:
            self._cleanup_source_snapshot_if_idle()
            raise BackgroundWorkflowError(
                "select a Refine resolution with more pixels than the current image"
            )
        try:
            settings = self._settings_provider()
        except Exception:
            self._cleanup_source_snapshot_if_idle()
            raise
        style = self._style_snapshot(document, revision)
        edit_lineage = image_edit_lineage(background.provenance)
        render_prompt = compose_refine_prompt(
            revision.description,
            style,
            edit_lineage,
        )
        source = ImageSourceSnapshot(
            card_id=card.id,
            revision_id=revision.id,
            background_id=background.id,
        )
        width, height = output_dimensions(resolution, document.aspect_ratio)
        request_id = uuid4()
        asset_id = uuid4()
        target = _RefineTarget(
            stack_id=document.id,
            card_id=card.id,
            revision=revision.model_copy(deep=True),
            bundle_path=store.bundle_path.resolve(),
            style=style,
            edit_lineage=edit_lineage,
            aspect_ratio=document.aspect_ratio,
            resolution=resolution,
            transformation=transformation,
            settings=settings,
            history_token=self.controller.current_undo_token,
            source_snapshot=source_snapshot,
            source_background_id=background.id,
        )
        request = MfluxRefineRequest(
            source=source,
            source_image_path=source_snapshot.snapshot_path,
            source_seed=image_operation_settings(background.provenance).seed,
            description=revision.description,
            style=style,
            edit_lineage=edit_lineage,
            render_prompt=render_prompt,
            resolution=resolution,
            transformation=transformation,
            image_strength=transformation.strength,
            output_path=self._temporary_directory / f"refined-{asset_id}.png",
            model_identifier=settings.mflux_model,
            aspect_ratio=document.aspect_ratio,
            width=width,
            height=height,
            step_count=settings.step_count,
            quantization=settings.quantization,
        )
        cancellation = MfluxCancellationToken()
        self._request_id = request_id
        self._request_target = target
        self._active_operation = "refine"
        self._set_busy(True, "Refining image...")
        try:
            operation = self.workers.run_mflux(
                lambda: self._mflux_generator.refine(
                    request,
                    progress=partial(
                        self._generation_progress,
                        request_id,
                    ),
                    cancellation=cancellation,
                ),
                stage="refining background image",
                request_cancel=cancellation.cancel,
                dispose_result=dispose_mflux_result,
                invocation_started=self._invocation_started,
                invocation_finished=self._invocation_finished,
            )
        except Exception:
            self._request_id = None
            self._request_target = None
            self._active_operation = None
            self._cleanup_source_snapshot_if_idle()
            self._set_busy(False, "Image refinement failed")
            raise
        self._operation = operation
        operation.succeeded.connect(partial(self._refine_succeeded, request_id, target, asset_id))
        operation.failed.connect(partial(self._operation_failed, request_id))
        return operation

    def available_edit_output_sizes(
        self,
        card_id: UUID,
    ) -> tuple[EditOutputSize, ...]:
        """Return exact current size followed by strictly larger presets."""
        document = self.controller.document
        card = self._card(document, card_id)
        background = card.active_revision.background
        if background is None:
            return ()
        try:
            width, height = self._require_store().image_asset_dimensions(
                background.image_path,
                card_id=card.id,
                asset_id=background.id,
            )
        except StackStoreError as error:
            raise BackgroundWorkflowError(
                "the current image is unavailable or unreadable"
            ) from error
        return (
            CurrentSourceSize(width=width, height=height),
            *(
                PresetOutputSize(resolution=resolution)
                for resolution in higher_output_resolutions(
                    width,
                    height,
                    document.aspect_ratio,
                )
            ),
        )

    def edit(
        self,
        card_id: UUID,
        *,
        instruction: str,
        preserve: EditPreserveOptions,
        output_size: EditOutputSize,
    ) -> WorkerOperation:
        """Edit the current image into one automatic complete revision."""
        self._require_ready(card_id)
        if not self.session.flush():
            raise BackgroundWorkflowError(
                self.session.state.error
                or "the current stack could not be saved"
            )
        normalized_instruction = instruction.strip()
        if not normalized_instruction:
            raise BackgroundWorkflowError(
                "enter an Edit Instruction before editing"
            )
        document = self.controller.document
        card = self._card(document, card_id)
        revision = card.active_revision
        background = revision.background
        if background is None:
            raise BackgroundWorkflowError("generate an image before editing")
        store = self._require_store()
        try:
            source_snapshot = store.snapshot_image_asset(
                background.image_path,
                card_id=card.id,
                asset_id=background.id,
                destination_directory=self._temporary_directory,
            )
        except StackStoreError as error:
            raise BackgroundWorkflowError(
                "the current image is unavailable or unreadable"
            ) from error
        try:
            self._track_source_snapshot(source_snapshot)
        except Exception:
            if not source_snapshot.dispose():
                logger.warning(
                    "Edit source snapshot cleanup preserved a changed file: %s",
                    source_snapshot.snapshot_path,
                )
            raise
        available_output_sizes = (
            CurrentSourceSize(
                width=source_snapshot.width,
                height=source_snapshot.height,
            ),
            *(
                PresetOutputSize(resolution=resolution)
                for resolution in higher_output_resolutions(
                    source_snapshot.width,
                    source_snapshot.height,
                    document.aspect_ratio,
                )
            ),
        )
        if output_size not in available_output_sizes:
            self._cleanup_source_snapshot_if_idle()
            raise BackgroundWorkflowError(
                "select the current Edit size or a preset with more pixels"
            )
        try:
            settings = self._settings_provider()
            expanded_prompt = compose_edit_prompt(
                normalized_instruction,
                preserve,
            )
        except Exception:
            self._cleanup_source_snapshot_if_idle()
            raise
        inherited_lineage = image_edit_lineage(background.provenance)
        accepted_edit = AcceptedEdit(
            instruction=normalized_instruction,
            preserve=preserve,
            expanded_prompt=expanded_prompt,
        )
        edit_lineage = (*inherited_lineage, accepted_edit)
        width, height = (
            (output_size.width, output_size.height)
            if isinstance(output_size, CurrentSourceSize)
            else output_dimensions(
                output_size.resolution,
                document.aspect_ratio,
            )
        )
        source = ImageSourceSnapshot(
            card_id=card.id,
            revision_id=revision.id,
            background_id=background.id,
        )
        request_id = uuid4()
        asset_id = uuid4()
        target = _EditTarget(
            stack_id=document.id,
            card_id=card.id,
            revision=revision.model_copy(deep=True),
            bundle_path=store.bundle_path.resolve(),
            instruction=normalized_instruction,
            preserve=preserve,
            expanded_prompt=expanded_prompt,
            style=self._style_snapshot(document, revision),
            edit_lineage=edit_lineage,
            aspect_ratio=document.aspect_ratio,
            output_size=output_size,
            settings=settings,
            history_token=self.controller.current_undo_token,
            source_snapshot=source_snapshot,
            source_background_id=background.id,
        )
        request = MfluxEditRequest(
            source=source,
            source_image_path=source_snapshot.snapshot_path,
            instruction=normalized_instruction,
            preserve=preserve,
            expanded_prompt=expanded_prompt,
            output_size=output_size,
            edit_lineage=edit_lineage,
            seed=secrets.randbelow(2_147_483_648),
            output_path=self._temporary_directory
            / f"edited-{asset_id}.png",
            model_identifier=settings.mflux_model,
            aspect_ratio=document.aspect_ratio,
            width=width,
            height=height,
            step_count=settings.step_count,
            quantization=settings.quantization,
        )
        cancellation = MfluxCancellationToken()
        self._request_id = request_id
        self._request_target = target
        self._active_operation = "edit"
        self._set_busy(True, "Editing image...")
        try:
            operation = self.workers.run_mflux(
                lambda: self._mflux_generator.edit(
                    request,
                    progress=partial(
                        self._generation_progress,
                        request_id,
                    ),
                    cancellation=cancellation,
                ),
                stage="editing background image",
                request_cancel=cancellation.cancel,
                dispose_result=dispose_mflux_result,
                invocation_started=self._invocation_started,
                invocation_finished=self._invocation_finished,
            )
        except Exception:
            self._request_id = None
            self._request_target = None
            self._active_operation = None
            self._cleanup_source_snapshot_if_idle()
            self._set_busy(False, "Image editing failed")
            raise
        self._operation = operation
        operation.succeeded.connect(
            partial(self._edit_succeeded, request_id, target, asset_id)
        )
        operation.failed.connect(partial(self._operation_failed, request_id))
        return operation

    def _generation_progress(
        self,
        request_id: UUID,
        completed_steps: int,
        total_steps: int,
    ) -> None:
        if request_id == self._request_id:
            self.generation_progress_changed.emit(
                completed_steps,
                total_steps,
            )

    def clear_background(self, card_id: UUID) -> Stack:
        """Clear only the active revision's background."""
        card = self._card(self.controller.document, card_id)
        if card.active_revision.background is None:
            raise BackgroundWorkflowError("this revision has no image to clear")
        return self._apply_background(
            card.id,
            card.active_revision.id,
            None,
            "Image cleared",
        )

    def duplicate_revision(self, card_id: UUID, revision_id: UUID) -> Stack:
        previous_token = self.controller.current_undo_token
        changed = self.controller.execute(
            DuplicateRevisionCommand(
                card_id=card_id,
                source_revision_id=revision_id,
            )
        )
        self.progress_changed.emit("Revision duplicated")
        self.document_changed.emit(changed)
        self._emit_change_applied("Revision duplicated", previous_token)
        return changed

    def activate_revision(self, card_id: UUID, revision_id: UUID) -> Stack:
        changed = self.controller.execute(
            ActivateRevisionCommand(card_id=card_id, revision_id=revision_id)
        )
        self.progress_changed.emit("Revision activated")
        self.document_changed.emit(changed)
        return changed

    def delete_revision(self, card_id: UUID, revision_id: UUID) -> Stack:
        previous_token = self.controller.current_undo_token
        changed = self.controller.execute(
            DeleteRevisionCommand(card_id=card_id, revision_id=revision_id)
        )
        self.progress_changed.emit("Revision deleted")
        self.document_changed.emit(changed)
        self._emit_change_applied("Revision deleted", previous_token)
        return changed

    def cancel(self) -> None:
        if self._operation is not None and not self._operation.is_finished:
            self._operation.cancel()
        self._mflux_generator.release()
        self._request_id = None
        self._request_target = None
        self._operation = None
        self._discard_pending_image()
        self._cleanup_source_snapshot_if_idle()
        if self._busy:
            operation = self._active_operation
            self._active_operation = None
            self._set_busy(
                False,
                (
                    "Refine cancelled"
                    if operation == "refine"
                    else (
                        "Edit cancelled"
                        if operation == "edit"
                        else "Generation cancelled"
                    )
                ),
            )

    def close(self) -> None:
        self._close_requested.set()
        self.cancel()
        if (
            self._owned_temporary_directory is not None
            and not self._invocation_active.is_set()
            and not self._temporary_cleanup_blocked
        ):
            self._owned_temporary_directory.cleanup()

    def _invocation_finished(self) -> None:
        self._invocation_active.clear()
        self._cleanup_source_snapshot_if_idle()
        self.invocation_active_changed.emit(self.invocation_active)
        if not self._close_requested.is_set():
            return
        self._mflux_generator.release()
        if (
            self._owned_temporary_directory is not None
            and not self._temporary_cleanup_blocked
        ):
            self._owned_temporary_directory.cleanup()

    def is_generating_for(self, card_id: UUID) -> bool:
        return (
            self._busy
            and self._active_operation == "generate"
            and self._request_target is not None
            and self._request_target.card_id == card_id
        )

    def is_refining_for(self, card_id: UUID) -> bool:
        return (
            self._busy
            and self._active_operation == "refine"
            and self._request_target is not None
            and self._request_target.card_id == card_id
        )

    def is_editing_for(self, card_id: UUID) -> bool:
        return (
            self._busy
            and self._active_operation == "edit"
            and self._request_target is not None
            and self._request_target.card_id == card_id
        )

    def _generation_succeeded(
        self,
        request_id: UUID,
        target: _GenerationTarget,
        asset_id: UUID,
        result: object,
    ) -> None:
        if request_id != self._request_id:
            if isinstance(result, MfluxGenerateResult):
                result.dispose_output()
            return
        if not isinstance(result, MfluxGenerateResult):
            self._finish_with_error(
                BackgroundWorkflowError("image generation returned an unexpected result")
            )
            return
        self._pending_result = result
        if not self._target_is_current(target):
            self._finish_with_error(
                BackgroundWorkflowError(
                    "the stack, revision, Description, or image changed before generation completed"
                )
            )
            return
        stored_image_path: str | None = None
        store = self._require_store()
        try:
            stored_image_path = store.store_image_asset(
                result.output_path,
                card_id=target.card_id,
                asset_id=asset_id,
            )
            stored_asset = store.stored_image_asset(
                stored_image_path,
                card_id=target.card_id,
                asset_id=asset_id,
            )
            background = GeneratedBackground(
                id=asset_id,
                image_path=stored_image_path,
                provenance=result.provenance,
                created_at=result.provenance.settings.generated_at,
            )
            self._apply_background(
                target.card_id,
                target.revision_id,
                background,
                "Image generated",
                generated=True,
                owned_assets=(
                    self._owned_asset(
                        store,
                        target.card_id,
                        asset_id,
                        stored_asset,
                    ),
                ),
            )
        except (
            CommandError,
            DocumentMutationBlockedError,
            StackStoreError,
            ValidationError,
        ) as error:
            if stored_image_path is not None:
                try:
                    store.remove_image_asset_if_unreferenced(
                        stored_image_path,
                        card_id=target.card_id,
                        asset_id=asset_id,
                        stack=self.controller.document,
                    )
                except StackStoreError as cleanup_error:
                    error = BackgroundWorkflowError(
                        f"{error}; could not roll back generated asset: {cleanup_error}"
                    )
            self._finish_with_error(error)
            return
        self._pending_result = None
        result.dispose_output()
        self._operation = None
        self._request_id = None
        self._request_target = None
        self._active_operation = None
        self._set_busy(False, "Image generated")

    def _refine_succeeded(
        self,
        request_id: UUID,
        target: _RefineTarget,
        asset_id: UUID,
        result: object,
    ) -> None:
        if request_id != self._request_id:
            if isinstance(result, MfluxRefineResult):
                result.dispose_output()
            return
        if not isinstance(result, MfluxRefineResult):
            self._finish_with_error(
                BackgroundWorkflowError("image refinement returned an unexpected result")
            )
            return
        self._pending_result = result
        if not self._refine_target_is_current(target):
            self._finish_with_error(
                BackgroundWorkflowError(
                    "the stack or source revision changed before Refine completed"
                )
            )
            return
        store = self._require_store()
        image_path = store.image_asset_path(target.card_id, asset_id)
        background = GeneratedBackground(
            id=asset_id,
            image_path=image_path,
            provenance=result.provenance,
            created_at=result.provenance.settings.generated_at,
        )
        command = CreateRefinedRevisionCommand(
            card_id=target.card_id,
            source_revision_id=target.revision.id,
            background=background,
        )
        before = self.controller.document
        owned_assets: list[OwnedImageAsset] = []

        def persist(candidate: Stack) -> None:
            try:
                stored = store.store_image_asset_and_save(
                    result.output_path,
                    destination_card_id=target.card_id,
                    destination_asset_id=asset_id,
                    previous_stack=before,
                    changed_stack=candidate,
                    expected_source_snapshot=target.source_snapshot,
                    expected_source_card_id=target.card_id,
                    expected_source_asset_id=target.source_background_id,
                )
            except StackStoreTransactionError as error:
                if error.owned_asset is not None:
                    owned_assets.append(
                        self._owned_asset(
                            store,
                            target.card_id,
                            asset_id,
                            error.owned_asset,
                        )
                    )
                raise
            owned_assets.append(
                self._owned_asset(
                    store,
                    target.card_id,
                    asset_id,
                    stored,
                )
            )

        previous_token = self.controller.current_undo_token
        try:
            changed = store.run_locked(
                lambda: self.session.execute_persisted(
                    command,
                    persist=persist,
                    owned_assets=owned_assets,
                )
            )
        except (
            CommandError,
            DocumentMutationBlockedError,
            DocumentSessionError,
            StackStoreError,
            ValidationError,
        ) as error:
            if self.controller.document != before:
                self.document_changed.emit(self.controller.document)
            self._finish_with_error(error)
            return
        self.progress_changed.emit("Image refined")
        self.document_changed.emit(changed)
        self._emit_change_applied("Image refined", previous_token)
        self._pending_result = None
        result.dispose_output()
        self._operation = None
        self._request_id = None
        self._request_target = None
        self._active_operation = None
        self._cleanup_source_snapshot_if_idle()
        self._set_busy(False, "Image refined")

    def _edit_succeeded(
        self,
        request_id: UUID,
        target: _EditTarget,
        asset_id: UUID,
        result: object,
    ) -> None:
        if request_id != self._request_id:
            if isinstance(result, MfluxEditResult):
                result.dispose_output()
            return
        if not isinstance(result, MfluxEditResult):
            self._finish_with_error(
                BackgroundWorkflowError(
                    "image editing returned an unexpected result"
                )
            )
            return
        self._pending_result = result
        if not self._edit_target_is_current(target):
            self._finish_with_error(
                BackgroundWorkflowError(
                    "the stack or source revision changed before Edit completed"
                )
            )
            return
        store = self._require_store()
        image_path = store.image_asset_path(target.card_id, asset_id)
        background = GeneratedBackground(
            id=asset_id,
            image_path=image_path,
            provenance=result.provenance,
            created_at=result.provenance.settings.generated_at,
        )
        command = CreateEditedRevisionCommand(
            card_id=target.card_id,
            source_revision_id=target.revision.id,
            background=background,
        )
        before = self.controller.document
        owned_assets: list[OwnedImageAsset] = []

        def persist(candidate: Stack) -> None:
            try:
                stored = store.store_image_asset_and_save(
                    result.output_path,
                    destination_card_id=target.card_id,
                    destination_asset_id=asset_id,
                    previous_stack=before,
                    changed_stack=candidate,
                    expected_source_snapshot=target.source_snapshot,
                    expected_source_card_id=target.card_id,
                    expected_source_asset_id=target.source_background_id,
                    expected_source_operation="Edit",
                )
            except StackStoreTransactionError as error:
                if error.owned_asset is not None:
                    owned_assets.append(
                        self._owned_asset(
                            store,
                            target.card_id,
                            asset_id,
                            error.owned_asset,
                        )
                    )
                raise
            owned_assets.append(
                self._owned_asset(
                    store,
                    target.card_id,
                    asset_id,
                    stored,
                )
            )

        previous_token = self.controller.current_undo_token
        try:
            changed = store.run_locked(
                lambda: self.session.execute_persisted(
                    command,
                    persist=persist,
                    owned_assets=owned_assets,
                )
            )
        except (
            CommandError,
            DocumentMutationBlockedError,
            DocumentSessionError,
            StackStoreError,
            ValidationError,
        ) as error:
            if self.controller.document != before:
                authoritative = self.controller.document
                self.document_changed.emit(authoritative)
                if self.controller.mutation_blocked:
                    self._pending_edit_completion = _PendingEditCompletion(
                        document=authoritative,
                        previous_token=previous_token,
                    )
            self._finish_with_error(error)
            return
        self.progress_changed.emit("Image edited")
        self.document_changed.emit(changed)
        self._emit_change_applied("Image edited", previous_token)
        self.edit_instruction_clear_requested.emit()
        self._pending_result = None
        result.dispose_output()
        self._operation = None
        self._request_id = None
        self._request_target = None
        self._active_operation = None
        self._cleanup_source_snapshot_if_idle()
        self._set_busy(False, "Image edited")

    def _session_state_changed(self, state: object) -> None:
        pending = self._pending_edit_completion
        if pending is None or not isinstance(state, DocumentSessionState):
            return
        if state.mutation_blocked or state.dirty or state.error is not None:
            return
        self._pending_edit_completion = None
        if self.controller.document != pending.document:
            return
        current_token = self.controller.current_undo_token
        if current_token is None or current_token == pending.previous_token:
            return
        self.progress_changed.emit("Image edited")
        self.document_changed.emit(self.controller.document)
        self.change_applied.emit("Image edited", current_token)
        self.edit_instruction_clear_requested.emit()

    def _apply_background(
        self,
        card_id: UUID,
        revision_id: UUID,
        background: GeneratedBackground | None,
        message: str,
        *,
        generated: bool = False,
        owned_assets: tuple[OwnedImageAsset, ...] = (),
    ) -> Stack:
        previous_token = self.controller.current_undo_token
        previous_revision = next(
            revision
            for revision in self._card(self.controller.document, card_id).revisions
            if revision.id == revision_id
        )
        changed = self.controller.execute(
            ReplaceRevisionBackgroundCommand(
                card_id=card_id,
                revision_id=revision_id,
                background=background,
            )
        )
        self.controller.register_owned_assets(owned_assets)
        self.progress_changed.emit(message)
        self.document_changed.emit(changed)
        self._emit_change_applied(
            message,
            previous_token,
            card_id=card_id if generated else None,
            revision_id=revision_id if generated else None,
            previous_revision=previous_revision if generated else None,
        )
        return changed

    def _emit_change_applied(
        self,
        message: str,
        previous_token: object,
        *,
        card_id: UUID | None = None,
        revision_id: UUID | None = None,
        previous_revision: CardRevision | None = None,
    ) -> None:
        token = self.controller.current_undo_token
        if token is not None and token != previous_token:
            if card_id is not None and revision_id is not None and previous_revision is not None:
                self.generation_applied.emit(
                    GeneratedRevisionChange(
                        message=message,
                        token=token,
                        card_id=card_id,
                        revision_id=revision_id,
                        previous_revision=previous_revision,
                    )
                )
            else:
                self.change_applied.emit(message, token)

    def _operation_failed(self, request_id: UUID, failure: object) -> None:
        if request_id == self._request_id:
            self._mflux_generator.release()
            self._finish_with_error(failure)

    def _finish_with_error(self, failure: object) -> None:
        operation = self._active_operation
        self._operation = None
        self._request_id = None
        self._request_target = None
        self._active_operation = None
        self._discard_pending_image()
        self._cleanup_source_snapshot_if_idle()
        self._set_busy(
            False,
            (
                "Image refinement failed"
                if operation == "refine"
                else (
                    "Image editing failed"
                    if operation == "edit"
                    else "Image generation failed"
                )
            ),
        )
        self.failed.emit(failure)

    def _discard_pending_image(self) -> None:
        if self._pending_result is not None:
            self._pending_result.dispose_output()
            self._pending_result = None

    def _invocation_started(self) -> None:
        self._invocation_active.set()
        self.invocation_active_changed.emit(True)

    def _track_source_snapshot(self, snapshot: StoredImageSnapshot) -> None:
        with self._source_snapshot_lock:
            if self._source_snapshot is not None:
                raise BackgroundWorkflowError("a Refine source snapshot is already active")
            self._source_snapshot = snapshot

    def _cleanup_source_snapshot_if_idle(self) -> bool:
        if self._invocation_active.is_set():
            return False
        with self._source_snapshot_lock:
            snapshot = self._source_snapshot
            if snapshot is None:
                return True
            self._snapshot_cleanup_in_progress = True
        disposed = snapshot.dispose()
        with self._source_snapshot_lock:
            self._snapshot_cleanup_in_progress = False
            if disposed:
                self._source_snapshot = None
                self._temporary_cleanup_blocked = False
            else:
                self._temporary_cleanup_blocked = True
        if not disposed:
            with self._source_snapshot_lock:
                assert self._source_snapshot is snapshot
            logger.warning(
                "Refine source snapshot cleanup preserved a changed file: %s",
                snapshot.snapshot_path,
            )
        return disposed

    def _set_busy(self, busy: bool, progress: str) -> None:
        self._busy = busy
        self.busy_changed.emit(busy)
        self.progress_changed.emit(progress)

    def _require_ready(self, card_id: UUID) -> None:
        if self._busy or self._invocation_active.is_set():
            raise BackgroundWorkflowError("background generation is already running")
        with self._source_snapshot_lock:
            if self._source_snapshot is not None:
                raise BackgroundWorkflowError(
                    "the previous Refine source snapshot could not be cleaned up"
                )
        if self.controller.mutation_blocked:
            raise BackgroundWorkflowError(
                self.controller.mutation_blocked_reason or "save the stack before changing an image"
            )
        if self.session.store is None:
            raise BackgroundWorkflowError("save the stack before changing an image")
        self._card(self.controller.document, card_id)

    def _require_store(self) -> StackStore:
        store = self.session.store
        if store is None:
            raise BackgroundWorkflowError("save the stack before changing an image")
        return store

    def _target(
        self,
        document: Stack,
        card: Card,
        references: tuple[_GenerationReferenceTarget, ...],
    ) -> _GenerationTarget:
        bundle_path = self.session.state.bundle_path
        if bundle_path is None:
            raise BackgroundWorkflowError("save the stack before changing an image")
        revision = card.active_revision
        return _GenerationTarget(
            stack_id=document.id,
            card_id=card.id,
            revision_id=revision.id,
            description=revision.description,
            background_id=(revision.background.id if revision.background is not None else None),
            bundle_path=bundle_path.resolve(),
            references=references,
            style=self._style_snapshot(document, revision),
            generate_resolution=revision.generate_resolution,
        )

    def _target_is_current(self, target: _GenerationTarget) -> bool:
        bundle_path = self.session.state.bundle_path
        if bundle_path is None or bundle_path.resolve() != target.bundle_path:
            return False
        document = self.controller.document
        if document.id != target.stack_id:
            return False
        card = next(
            (card for card in document.cards if card.id == target.card_id),
            None,
        )
        if card is None or card.active_revision_id != target.revision_id:
            return False
        revision = card.active_revision
        if not (
            revision.description == target.description
            and (revision.background.id if revision.background is not None else None)
            == target.background_id
            and self._style_snapshot(document, revision) == target.style
            and revision.generate_resolution == target.generate_resolution
        ):
            return False
        try:
            return self._resolve_references(document, card) == target.references
        except BackgroundWorkflowError:
            return False

    def _refine_target_is_current(self, target: _RefineTarget) -> bool:
        bundle_path = self.session.state.bundle_path
        if bundle_path is None or bundle_path.resolve() != target.bundle_path:
            return False
        document = self.controller.document
        if (
            document.id != target.stack_id
            or document.aspect_ratio != target.aspect_ratio
            or self.controller.current_undo_token != target.history_token
        ):
            return False
        card = next(
            (card for card in document.cards if card.id == target.card_id),
            None,
        )
        if card is None or card.active_revision_id != target.revision.id:
            return False
        revision = card.active_revision
        if (
            revision != target.revision
            or self._style_snapshot(document, revision) != target.style
            or revision.background is None
            or image_edit_lineage(revision.background.provenance) != target.edit_lineage
        ):
            return False
        return self._settings_provider() == target.settings

    def _edit_target_is_current(self, target: _EditTarget) -> bool:
        bundle_path = self.session.state.bundle_path
        if bundle_path is None or bundle_path.resolve() != target.bundle_path:
            return False
        document = self.controller.document
        if (
            document.id != target.stack_id
            or document.aspect_ratio != target.aspect_ratio
            or self.controller.current_undo_token != target.history_token
        ):
            return False
        card = next(
            (
                card
                for card in document.cards
                if card.id == target.card_id
            ),
            None,
        )
        if card is None or card.active_revision_id != target.revision.id:
            return False
        revision = card.active_revision
        if (
            revision != target.revision
            or self._style_snapshot(document, revision) != target.style
            or revision.background is None
            or image_edit_lineage(revision.background.provenance)
            != target.edit_lineage[:-1]
        ):
            return False
        return self._settings_provider() == target.settings

    @staticmethod
    def _owned_asset(
        store: StackStore,
        card_id: UUID,
        asset_id: UUID,
        stored: StoredImageAsset,
    ) -> OwnedImageAsset:
        return OwnedImageAsset(
            bundle_path=store.bundle_path,
            relative_path=stored.relative_path,
            card_id=card_id,
            asset_id=asset_id,
            device=stored.device,
            inode=stored.inode,
            directory_device=stored.directory_device,
            directory_inode=stored.directory_inode,
        )

    def _resolve_references(
        self,
        document: Stack,
        card: Card,
    ) -> tuple[_GenerationReferenceTarget, ...]:
        references: list[_GenerationReferenceTarget] = []
        for position, assignment in enumerate(
            card.active_revision.references,
            start=1,
        ):
            if isinstance(assignment, UnresolvedCardReference):
                name = assignment.target_name or "unknown card"
                raise BackgroundWorkflowError(f"Reference {position} {name!r} is unresolved")
            assert isinstance(assignment, ResolvedCardReference)
            source = next(
                (
                    candidate
                    for candidate in document.cards
                    if candidate.id == assignment.target_card_id
                ),
                None,
            )
            if source is None:
                raise BackgroundWorkflowError(f"Reference {position} card no longer exists")
            source_revision = source.active_revision
            if source_revision.background is None:
                raise BackgroundWorkflowError(
                    f"Reference {position} card {source.name!r} has no image"
                )
            references.append(
                _GenerationReferenceTarget(
                    card_id=source.id,
                    card_name=source.name,
                    revision_id=source_revision.id,
                    background_id=source_revision.background.id,
                    image_path=source_revision.background.image_path,
                )
            )
        return tuple(references)

    @staticmethod
    def _style_snapshot(
        document: Stack,
        revision: CardRevision,
    ) -> StyleSnapshot | None:
        style = document.style_by_id(revision.style_id)
        if style is None:
            return None
        return StyleSnapshot(
            style_id=style.id,
            name=style.name,
            prompt_text=style.prompt_text,
        )

    def _reference_asset_path(
        self,
        reference: _GenerationReferenceTarget,
        position: int,
    ) -> Path:
        path = self._require_store().asset_path(reference.image_path)
        if not path.is_file():
            raise BackgroundWorkflowError(
                f"Reference {position} image for {reference.card_name!r} is unavailable"
            )
        try:
            require_readable_image(path)
        except UnreadableImageError as error:
            raise BackgroundWorkflowError(
                f"Reference {position} image for {reference.card_name!r} is unreadable"
            ) from error
        return path

    @staticmethod
    def _card(document: Stack, card_id: UUID) -> Card:
        card = next(
            (candidate for candidate in document.cards if candidate.id == card_id),
            None,
        )
        if card is None:
            raise BackgroundWorkflowError(f"card {card_id} no longer exists")
        return card


__all__ = [
    "BackgroundGenerationSettings",
    "BackgroundWorkflow",
    "BackgroundWorkflowError",
    "GenerationSettingsProvider",
]
