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

from hotcards.application.applied_image_change import (
    AppliedImageChange,
    EditImageOperation,
    ImageOperation,
    image_operation_message,
)
from hotcards.application.commands import (
    ActivateRevisionCommand,
    ApplyEditResultCommand,
    CommandError,
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
from hotcards.application.workers import AdapterWorkers, WorkerOperation
from hotcards.domain.image_dimensions import (
    AspectRatio,
    higher_output_tiers,
    validate_exact_output_dimensions,
)
from hotcards.domain.models import (
    AcceptedEdit,
    Card,
    CardRevision,
    CurrentSourceSize,
    DerivedImageSourceSnapshot,
    EditDraft,
    EditOutputSize,
    GeneratedBackground,
    GenerateInputs,
    GenerateOutputSize,
    ImageReferenceSnapshot,
    PresetOutputSize,
    ResolvedCardReference,
    Stack,
    StyleSnapshot,
    UnresolvedCardReference,
    image_edit_lineage,
    image_operation_settings,
    selected_output_dimensions,
)
from hotcards.generation.image_generation import (
    compose_edit_prompt,
    compose_generation_prompt,
)
from hotcards.generation.mflux_generator import (
    MfluxCancellationToken,
    MfluxEditRequest,
    MfluxEditResult,
    MfluxGenerateRequest,
    MfluxGenerateResult,
    MfluxGenerator,
    dispose_mflux_result,
)
from hotcards.storage.stack_store import (
    ImageAssetExpectation,
    StackStore,
    StackStoreError,
    StackStoreTransactionError,
    StoredImageAsset,
    StoredImageSnapshot,
)

logger = logging.getLogger(__name__)


class BackgroundWorkflowError(ValueError):
    """A background operation cannot proceed without losing user intent."""


def _current_source_size(
    width: int,
    height: int,
    aspect_ratio: AspectRatio,
) -> CurrentSourceSize | None:
    try:
        validate_exact_output_dimensions(width, height, aspect_ratio)
        return CurrentSourceSize(width=width, height=height)
    except ValueError:
        return None


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
    reference_assets: tuple[ImageAssetExpectation, ...]
    style: StyleSnapshot | None
    generate_output_size: GenerateOutputSize
    aspect_ratio: AspectRatio
    settings: BackgroundGenerationSettings


@dataclass(frozen=True, slots=True)
class _GenerationReferenceTarget:
    card_id: UUID
    card_name: str = field(compare=False)
    revision_id: UUID
    background_id: UUID
    image_path: str


@dataclass(frozen=True, slots=True)
class _EditTarget:
    stack_id: UUID
    card_id: UUID
    revision: CardRevision
    bundle_path: Path
    draft: EditDraft
    instruction: str
    expanded_prompt: str
    style: StyleSnapshot | None
    edit_lineage: tuple[AcceptedEdit, ...]
    aspect_ratio: AspectRatio
    output_size: EditOutputSize
    settings: BackgroundGenerationSettings
    history_token: UndoToken | None
    source_snapshot: StoredImageSnapshot
    source_background_id: UUID


@dataclass(slots=True)
class _PendingImageCompletion:
    store: StackStore
    document: Stack
    card_id: UUID
    previous_revision: CardRevision
    operation: ImageOperation | EditImageOperation
    token: UndoToken | None = None

    def record_history_token(self, token: UndoToken) -> None:
        self.token = token


GenerationSettingsProvider = Callable[[], BackgroundGenerationSettings]


class BackgroundWorkflow(QObject):
    """Generate, edit, and manage complete card revisions."""

    busy_changed = Signal(bool)
    invocation_active_changed = Signal(bool)
    progress_changed = Signal(str)
    generation_progress_changed = Signal(int, int)
    failed = Signal(object)
    document_changed = Signal(object)
    change_applied = Signal(str, object)
    image_applied = Signal(object)

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
        # Only identity-bound owners remove files; never recursively reclaim this workspace.
        self._owned_temporary_directory = (
            tempfile.TemporaryDirectory(prefix="hotcards-background-", delete=False)
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
        self._request_target: _GenerationTarget | _EditTarget | None = None
        self._pending_result: MfluxGenerateResult | MfluxEditResult | None = None
        self._close_requested = Event()
        self._invocation_active = Event()
        self._source_snapshot_lock = Lock()
        self._source_snapshots: tuple[StoredImageSnapshot, ...] = ()
        self._snapshot_cleanup_in_progress = False
        self._temporary_cleanup_blocked = False
        self._busy = False
        self._active_operation: Literal["generate", "edit"] | None = None
        self._pending_image_completion: _PendingImageCompletion | None = None
        self.session.state_changed.connect(self._session_state_changed)

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def active_operation(self) -> Literal["generate", "edit"] | None:
        return self._active_operation

    @property
    def invocation_active(self) -> bool:
        with self._source_snapshot_lock:
            return (
                self._invocation_active.is_set()
                or self._snapshot_cleanup_in_progress
                or (bool(self._source_snapshots) and self._temporary_cleanup_blocked)
            )

    def generate(self, card_id: UUID) -> WorkerOperation:
        """Generate and directly apply a background to the active revision."""
        self._require_ready(card_id)
        document = self.controller.document
        card = self._card(document, card_id)
        revision = card.active_revision
        if not revision.description.strip():
            raise BackgroundWorkflowError("enter a Description before generating")
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
            output_size=revision.generate_output_size,
        )
        render_prompt = compose_generation_prompt(inputs)
        request_id = uuid4()
        asset_id = uuid4()
        output_path = self._temporary_directory / f"generated-{asset_id}.png"
        width, height = selected_output_dimensions(
            revision.generate_output_size,
            document.aspect_ratio,
        )
        try:
            reference_assets = tuple(
                ImageAssetExpectation(
                    snapshot=self._reference_snapshot(reference, position),
                    card_id=reference.card_id,
                    asset_id=reference.background_id,
                )
                for position, reference in enumerate(references, start=1)
            )
            target = self._target(document, card, references, reference_assets, settings)
            request = MfluxGenerateRequest(
                inputs=inputs,
                render_prompt=render_prompt,
                output_path=output_path,
                model_identifier=settings.mflux_model,
                seed=(
                    secrets.randbelow(2_147_483_648)
                    if settings.random_seed
                    else settings.fixed_seed
                ),
                aspect_ratio=document.aspect_ratio,
                width=width,
                height=height,
                step_count=settings.step_count,
                quantization=settings.quantization,
                reference_image_paths=tuple(
                    reference.snapshot.snapshot_path for reference in reference_assets
                ),
            )
        except Exception:
            self._cleanup_source_snapshots_if_idle()
            raise
        self._request_id = request_id
        self._request_target = target
        cancellation = MfluxCancellationToken()
        self._active_operation = "generate"
        self._set_busy(True, "Generating image...")
        try:
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
        except Exception:
            self._request_id = None
            self._request_target = None
            self._active_operation = None
            self._cleanup_source_snapshots_if_idle()
            self._set_busy(False, "Image generation failed")
            raise
        self._operation = operation
        operation.succeeded.connect(
            partial(self._generation_succeeded, request_id, target, asset_id)
        )
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
        current_output_size = _current_source_size(
            width,
            height,
            document.aspect_ratio,
        )
        return (
            *((current_output_size,) if current_output_size is not None else ()),
            *(
                PresetOutputSize(tier=tier)
                for tier in higher_output_tiers(
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
        draft: EditDraft,
        output_size: EditOutputSize,
    ) -> WorkerOperation:
        """Durably replace the current revision's image with an Edit result."""
        self._require_ready(card_id)
        if not self.session.flush():
            raise BackgroundWorkflowError(
                self.session.state.error or "the current stack could not be saved"
            )
        document = self.controller.document
        card = self._card(document, card_id)
        revision = card.active_revision
        if revision.edit_draft != draft:
            raise BackgroundWorkflowError("the Edit Instruction changed before editing started")
        normalized_instruction = draft.instruction.strip()
        if not normalized_instruction:
            raise BackgroundWorkflowError("enter an Edit Instruction before editing")
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
        current_output_size = _current_source_size(
            source_snapshot.width,
            source_snapshot.height,
            document.aspect_ratio,
        )
        available_output_sizes = (
            *((current_output_size,) if current_output_size is not None else ()),
            *(
                PresetOutputSize(tier=tier)
                for tier in higher_output_tiers(
                    source_snapshot.width,
                    source_snapshot.height,
                    document.aspect_ratio,
                )
            ),
        )
        if output_size not in available_output_sizes:
            self._cleanup_source_snapshots_if_idle()
            raise BackgroundWorkflowError(
                "select the current Edit size or a preset with more pixels"
            )
        try:
            settings = self._settings_provider()
            style = self._style_snapshot(document, revision)
            expanded_prompt = compose_edit_prompt(
                normalized_instruction,
                style,
            )
        except Exception:
            self._cleanup_source_snapshots_if_idle()
            raise
        inherited_lineage = image_edit_lineage(background.provenance)
        width, height = (
            (output_size.width, output_size.height)
            if isinstance(output_size, CurrentSourceSize)
            else selected_output_dimensions(output_size, document.aspect_ratio)
        )
        source = DerivedImageSourceSnapshot(
            card_id=card.id,
            revision_id=revision.id,
            background_id=background.id,
            width=source_snapshot.width,
            height=source_snapshot.height,
            seed=image_operation_settings(background.provenance).seed,
        )
        request_id = uuid4()
        asset_id = uuid4()
        target = _EditTarget(
            stack_id=document.id,
            card_id=card.id,
            revision=revision.model_copy(deep=True),
            bundle_path=store.bundle_path.resolve(),
            draft=draft.model_copy(deep=True),
            instruction=normalized_instruction,
            expanded_prompt=expanded_prompt,
            style=style,
            edit_lineage=inherited_lineage,
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
            expanded_prompt=expanded_prompt,
            inherited_edit_lineage=inherited_lineage,
            output_size=output_size,
            seed=secrets.randbelow(2_147_483_648),
            output_path=self._temporary_directory / f"edited-{asset_id}.png",
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
            self._cleanup_source_snapshots_if_idle()
            self._set_busy(False, "Image editing failed")
            raise
        self._operation = operation
        operation.succeeded.connect(partial(self._edit_succeeded, request_id, target, asset_id))
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
        self._cleanup_source_snapshots_if_idle()
        if self._busy:
            operation = self._active_operation
            self._active_operation = None
            self._set_busy(
                False,
                "Edit cancelled" if operation == "edit" else "Generation cancelled",
            )

    def close(self) -> None:
        self._close_requested.set()
        self.cancel()
        if self._owned_temporary_directory is not None and not self.invocation_active:
            self._cleanup_temporary_directory()

    def _invocation_finished(self) -> None:
        self._invocation_active.clear()
        self._cleanup_source_snapshots_if_idle()
        self.invocation_active_changed.emit(self.invocation_active)
        if not self._close_requested.is_set():
            return
        self._mflux_generator.release()
        if self._owned_temporary_directory is not None and not self.invocation_active:
            self._cleanup_temporary_directory()

    def _cleanup_temporary_directory(self) -> None:
        try:
            self._temporary_directory.rmdir()
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning(
                "Image workspace cleanup preserved remaining files: %s",
                self._temporary_directory,
            )

    def is_generating_for(self, card_id: UUID) -> bool:
        return (
            self._busy
            and self._active_operation == "generate"
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
        self._accept_image(
            request_id,
            target,
            asset_id,
            result,
            is_current=lambda: self._target_is_current(target),
        )

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
                BackgroundWorkflowError("image editing returned an unexpected result")
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
        self._accept_image(
            request_id,
            target,
            asset_id,
            result,
            is_current=lambda: self._edit_target_is_current(target),
        )

    def _accept_image(
        self,
        request_id: UUID,
        target: _GenerationTarget | _EditTarget,
        asset_id: UUID,
        result: MfluxGenerateResult | MfluxEditResult,
        *,
        is_current: Callable[[], bool],
    ) -> None:
        completion: _PendingImageCompletion | None = None

        def accept() -> None:
            nonlocal completion
            if not self.session.flush():
                raise BackgroundWorkflowError(
                    self.session.state.error or "the current stack could not be saved"
                )
            if request_id != self._request_id or not is_current():
                raise BackgroundWorkflowError("the image request changed before acceptance")
            store = self._require_store()
            before = self.controller.document
            previous_revision = self._card(before, target.card_id).active_revision
            background = GeneratedBackground(
                id=asset_id,
                image_path=store.image_asset_path(target.card_id, asset_id),
                provenance=result.provenance,
                created_at=result.provenance.settings.generated_at,
            )
            command = (
                ApplyEditResultCommand(
                    card_id=target.card_id,
                    revision_id=previous_revision.id,
                    background=background,
                    submitted_draft_generation_id=target.draft.generation_id,
                )
                if isinstance(target, _EditTarget)
                else ReplaceRevisionBackgroundCommand(
                    card_id=target.card_id,
                    revision_id=previous_revision.id,
                    background=background,
                )
            )
            completion = _PendingImageCompletion(
                store=store,
                document=command.apply(before),
                card_id=target.card_id,
                previous_revision=previous_revision,
                operation=(
                    EditImageOperation(instruction=target.instruction)
                    if isinstance(target, _EditTarget)
                    else ImageOperation("generate")
                ),
            )
            owned_assets: list[OwnedImageAsset] = []
            derived = target if isinstance(target, _EditTarget) else None

            def persist(candidate: Stack) -> None:
                try:
                    stored = store.store_image_asset_and_save(
                        result.output_path,
                        destination_card_id=target.card_id,
                        destination_asset_id=asset_id,
                        previous_stack=before,
                        changed_stack=candidate,
                        expected_source_snapshot=derived.source_snapshot if derived else None,
                        expected_source_card_id=derived.card_id if derived else None,
                        expected_source_asset_id=derived.source_background_id if derived else None,
                        expected_source_operation="Edit",
                        expected_references=(
                            target.reference_assets if isinstance(target, _GenerationTarget) else ()
                        ),
                    )
                except StackStoreTransactionError as error:
                    if isinstance(error.owned_asset, StoredImageAsset):
                        owned_assets.append(
                            self._owned_asset(store, target.card_id, asset_id, error.owned_asset)
                        )
                    raise
                owned_assets.append(self._owned_asset(store, target.card_id, asset_id, stored))

            self.session.execute_persisted(
                command,
                persist=persist,
                owned_assets=owned_assets,
                on_recorded=completion.record_history_token,
            )

        try:
            # Existing origin facts cannot substitute for a complete new-operation receipt.
            result.model_dump()
            self._require_store().run_locked(accept)
        except (
            BackgroundWorkflowError,
            CommandError,
            DocumentMutationBlockedError,
            DocumentSessionError,
            StackStoreError,
            ValidationError,
        ) as error:
            if completion is not None:
                if completion.token is not None:
                    self._publish_image_completion(completion)
                elif (
                    self.controller.mutation_blocked
                    and self.controller.document == completion.document
                ):
                    self._pending_image_completion = completion
                    self.document_changed.emit(completion.document)
            self._finish_with_error(error)
            return
        assert completion is not None
        self._publish_image_completion(completion)
        self._discard_pending_image()
        self._operation = None
        self._request_id = None
        self._request_target = None
        self._active_operation = None
        self._cleanup_source_snapshots_if_idle()
        self._set_busy(False, image_operation_message(completion.operation))

    def _session_state_changed(self, state: object) -> None:
        pending = self._pending_image_completion
        if pending is None or not isinstance(state, DocumentSessionState):
            return
        if self.session.store is not pending.store:
            self._pending_image_completion = None
            return
        if pending.token is None:
            return
        self._pending_image_completion = None
        self._publish_image_completion(pending)

    def _publish_image_completion(self, completion: _PendingImageCompletion) -> None:
        token = completion.token
        if (
            self.session.store is not completion.store
            or token is None
            or token not in self.controller.retained_history_tokens
        ):
            return
        self.progress_changed.emit(image_operation_message(completion.operation))
        self.document_changed.emit(self.controller.document)
        self.image_applied.emit(
            AppliedImageChange(
                token=token,
                card_id=completion.card_id,
                revision_id=completion.previous_revision.id,
                previous_revision=completion.previous_revision,
                operation=completion.operation,
            )
        )

    def _emit_change_applied(
        self,
        message: str,
        previous_token: object,
    ) -> None:
        token = self.controller.current_undo_token
        if token is not None and token != previous_token:
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
        self._cleanup_source_snapshots_if_idle()
        self._set_busy(
            False,
            "Image editing failed" if operation == "edit" else "Image generation failed",
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
            self._source_snapshots = (*self._source_snapshots, snapshot)

    def _cleanup_source_snapshots_if_idle(self) -> bool:
        with self._source_snapshot_lock:
            if self._invocation_active.is_set() or self._snapshot_cleanup_in_progress:
                return False
            snapshots = self._source_snapshots
            if not snapshots:
                return True
            self._snapshot_cleanup_in_progress = True
        retained = tuple(snapshot for snapshot in snapshots if not snapshot.dispose())
        with self._source_snapshot_lock:
            self._snapshot_cleanup_in_progress = False
            self._source_snapshots = retained
            self._temporary_cleanup_blocked = bool(retained)
        for snapshot in retained:
            logger.warning(
                "Image source snapshot cleanup preserved a changed file: %s",
                snapshot.snapshot_path,
            )
        return not retained

    def _set_busy(self, busy: bool, progress: str) -> None:
        self._busy = busy
        self.busy_changed.emit(busy)
        self.progress_changed.emit(progress)

    def _require_ready(self, card_id: UUID) -> None:
        if self._busy or self._invocation_active.is_set():
            raise BackgroundWorkflowError("background generation is already running")
        with self._source_snapshot_lock:
            if self._source_snapshots:
                raise BackgroundWorkflowError(
                    "the previous image source snapshot could not be cleaned up"
                )
        if self._close_requested.is_set():
            raise BackgroundWorkflowError("the background workflow is closed")
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
        reference_assets: tuple[ImageAssetExpectation, ...],
        settings: BackgroundGenerationSettings,
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
            reference_assets=reference_assets,
            style=self._style_snapshot(document, revision),
            generate_output_size=revision.generate_output_size,
            aspect_ratio=document.aspect_ratio,
            settings=settings,
        )

    def _target_is_current(self, target: _GenerationTarget) -> bool:
        bundle_path = self.session.state.bundle_path
        if bundle_path is None or bundle_path.resolve() != target.bundle_path:
            return False
        document = self.controller.document
        if (
            document.id != target.stack_id
            or document.aspect_ratio != target.aspect_ratio
            or self._settings_provider() != target.settings
        ):
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
            and revision.generate_output_size == target.generate_output_size
        ):
            return False
        try:
            return self._resolve_references(document, card) == target.references
        except BackgroundWorkflowError:
            return False

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
            (card for card in document.cards if card.id == target.card_id),
            None,
        )
        if card is None or card.active_revision_id != target.revision.id:
            return False
        revision = card.active_revision
        if (
            revision.model_copy(update={"edit_draft": target.revision.edit_draft})
            != target.revision
            or self._style_snapshot(document, revision) != target.style
            or revision.background is None
            or image_edit_lineage(revision.background.provenance) != target.edit_lineage
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

    def _reference_snapshot(
        self,
        reference: _GenerationReferenceTarget,
        position: int,
    ) -> StoredImageSnapshot:
        try:
            snapshot = self._require_store().snapshot_image_asset(
                reference.image_path,
                card_id=reference.card_id,
                asset_id=reference.background_id,
                destination_directory=self._temporary_directory,
            )
        except StackStoreError as error:
            raise BackgroundWorkflowError(
                f"Reference {position} image for {reference.card_name!r} "
                "is unavailable or unreadable"
            ) from error
        self._track_source_snapshot(snapshot)
        return snapshot

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
