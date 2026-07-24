"""Restrained three-pane HyperGen application shell."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import ValidationError
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from hypergen.application.background_workflow import (
    BackgroundGenerationSettings,
    BackgroundWorkflow,
    BackgroundWorkflowError,
)
from hypergen.application.commands import (
    AddInteractionCommand,
    AddPolygonCommand,
    CommandError,
    DeleteInteractionCommand,
    DeletePolygonCommand,
    DocumentCommand,
    ReplacePolygonCommand,
    SetRunOverlayModeCommand,
)
from hypergen.application.document_controller import DocumentController
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
from hypergen.application.run_session import RunSession, RunSessionState
from hypergen.application.scene_enrichment_workflow import (
    SceneEnrichmentWorkflow,
    SceneEnrichmentWorkflowError,
)
from hypergen.application.workers import (
    AdapterKind,
    AdapterWorkers,
    AvailabilityDiagnostic,
    WorkerFailure,
    WorkerOperation,
)
from hypergen.domain.models import (
    HotspotSet,
    Interaction,
    NavigateAction,
    Polygon,
    RunOverlayMode,
    Stack,
    UnresolvedCardReference,
)
from hypergen.generation.ollama_client import OllamaSettings
from hypergen.storage.stack_store import StackStoreError
from hypergen.ui.card_canvas import CardCanvas
from hypergen.ui.card_sidebar import CardSidebar
from hypergen.ui.crop_dialog import CropDialog
from hypergen.ui.inspector import Inspector
from hypergen.ui.new_stack_dialog import NewStackDialog
from hypergen.ui.settings_dialog import (
    SettingsDialog,
    SettingsStore,
    load_machine_settings,
)

AvailabilityChecks = Mapping[AdapterKind, Callable[[], Any]]
AvailabilityChecksFactory = Callable[[], AvailabilityChecks]
SettingsDialogFactory = Callable[[SettingsStore, QWidget], QDialog]


class MainWindow(QMainWindow):
    """Application shell whose panes render one DocumentController."""

    def __init__(
        self,
        controller: DocumentController,
        workers: AdapterWorkers,
        settings: SettingsStore | None = None,
        *,
        availability_checks: AvailabilityChecks | None = None,
        availability_checks_factory: AvailabilityChecksFactory | None = None,
        settings_dialog_factory: SettingsDialogFactory = SettingsDialog,
        document_session: DocumentSession | None = None,
        background_workflow: BackgroundWorkflow | None = None,
        scene_enrichment_workflow: SceneEnrichmentWorkflow | None = None,
        hotspot_generation_workflow: HotspotGenerationWorkflow | None = None,
        start_diagnostics: bool = True,
        owns_workers: bool = False,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.workers = workers
        self.settings = settings if settings is not None else QSettings()
        self.document_session = document_session
        self.background_workflow = background_workflow
        self.scene_enrichment_workflow = scene_enrichment_workflow
        self.hotspot_generation_workflow = hotspot_generation_workflow
        self._availability_checks = dict(availability_checks or {})
        self._availability_checks_factory = availability_checks_factory
        self._settings_dialog_factory = settings_dialog_factory
        self._owns_workers = owns_workers
        self._selected_card_id = (
            controller.document.cards[0].id if controller.document.cards else None
        )
        self._availability: dict[AdapterKind, bool | None] = {
            AdapterKind.OLLAMA: None,
            AdapterKind.MFLUX: None,
        }
        self._diagnostic_messages: dict[AdapterKind, str] = {}
        self._diagnostic_operations: list[WorkerOperation] = []
        self._diagnostic_generation = 0
        self._rendering = False
        self._is_running = False
        self._run_session = RunSession()
        if self.background_workflow is None and self.document_session is not None:
            self.background_workflow = BackgroundWorkflow(
                controller,
                self.document_session,
                workers,
                self._background_generation_settings,
                parent=self,
            )
        if self.scene_enrichment_workflow is None:
            self.scene_enrichment_workflow = SceneEnrichmentWorkflow(
                controller,
                workers,
                self._ollama_settings,
                parent=self,
            )
        if self.hotspot_generation_workflow is None:
            self.hotspot_generation_workflow = HotspotGenerationWorkflow(
                controller,
                workers,
                self._ollama_settings,
                self._resolve_revision_image_path,
                parent=self,
            )

        self.setWindowTitle(f"HyperGen — {controller.document.name}")
        self.setObjectName("mainWindow")
        self.resize(1180, 760)
        self._build_toolbar()
        self._build_panes()
        self._build_menu()
        self.workers.availability_changed.connect(self.apply_availability_diagnostic)
        if self.document_session is not None:
            self.document_session.document_replaced.connect(self._document_replaced)
            self.document_session.state_changed.connect(self._session_state_changed)
        self.render_document(controller.document)
        self._session_state_changed(
            self.document_session.state
            if self.document_session is not None
            else DocumentSessionState(bundle_path=None, dirty=False, error=None)
        )
        self._update_generation_actions()
        if start_diagnostics:
            self.run_availability_checks()

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Authoring")
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addWidget(QLabel("Mode"))
        self.mode_selector = QComboBox()
        self.mode_selector.setObjectName("modeSelector")
        self.mode_selector.addItems(["Author", "Run"])
        self.mode_selector.setToolTip("Switch between authoring and interactive preview")
        self.mode_selector.currentIndexChanged.connect(self._mode_changed)
        toolbar.addWidget(self.mode_selector)
        toolbar.addSeparator()

        self.overlay_label = QLabel("Overlay")
        toolbar.addWidget(self.overlay_label)
        self.overlay_selector = QComboBox()
        self.overlay_selector.setObjectName("overlaySelector")
        for mode, label in (
            (RunOverlayMode.HIDDEN, "Hidden"),
            (RunOverlayMode.HOVER, "On hover"),
            (RunOverlayMode.VISIBLE, "Visible"),
        ):
            self.overlay_selector.addItem(label, mode)
        self.overlay_selector.currentIndexChanged.connect(self._overlay_changed)
        toolbar.addWidget(self.overlay_selector)
        toolbar.addSeparator()

        self.back_action = QAction("Back", self)
        self.back_action.setObjectName("runBackAction")
        self.back_action.setToolTip("Return to the previous visited card")
        self.back_action.triggered.connect(self._run_back)
        self.restart_action = QAction("Restart", self)
        self.restart_action.setObjectName("runRestartAction")
        self.restart_action.setToolTip("Return to the stack's start card")
        self.restart_action.triggered.connect(self._run_restart)
        self.player_navigation_actions = (
            self.back_action,
            self.restart_action,
        )
        for action in self.player_navigation_actions:
            action.setEnabled(False)
            action.setVisible(False)
            toolbar.addAction(action)
        self.overlay_label.setVisible(False)
        self.overlay_selector.setVisible(False)

    def _build_panes(self) -> None:
        self.card_sidebar = CardSidebar(self.controller)
        self.card_sidebar.card_selected.connect(self.select_card)
        self.card_sidebar.document_changed.connect(self.render_document)
        self.card_sidebar.delete_requested.connect(self._confirm_delete_card)

        self.canvas_pages = QStackedWidget()
        self.canvas_pages.setObjectName("canvasPages")

        empty_canvas = QWidget()
        empty_canvas.setObjectName("emptyCanvas")
        empty_layout = QVBoxLayout(empty_canvas)
        empty_layout.addStretch(1)
        self.empty_canvas_title = QLabel("Create your first card")
        self.empty_canvas_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_canvas_title.setStyleSheet("font-size: 20px; font-weight: 600;")
        empty_layout.addWidget(self.empty_canvas_title)
        self.empty_canvas_description = QLabel(
            "Cards are the scenes readers visit. Start with one, then add its background and links."
        )
        self.empty_canvas_description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_canvas_description.setWordWrap(True)
        empty_layout.addWidget(self.empty_canvas_description)
        self.create_first_card_button = QPushButton("Create Your First Card")
        self.create_first_card_button.setObjectName("createFirstCardButton")
        empty_layout.addWidget(
            self.create_first_card_button,
            0,
            Qt.AlignmentFlag.AlignHCenter,
        )
        empty_layout.addStretch(1)
        self.canvas_pages.addWidget(empty_canvas)

        card_canvas = QWidget()
        card_canvas.setObjectName("cardCanvasPanel")
        canvas_layout = QVBoxLayout(card_canvas)
        canvas_toolbar = QHBoxLayout()
        canvas_toolbar.addStretch(1)
        self.fit_canvas_button = QPushButton("Fit")
        self.fit_canvas_button.setObjectName("fitCanvasButton")
        canvas_toolbar.addWidget(self.fit_canvas_button)
        canvas_layout.addLayout(canvas_toolbar)
        self.canvas_title = QLabel("Card Canvas")
        self.canvas_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas_title.setStyleSheet("font-size: 18px; font-weight: 600;")
        canvas_layout.addWidget(self.canvas_title)
        self.canvas_card_name = QLabel()
        self.canvas_card_name.setObjectName("canvasCardName")
        self.canvas_card_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        canvas_layout.addWidget(self.canvas_card_name)
        self.card_canvas = CardCanvas()
        canvas_layout.addWidget(self.card_canvas, 1)
        self.canvas_pages.addWidget(card_canvas)
        self.fit_canvas_button.clicked.connect(self.card_canvas.fit_to_window)

        self.inspector = Inspector(
            self.controller,
            image_path_resolver=self._resolve_revision_image_path,
        )
        self.inspector.document_changed.connect(self.render_document)
        self.inspector.render_inputs_changed.connect(
            self._update_generation_actions
        )
        self.inspector.enrich_scene_requested.connect(self._enrich_scene)
        self.inspector.accept_scene_enrichment_requested.connect(
            self._accept_scene_enrichment
        )
        self.inspector.discard_scene_enrichment_requested.connect(
            self._discard_scene_enrichment
        )
        self.inspector.generate_hotspots_requested.connect(
            self._generate_hotspots
        )
        self.inspector.apply_hotspot_candidate_requested.connect(
            self._apply_hotspot_candidate
        )
        self.inspector.discard_hotspot_candidate_requested.connect(
            self._discard_hotspot_candidate
        )
        self.inspector.candidate_label_changed.connect(
            self._rename_hotspot_candidate
        )
        self.inspector.candidate_destination_changed.connect(
            self._set_hotspot_candidate_destination
        )
        self.inspector.candidate_create_destination_requested.connect(
            self._create_hotspot_candidate_destination
        )
        self.inspector.candidate_reorder_requested.connect(
            self._reorder_hotspot_candidate
        )
        self.inspector.candidate_delete_requested.connect(
            self._delete_hotspot_candidate
        )
        self.inspector.generate_background_requested.connect(self._generate_background)
        self.inspector.import_background_requested.connect(self._import_background)
        self.inspector.accept_background_draft_requested.connect(
            self._accept_background_draft
        )
        self.inspector.discard_background_draft_requested.connect(
            self._discard_background_draft
        )
        self.inspector.revision_activation_requested.connect(self._activate_revision)
        self.inspector.revision_deletion_requested.connect(self._delete_revision)
        self.inspector.hotspot_selected.connect(
            self.card_canvas.select_interaction
        )
        self.inspector.add_hotspot_requested.connect(
            lambda: self.card_canvas.begin_polygon()
        )
        self.inspector.add_hotspot_component_requested.connect(
            self.card_canvas.begin_polygon
        )
        self.card_canvas.interaction_selected.connect(
            self.inspector.select_interaction
        )
        self.card_canvas.polygon_created.connect(self._create_hotspot_polygon)
        self.card_canvas.polygon_changed.connect(self._replace_hotspot_polygon)
        self.card_canvas.polygon_deletion_requested.connect(
            self._delete_hotspot_polygon
        )
        self.card_canvas.interaction_deletion_requested.connect(
            self._delete_hotspot_interaction
        )
        self.card_canvas.editing_error.connect(self.inspector.set_hotspot_error)
        self.card_canvas.interaction_activated.connect(
            self._run_interaction_activated
        )
        if self.background_workflow is not None:
            self.background_workflow.drafts_changed.connect(
                self._background_drafts_changed
            )
            self.background_workflow.busy_changed.connect(
                lambda _busy: self._update_generation_actions()
            )
            self.background_workflow.progress_changed.connect(
                self._background_progress_changed
            )
            self.background_workflow.failed.connect(self._background_failed)
            self.background_workflow.document_changed.connect(self.render_document)
        self.scene_enrichment_workflow.draft_changed.connect(
            self._scene_enrichment_changed
        )
        self.scene_enrichment_workflow.busy_changed.connect(
            lambda _busy: self._update_generation_actions()
        )
        self.scene_enrichment_workflow.progress_changed.connect(
            self._scene_enrichment_progress_changed
        )
        self.scene_enrichment_workflow.failed.connect(
            self._scene_enrichment_failed
        )
        self.scene_enrichment_workflow.document_changed.connect(
            self.render_document
        )
        self.hotspot_generation_workflow.candidate_changed.connect(
            self._hotspot_candidate_changed
        )
        self.hotspot_generation_workflow.busy_changed.connect(
            lambda _busy: self._update_generation_actions()
        )
        self.hotspot_generation_workflow.progress_changed.connect(
            self._hotspot_generation_progress_changed
        )
        self.hotspot_generation_workflow.failed.connect(
            self._hotspot_generation_failed
        )
        self.hotspot_generation_workflow.document_changed.connect(
            self.render_document
        )

        self.pane_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.pane_splitter.setObjectName("threePaneSplitter")
        self.pane_splitter.addWidget(self.card_sidebar)
        self.pane_splitter.addWidget(self.canvas_pages)
        self.pane_splitter.addWidget(self.inspector)
        self.pane_splitter.setStretchFactor(0, 0)
        self.pane_splitter.setStretchFactor(1, 1)
        self.pane_splitter.setStretchFactor(2, 0)
        self.pane_splitter.setSizes([220, 700, 280])
        self.setCentralWidget(self.pane_splitter)

        self.service_status_label = QLabel("Checking local AI services…")
        self.service_status_label.setObjectName("serviceStatusLabel")
        self.statusBar().addPermanentWidget(self.service_status_label, 1)
        self.check_services_button = QPushButton("Check AI Services")
        self.check_services_button.setObjectName("checkServicesButton")
        self.check_services_button.clicked.connect(self.run_availability_checks)
        self.statusBar().addPermanentWidget(self.check_services_button)
        self.review_settings_button = QPushButton("Review Settings…")
        self.review_settings_button.setObjectName("reviewSettingsButton")
        self.review_settings_button.setVisible(False)
        self.review_settings_button.clicked.connect(self.open_advanced_settings)
        self.statusBar().addPermanentWidget(self.review_settings_button)
        self.create_first_card_button.clicked.connect(self._primary_empty_action)

        self.document_status_label = QLabel()
        self.document_status_label.setObjectName("documentStatusLabel")
        self.statusBar().addWidget(self.document_status_label, 1)
        self.run_status_label = QLabel()
        self.run_status_label.setObjectName("runStatusLabel")
        self.run_status_label.setStyleSheet("color: #d8a657;")
        self.run_status_label.setVisible(False)
        self.statusBar().addWidget(self.run_status_label, 2)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        self.new_stack_action = QAction("New Stack…", self)
        self.new_stack_action.setObjectName("newStackAction")
        self.new_stack_action.setShortcut(QKeySequence.StandardKey.New)
        self.new_stack_action.triggered.connect(self.new_stack)
        file_menu.addAction(self.new_stack_action)
        self.open_stack_action = QAction("Open Stack…", self)
        self.open_stack_action.setObjectName("openStackAction")
        self.open_stack_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_stack_action.triggered.connect(self.open_stack)
        file_menu.addAction(self.open_stack_action)
        file_menu.addSeparator()
        self.save_action = QAction("Save", self)
        self.save_action.setObjectName("saveStackAction")
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(self.save_document)
        file_menu.addAction(self.save_action)
        self.save_as_action = QAction("Save As…", self)
        self.save_as_action.setObjectName("saveStackAsAction")
        self.save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.save_as_action.triggered.connect(self.save_as)
        file_menu.addAction(self.save_as_action)

        edit_menu = self.menuBar().addMenu("Edit")
        self.undo_action = QAction("Undo", self)
        self.undo_action.setObjectName("undoAction")
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.triggered.connect(self.undo)
        edit_menu.addAction(self.undo_action)
        self.redo_action = QAction("Redo", self)
        self.redo_action.setObjectName("redoAction")
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_action.triggered.connect(self.redo)
        edit_menu.addAction(self.redo_action)

        self.advanced_settings_action = QAction("Advanced Settings…", self)
        self.advanced_settings_action.setObjectName("advancedSettingsAction")
        self.advanced_settings_action.triggered.connect(self.open_advanced_settings)
        self.menuBar().addMenu("HyperGen").addAction(self.advanced_settings_action)

    def render_document(self, _document: Stack | None = None) -> None:
        """Refresh all panes from the controller's authoritative snapshot."""
        self.hotspot_generation_workflow.discard_if_stale()
        snapshot = self.controller.document
        card_ids = {card.id for card in snapshot.cards}
        if self.background_workflow is not None:
            self.background_workflow.discard_orphaned_drafts(card_ids)
        if self._is_running:
            self._selected_card_id = self._run_session.state.current_card_id
        elif self._selected_card_id not in card_ids:
            self._selected_card_id = snapshot.cards[0].id if snapshot.cards else None
        self._rendering = True
        try:
            draft_card_ids = (
                self.background_workflow.draft_card_ids
                if self.background_workflow is not None
                else ()
            )
            self.card_sidebar.render(
                snapshot,
                self._selected_card_id,
                draft_card_ids=draft_card_ids,
            )
            self.inspector.render(snapshot, self._selected_card_id)
            selected_card = next(
                (card for card in snapshot.cards if card.id == self._selected_card_id),
                None,
            )
            if selected_card is None:
                self.canvas_pages.setCurrentIndex(0)
                self.canvas_card_name.clear()
                self.inspector.show_background_draft(None)
                self.inspector.show_scene_enrichment(None)
                self.inspector.show_hotspot_candidate(None)
            else:
                self.canvas_pages.setCurrentIndex(1)
                self.canvas_card_name.setText(selected_card.name)
                draft = (
                    self.background_workflow.draft_for(selected_card.id)
                    if self.background_workflow is not None and not self._is_running
                    else None
                )
                self.inspector.show_background_draft(draft)
                enrichment = self.scene_enrichment_workflow.draft
                self.inspector.show_scene_enrichment(
                    enrichment
                    if (
                        enrichment is not None
                        and enrichment.card_id == selected_card.id
                        and not self._is_running
                    )
                    else None
                )
                candidate = self.hotspot_generation_workflow.candidate
                self.inspector.show_hotspot_candidate(
                    candidate
                    if (
                        candidate is not None
                        and candidate.card_id == selected_card.id
                        and not self._is_running
                    )
                    else None
                )
                self._render_card_canvas(selected_card)
            overlay_index = self.overlay_selector.findData(snapshot.run_overlay_mode)
            self.overlay_selector.setCurrentIndex(overlay_index)
        finally:
            self._rendering = False
        self._update_document_actions()
        self._update_window_title()
        self._update_generation_actions()
        self._update_run_actions()

    def select_card(self, card_id: object) -> None:
        if self._is_running:
            return
        selected_card_id = card_id if isinstance(card_id, UUID) else None
        if selected_card_id != self._selected_card_id:
            self.scene_enrichment_workflow.cancel()
            self.hotspot_generation_workflow.cancel()
        self._selected_card_id = selected_card_id
        if self.card_sidebar.selected_card_id != self._selected_card_id:
            self.card_sidebar.select_card(self._selected_card_id)
        self.render_document()

    def new_stack(self) -> None:
        """Create and bind a new stack before exposing its initial card."""
        if self.document_session is None:
            return
        self.inspector.commit_card_metadata()
        dialog = NewStackDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        stack = dialog.stack()
        suggested_name = f"{stack.name}.hypergen"
        selected_path, _filter = QFileDialog.getSaveFileName(
            self,
            "Create HyperGen Stack",
            suggested_name,
            "HyperGen Stack (*.hypergen)",
        )
        if not selected_path:
            return
        if not self._confirm_drafts_discard("creating a new stack"):
            return
        if not self._confirm_generation_cancel("creating a new stack"):
            return
        try:
            self.document_session.create(stack, self._bundle_path(selected_path))
        except DocumentSessionError as error:
            self._show_document_error("Could Not Create Stack", str(error))
        else:
            if self.background_workflow is not None:
                self.background_workflow.discard_all_drafts()

    def open_stack(self) -> None:
        """Open a validated bundle without replacing the current session on failure."""
        if self.document_session is None:
            return
        self.inspector.commit_card_metadata()
        selected_path = QFileDialog.getExistingDirectory(
            self,
            "Open HyperGen Stack",
        )
        if not selected_path:
            return
        if not self._confirm_drafts_discard("opening another stack"):
            return
        if not self._confirm_generation_cancel("opening another stack"):
            return
        try:
            self.document_session.open(Path(selected_path))
        except DocumentSessionError as error:
            self._show_document_error("Could Not Open Stack", str(error))
        else:
            if self.background_workflow is not None:
                self.background_workflow.discard_all_drafts()

    def save_document(self) -> bool:
        """Flush accepted mutations and keep a failed save visible."""
        if self.document_session is None:
            return True
        self.inspector.commit_card_metadata()
        saved = self.document_session.flush()
        if not saved:
            self._show_document_error(
                "Could Not Save Stack",
                self.document_session.state.error or "The document could not be saved.",
            )
        return saved

    def save_as(self) -> None:
        """Clone the current bound bundle and rebind future autosaves."""
        if self.document_session is None or self.document_session.store is None:
            return
        self.inspector.commit_card_metadata()
        selected_path, _filter = QFileDialog.getSaveFileName(
            self,
            "Save HyperGen Stack As",
            f"{self.controller.document.name}.hypergen",
            "HyperGen Stack (*.hypergen)",
        )
        if not selected_path:
            return
        if not self._confirm_generation_cancel("saving the stack under a new name"):
            return
        try:
            self.document_session.save_as(self._bundle_path(selected_path))
        except DocumentSessionError as error:
            self._show_document_error("Could Not Save Stack As", str(error))

    def undo(self) -> None:
        if self.controller.undo():
            self.render_document()

    def redo(self) -> None:
        if self.controller.redo():
            self.render_document()

    def _primary_empty_action(self) -> None:
        if self._is_running:
            return
        if self.document_session is not None and self.document_session.store is None:
            self.new_stack()
        else:
            self.card_sidebar.add_card()

    def _confirm_delete_card(self, card_id: object) -> None:
        if not isinstance(card_id, UUID):
            return
        card = next(
            (candidate for candidate in self.controller.document.cards if candidate.id == card_id),
            None,
        )
        if card is None:
            return
        start_warning = (
            "\n\nThis is the start card; the stack will no longer have a start card."
            if self.controller.document.start_card_id == card.id
            else ""
        )
        message = (
            f'Delete "{card.name}"? Inbound links will be kept as unresolved references.'
            f"{start_warning}"
        )
        if self._ask_delete_card(message):
            draft = (
                self.background_workflow.draft_for(card.id)
                if self.background_workflow is not None
                else None
            )
            if (
                self.background_workflow is not None
                and self.background_workflow.is_generating_for(card.id)
                and not self._confirm_generation_cancel("deleting this card")
            ):
                return
            self.scene_enrichment_workflow.cancel()
            self.hotspot_generation_workflow.cancel()
            self.card_sidebar.delete_card(card.id)
            if draft is not None:
                self.background_workflow.discard_draft(card.id)

    def _ask_delete_card(self, message: str) -> bool:
        dialog = QMessageBox(
            QMessageBox.Icon.Warning,
            "Delete Card",
            message,
            parent=self,
        )
        delete_button = dialog.addButton(
            "Delete Card",
            QMessageBox.ButtonRole.DestructiveRole,
        )
        dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.exec()
        return dialog.clickedButton() is delete_button

    def _document_replaced(self, _document: object) -> None:
        self.scene_enrichment_workflow.cancel()
        self.hotspot_generation_workflow.cancel()
        if self._is_running:
            state = self._run_session.start(self.controller.document)
            self._selected_card_id = state.current_card_id
        else:
            self._selected_card_id = None
        self.render_document()

    def _session_state_changed(self, state: object) -> None:
        if not isinstance(state, DocumentSessionState):
            return
        bound = state.bundle_path is not None
        self.card_sidebar.set_document_editable(
            self.document_session is None or bound
        )
        if self.document_session is not None and not bound:
            self.create_first_card_button.setText("Create New Stack")
            self.document_status_label.setText("Create or open a stack")
            self.document_status_label.setToolTip("")
        elif state.error is not None:
            self.create_first_card_button.setText("Create Your First Card")
            self.document_status_label.setText("Save failed")
            self.document_status_label.setToolTip(state.error)
        elif state.dirty:
            self.create_first_card_button.setText("Create Your First Card")
            self.document_status_label.setText("Unsaved changes")
            self.document_status_label.setToolTip("Autosave is pending")
        else:
            self.create_first_card_button.setText("Create Your First Card")
            label = (
                state.bundle_path.name
                if state.bundle_path is not None
                else "In-memory document"
            )
            self.document_status_label.setText(label)
            self.document_status_label.setToolTip(
                str(state.bundle_path) if state.bundle_path is not None else ""
            )
        self._update_document_actions()
        self._update_window_title()
        self._update_generation_actions()

    def _update_document_actions(self) -> None:
        bound = self.document_session is None or self.document_session.store is not None
        self.save_action.setEnabled(
            self.document_session is not None and self.document_session.store is not None
        )
        self.save_as_action.setEnabled(
            self.document_session is not None and self.document_session.store is not None
        )
        self.new_stack_action.setEnabled(not self._is_running)
        self.open_stack_action.setEnabled(not self._is_running)
        self.save_as_action.setEnabled(
            not self._is_running
            and self.document_session is not None
            and self.document_session.store is not None
        )
        self.undo_action.setEnabled(
            bound and not self._is_running and self.controller.can_undo
        )
        self.redo_action.setEnabled(
            bound and not self._is_running and self.controller.can_redo
        )
        self.advanced_settings_action.setEnabled(not self._is_running)
        self.mode_selector.setEnabled(bound)
        self.overlay_selector.setEnabled(bound and self._is_running)
        self.card_sidebar.set_document_editable(bound and not self._is_running)

    def _update_window_title(self) -> None:
        dirty = self.document_session is not None and self.document_session.state.dirty
        suffix = " *" if dirty else ""
        self.setWindowTitle(f"HyperGen — {self.controller.document.name}{suffix}")

    @staticmethod
    def _bundle_path(selected_path: str) -> Path:
        path = Path(selected_path)
        return path if path.suffix == ".hypergen" else path.with_suffix(".hypergen")

    def _show_document_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def run_availability_checks(self) -> None:
        """Submit injected service checks without blocking the UI thread."""
        if self._is_running or self._diagnostic_operations:
            return
        checks = (
            dict(self._availability_checks_factory())
            if self._availability_checks_factory is not None
            else self._availability_checks
        )
        self._diagnostic_generation += 1
        generation = self._diagnostic_generation
        for adapter, check in checks.items():
            self._availability[adapter] = None
            self._diagnostic_messages.pop(adapter, None)
            operation = (
                self.workers.check_ollama(check, emit_diagnostic=False)
                if adapter is AdapterKind.OLLAMA
                else self.workers.check_mflux(check, emit_diagnostic=False)
            )
            self._diagnostic_operations.append(operation)
            operation.succeeded.connect(
                partial(self._availability_check_succeeded, adapter, generation)
            )
            operation.failed.connect(partial(self._availability_check_failed, adapter, generation))
            operation.finished.connect(partial(self._diagnostic_finished, operation))
        self._update_generation_actions()

    def _generate_background(self) -> None:
        workflow = self.background_workflow
        card_id = self._selected_card_id
        if workflow is None or card_id is None:
            return
        if not self.inspector.commit_card_metadata():
            return
        replacing_draft = workflow.draft_for(card_id) is not None
        if replacing_draft and not self._confirm_draft_replacement():
            return
        try:
            workflow.generate(card_id, replace_draft=replacing_draft)
        except BackgroundWorkflowError as error:
            self.inspector.set_background_status(str(error), detail=str(error))
        self._update_generation_actions()

    def _enrich_scene(self) -> None:
        card_id = self._selected_card_id
        if card_id is None or not self.inspector.commit_card_metadata():
            return
        try:
            self.scene_enrichment_workflow.start(card_id)
        except SceneEnrichmentWorkflowError as error:
            self.inspector.set_scene_enrichment_status(str(error), detail=str(error))
        self._update_generation_actions()

    def _accept_scene_enrichment(self) -> None:
        card_id = self._selected_card_id
        if card_id is None:
            return
        try:
            self.scene_enrichment_workflow.apply(
                card_id,
                self.inspector.enriched_scene_edit.toPlainText(),
            )
        except SceneEnrichmentWorkflowError as error:
            self.inspector.set_scene_enrichment_status(str(error), detail=str(error))

    def _discard_scene_enrichment(self) -> None:
        self.scene_enrichment_workflow.discard()

    def _scene_enrichment_changed(self) -> None:
        self.render_document()

    def _scene_enrichment_progress_changed(self, message: str) -> None:
        self.inspector.set_scene_enrichment_status(message)
        self._update_generation_actions()

    def _scene_enrichment_failed(self, failure: object) -> None:
        detail = failure.message if isinstance(failure, WorkerFailure) else str(failure)
        self.inspector.set_scene_enrichment_status(
            "Scene enrichment failed",
            detail=detail,
        )
        self._update_generation_actions()

    def _generate_hotspots(self) -> None:
        card_id = self._selected_card_id
        if card_id is None or not self.inspector.commit_card_metadata():
            return
        try:
            self.hotspot_generation_workflow.start(card_id)
        except HotspotGenerationWorkflowError as error:
            self.inspector.set_hotspot_generation_status(
                str(error),
                detail=str(error),
            )
        self._update_generation_actions()

    def _apply_hotspot_candidate(self) -> None:
        try:
            self.hotspot_generation_workflow.apply()
        except (HotspotGenerationWorkflowError, CommandError, ValidationError) as error:
            self.inspector.set_hotspot_error(str(error))

    def _discard_hotspot_candidate(self) -> None:
        self.hotspot_generation_workflow.discard()

    def _rename_hotspot_candidate(
        self,
        interaction_id: object,
        label: str,
    ) -> None:
        if not isinstance(interaction_id, UUID):
            return
        self._edit_hotspot_candidate(
            lambda: self.hotspot_generation_workflow.rename_interaction(
                interaction_id,
                label,
            )
        )

    def _set_hotspot_candidate_destination(
        self,
        interaction_id: object,
        card_id: object,
    ) -> None:
        if not isinstance(interaction_id, UUID):
            return
        self._edit_hotspot_candidate(
            lambda: self.hotspot_generation_workflow.set_destination(
                interaction_id,
                card_id if isinstance(card_id, UUID) else None,
            )
        )

    def _create_hotspot_candidate_destination(
        self,
        interaction_id: object,
        name: str,
    ) -> None:
        if not isinstance(interaction_id, UUID):
            return
        self._edit_hotspot_candidate(
            lambda: self.hotspot_generation_workflow.create_destination_card(
                interaction_id,
                name,
            )
        )

    def _reorder_hotspot_candidate(
        self,
        interaction_id: object,
        new_index: int,
    ) -> None:
        if not isinstance(interaction_id, UUID):
            return
        self._edit_hotspot_candidate(
            lambda: self.hotspot_generation_workflow.reorder_interaction(
                interaction_id,
                new_index,
            )
        )

    def _delete_hotspot_candidate(self, interaction_id: object) -> None:
        if not isinstance(interaction_id, UUID):
            return
        self._edit_hotspot_candidate(
            lambda: self.hotspot_generation_workflow.delete_interaction(
                interaction_id
            )
        )

    def _edit_hotspot_candidate(self, edit: Callable[[], object]) -> None:
        try:
            edit()
        except (HotspotGenerationWorkflowError, ValidationError, CommandError) as error:
            self.inspector.set_hotspot_error(str(error))
        else:
            self.inspector.set_hotspot_error("")

    def _hotspot_candidate_changed(self) -> None:
        candidate = self.hotspot_generation_workflow.candidate
        if candidate is None:
            self.inspector.show_hotspot_candidate(None)
            self.inspector.render(
                self.controller.document,
                self._selected_card_id,
            )
        else:
            self.inspector.show_hotspot_candidate(
                candidate
                if (
                    candidate.card_id == self._selected_card_id
                    and not self._is_running
                )
                else None
            )
        card = next(
            (
                card
                for card in self.controller.document.cards
                if card.id == self._selected_card_id
            ),
            None,
        )
        if card is not None:
            self._render_card_canvas(card)
        self._update_generation_actions()

    def _hotspot_generation_progress_changed(self, message: str) -> None:
        self.inspector.set_hotspot_generation_status(message)
        self._update_generation_actions()

    def _hotspot_generation_failed(self, failure: object) -> None:
        detail = failure.message if isinstance(failure, WorkerFailure) else str(failure)
        self.inspector.set_hotspot_generation_status(
            "Hotspot generation failed",
            detail=detail,
        )
        self._update_generation_actions()

    def _import_background(self) -> None:
        workflow = self.background_workflow
        card_id = self._selected_card_id
        if workflow is None or card_id is None:
            return
        if not self.inspector.commit_card_metadata():
            return
        selected_path, _filter = QFileDialog.getOpenFileName(
            self,
            "Import Background",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.tif *.tiff)",
        )
        if not selected_path:
            return
        source_path = Path(selected_path)
        position_x = 0.5
        position_y = 0.5
        try:
            canvas_size = self.controller.document.canvas
            if CropDialog.requires_crop(source_path, canvas_size):
                crop_dialog = CropDialog(source_path, canvas_size, self)
                if crop_dialog.exec() != QDialog.DialogCode.Accepted:
                    return
                position_x = crop_dialog.position_x
                position_y = crop_dialog.position_y
            workflow.import_image(
                card_id,
                source_path,
                position_x=position_x,
                position_y=position_y,
            )
        except (BackgroundWorkflowError, ValueError) as error:
            self.inspector.set_background_status(str(error), detail=str(error))
        self._update_generation_actions()

    def _accept_background_draft(self) -> None:
        if self.background_workflow is None or self._selected_card_id is None:
            return
        try:
            self.background_workflow.apply_draft(self._selected_card_id)
        except (BackgroundWorkflowError, StackStoreError, CommandError) as error:
            self.inspector.set_background_status(str(error), detail=str(error))

    def _discard_background_draft(self) -> None:
        if self.background_workflow is not None and self._selected_card_id is not None:
            self.background_workflow.discard_draft(self._selected_card_id)

    def _activate_revision(self, revision_id: object) -> None:
        if (
            self.background_workflow is None
            or self._selected_card_id is None
            or not isinstance(revision_id, UUID)
        ):
            return
        try:
            self.background_workflow.activate_revision(
                self._selected_card_id,
                revision_id,
            )
        except (BackgroundWorkflowError, CommandError) as error:
            self.inspector.set_background_status(str(error), detail=str(error))

    def _delete_revision(self, revision_id: object) -> None:
        if (
            self.background_workflow is None
            or self._selected_card_id is None
            or not isinstance(revision_id, UUID)
        ):
            return
        dialog = QMessageBox(
            QMessageBox.Icon.Warning,
            "Delete Background Revision",
            "Delete this revision and its associated hotspots?",
            parent=self,
        )
        delete_button = dialog.addButton(
            "Delete Revision",
            QMessageBox.ButtonRole.DestructiveRole,
        )
        dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.exec()
        if dialog.clickedButton() is not delete_button:
            return
        try:
            self.background_workflow.delete_revision(
                self._selected_card_id,
                revision_id,
            )
        except (BackgroundWorkflowError, CommandError) as error:
            self.inspector.set_background_status(str(error), detail=str(error))

    def _background_drafts_changed(self) -> None:
        self.render_document()

    def _confirm_drafts_discard(self, action: str) -> bool:
        workflow = self.background_workflow
        if workflow is None or not workflow.drafts:
            return True
        draft_ids = workflow.draft_card_ids
        card_names = [
            card.name
            for card in self.controller.document.cards
            if card.id in draft_ids
        ]
        affected = ", ".join(card_names)
        answer = QMessageBox.question(
            self,
            "Discard Background Drafts?",
            (
                f"Discard {len(workflow.drafts)} background "
                f"{'draft' if len(workflow.drafts) == 1 else 'drafts'} before {action}?"
                f"\n\nCards: {affected}"
            ),
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Discard:
            return False
        return True

    def _confirm_draft_replacement(self) -> bool:
        answer = QMessageBox.question(
            self,
            "Generate Replacement?",
            (
                "Generate a replacement for this card's current background draft? "
                "The current draft will be kept if generation fails."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _confirm_generation_cancel(self, action: str) -> bool:
        if (
            self.background_workflow is None
            or not self.background_workflow.busy
        ):
            return True
        answer = QMessageBox.question(
            self,
            "Cancel Background Generation?",
            f"Background generation is still running. Cancel it before {action}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        self.background_workflow.cancel()
        return True

    def _background_progress_changed(self, message: str) -> None:
        self.inspector.set_background_status(message)
        self._update_generation_actions()

    def _background_failed(self, failure: object) -> None:
        if isinstance(failure, WorkerFailure):
            self.inspector.set_background_status(
                "Background generation failed",
                detail=failure.message,
            )
        else:
            self.inspector.set_background_status(str(failure), detail=str(failure))
        self._update_generation_actions()

    def _availability_check_succeeded(
        self,
        adapter: AdapterKind,
        generation: int,
        _result: object,
    ) -> None:
        if generation != self._diagnostic_generation:
            return
        self.apply_availability_diagnostic(
            AvailabilityDiagnostic(
                adapter=adapter,
                available=True,
                message=f"{adapter.value} is available.",
            )
        )

    def _availability_check_failed(
        self,
        adapter: AdapterKind,
        generation: int,
        failure: object,
    ) -> None:
        if generation != self._diagnostic_generation:
            return
        from hypergen.application.workers import WorkerFailure

        if not isinstance(failure, WorkerFailure):
            return
        self.apply_availability_diagnostic(
            AvailabilityDiagnostic(
                adapter=adapter,
                available=False,
                message=failure.message,
                failure=failure,
            )
        )

    def _diagnostic_finished(self, operation: WorkerOperation) -> None:
        if operation in self._diagnostic_operations:
            self._diagnostic_operations.remove(operation)
            self._update_generation_actions()

    def apply_availability_diagnostic(
        self,
        diagnostic: AvailabilityDiagnostic,
    ) -> None:
        """Reflect AdapterWorkers availability transitions in generation actions."""
        self._availability[diagnostic.adapter] = diagnostic.available
        self._diagnostic_messages[diagnostic.adapter] = diagnostic.message
        self._update_generation_actions()

    def _update_generation_actions(self) -> None:
        bound = self.document_session is None or self.document_session.store is not None
        workflow_available = self.background_workflow is not None
        has_card = (
            workflow_available
            and bound
            and self._selected_card_id is not None
            and not self._is_running
        )
        has_render_prompt = self.inspector.has_render_prompt_input()
        mflux_available = self._availability[AdapterKind.MFLUX] is True
        workflow_busy = (
            self.background_workflow.busy
            if self.background_workflow is not None
            else False
        )
        selected_has_draft = (
            self.background_workflow is not None
            and self._selected_card_id is not None
            and self.background_workflow.draft_for(self._selected_card_id) is not None
        )
        generate_reason = "Ready to generate"
        if not has_card:
            generate_reason = "Select a card in a saved stack"
        elif workflow_busy:
            generate_reason = (
                "Background generation is running for this card"
                if self.background_workflow is not None
                and self._selected_card_id is not None
                and self.background_workflow.is_generating_for(self._selected_card_id)
                else (
                    "Background generation is running for another card; "
                    "MFLUX runs one job at a time"
                )
            )
        elif not has_render_prompt:
            generate_reason = "Enter a Scene or Style before generating"
        elif not mflux_available:
            generate_reason = self._action_diagnostic(AdapterKind.MFLUX)
        elif selected_has_draft:
            generate_reason = "Ready to generate a replacement draft"
        import_reason = (
            "Ready to import"
            if has_card and not selected_has_draft and not workflow_busy
            else (
                "Accept or discard this card's draft before importing"
                if selected_has_draft
                else (
                    "Background generation is running; MFLUX runs one job at a time"
                    if workflow_busy
                    else "Select a card in a saved stack"
                )
            )
        )
        self.inspector.set_background_capabilities(
            can_generate=(
                has_card
                and has_render_prompt
                and mflux_available
            ),
            generate_reason=generate_reason,
            can_import=has_card and not selected_has_draft,
            import_reason=import_reason,
            busy=workflow_busy,
        )
        enrichment_busy = self.scene_enrichment_workflow.busy
        enrichment_draft = self.scene_enrichment_workflow.draft
        ollama_available = self._availability[AdapterKind.OLLAMA] is True
        can_enrich = (
            has_card
            and self.inspector.has_scene_input()
            and ollama_available
            and not enrichment_busy
            and enrichment_draft is None
        )
        enrich_reason = "Ready to enrich Scene"
        if not has_card:
            enrich_reason = "Select a card in a saved stack"
        elif not self.inspector.has_scene_input():
            enrich_reason = "Enter a Scene before enriching"
        elif enrichment_busy:
            enrich_reason = "Scene enrichment is running"
        elif enrichment_draft is not None:
            enrich_reason = "Accept or discard the current enriched Scene"
        elif not ollama_available:
            enrich_reason = self._action_diagnostic(AdapterKind.OLLAMA)
        self.inspector.set_scene_enrichment_capabilities(
            can_enrich=can_enrich,
            reason=enrich_reason,
        )
        selected_card = next(
            (
                card
                for card in self.controller.document.cards
                if card.id == self._selected_card_id
            ),
            None,
        )
        has_active_revision = (
            selected_card is not None
            and selected_card.active_revision_id is not None
        )
        hotspot_busy = self.hotspot_generation_workflow.busy
        hotspot_candidate = self.hotspot_generation_workflow.candidate
        can_generate_hotspots = (
            has_card
            and has_active_revision
            and ollama_available
            and not hotspot_busy
            and hotspot_candidate is None
        )
        hotspot_reason = "Ready to generate hotspot candidates"
        if not has_card:
            hotspot_reason = "Select a card in a saved stack"
        elif not has_active_revision:
            hotspot_reason = "Apply a background before generating hotspots"
        elif hotspot_busy:
            hotspot_reason = "Hotspot generation is running"
        elif hotspot_candidate is not None:
            hotspot_reason = "Apply or discard the current hotspot candidate"
        elif not ollama_available:
            hotspot_reason = self._action_diagnostic(AdapterKind.OLLAMA)
        self.inspector.set_hotspot_generation_capabilities(
            can_generate=can_generate_hotspots,
            reason=hotspot_reason,
        )
        pending = [
            adapter for adapter, available in self._availability.items() if available is None
        ]
        unavailable = [
            adapter for adapter, available in self._availability.items() if available is False
        ]
        diagnostics_running = bool(self._diagnostic_operations)
        if pending and diagnostics_running:
            summary = "Checking local AI services…"
        elif pending:
            summary = "AI services not checked"
        elif unavailable:
            labels = {
                AdapterKind.OLLAMA: "Ollama",
                AdapterKind.MFLUX: "MFLUX",
            }
            names = " and ".join(labels[adapter] for adapter in unavailable)
            summary = f"{names} unavailable"
        else:
            summary = "Local AI services ready"
        self.service_status_label.setText(summary)
        self.service_status_label.setToolTip(
            "\n".join(
                self._diagnostic_messages.get(
                    adapter,
                    f"{adapter.value} availability check pending",
                )
                for adapter in AdapterKind
            )
        )
        self.review_settings_button.setVisible(
            bool(unavailable) and not self._is_running
        )
        self.check_services_button.setText(
            "Check Again" if unavailable else "Check AI Services"
        )
        self.check_services_button.setVisible(
            not self._is_running
            and not diagnostics_running
            and bool(pending or unavailable)
        )

    def _render_card_canvas(self, card: object) -> None:
        from hypergen.domain.models import Card

        if not isinstance(card, Card):
            return
        self.card_canvas.set_canvas_size(self.controller.document.canvas)
        draft = (
            self.background_workflow.draft_for(card.id)
            if self.background_workflow is not None and not self._is_running
            else None
        )
        if draft is not None:
            self.card_canvas.show_image(draft.image_path, candidate=True)
            self.card_canvas.set_hotspots(None, None, editable=False)
            return
        revision = next(
            (
                revision
                for revision in card.image_revisions
                if revision.id == card.active_revision_id
            ),
            None,
        )
        if revision is None:
            self.card_canvas.show_message("No background revision")
            if self._is_running:
                self._set_run_warning(
                    f'"{card.name}" has no active background revision.'
                )
            return
        if self.document_session is None or self.document_session.store is None:
            self.card_canvas.show_message("Background bundle is unavailable")
            if self._is_running:
                self._set_run_warning(
                    f'The background for "{card.name}" is unavailable.'
                )
            return
        try:
            asset_path = self.document_session.store.asset_path(revision.image_path)
        except StackStoreError as error:
            self.card_canvas.show_message(str(error))
            if self._is_running:
                self._set_run_warning(str(error))
            return
        self.card_canvas.show_image(asset_path)
        if self._is_running:
            self.card_canvas.set_run_hotspots(
                revision.hotspot_set,
                self.controller.document.run_overlay_mode,
            )
            if (
                revision.hotspot_set is None
                or not revision.hotspot_set.interactions
            ):
                self._set_run_warning(
                    f'"{card.name}" has no hotspots to navigate.'
                )
            return
        candidate = self._active_hotspot_candidate()
        if candidate is not None:
            self.card_canvas.set_hotspots(
                candidate.hotspot_set,
                self.inspector.selected_interaction_id,
                editable=True,
            )
            return
        self.card_canvas.set_hotspots(
            revision.hotspot_set,
            self.inspector.selected_interaction_id,
            editable=True,
        )

    def _active_hotspot_candidate(self) -> HotspotGenerationDraft | None:
        candidate = self.hotspot_generation_workflow.candidate
        if candidate is None or candidate.card_id != self._selected_card_id:
            return None
        card = next(
            (
                card
                for card in self.controller.document.cards
                if card.id == candidate.card_id
            ),
            None,
        )
        if card is None or card.active_revision_id != candidate.revision_id:
            return None
        return candidate

    def _resolve_revision_image_path(self, image_path: str) -> Path | None:
        if self.document_session is None or self.document_session.store is None:
            return None
        try:
            return self.document_session.store.asset_path(image_path)
        except StackStoreError:
            return None

    def _create_hotspot_polygon(
        self,
        interaction_id: object,
        polygon: object,
    ) -> None:
        candidate = self._active_hotspot_candidate()
        if candidate is not None and isinstance(polygon, Polygon):
            selected_id = interaction_id if isinstance(interaction_id, UUID) else None
            try:
                if selected_id is None:
                    selected_id = self.hotspot_generation_workflow.add_interaction(
                        polygon
                    )
                else:
                    self.hotspot_generation_workflow.add_polygon(
                        selected_id,
                        polygon,
                    )
            except (HotspotGenerationWorkflowError, ValidationError) as error:
                self.inspector.set_hotspot_error(str(error))
                return
            self.inspector.select_interaction(selected_id)
            self.card_canvas.select_interaction(selected_id)
            return
        context = self._active_hotspot_context()
        if context is None or not isinstance(polygon, Polygon):
            return
        card_id, revision_id, hotspot_set = context
        if isinstance(interaction_id, UUID):
            command = AddPolygonCommand(
                card_id=card_id,
                revision_id=revision_id,
                interaction_id=interaction_id,
                polygon=polygon,
            )
            selected_id = interaction_id
        else:
            existing_labels = {
                interaction.label.casefold()
                for interaction in hotspot_set.interactions
            }
            number = len(hotspot_set.interactions) + 1
            while f"Hotspot {number}".casefold() in existing_labels:
                number += 1
            interaction = Interaction(
                label=f"Hotspot {number}",
                action=NavigateAction(target=UnresolvedCardReference()),
                polygons=(polygon,),
            )
            command = AddInteractionCommand(
                card_id=card_id,
                revision_id=revision_id,
                interaction=interaction,
            )
            selected_id = interaction.id
        self._execute_hotspot_canvas_command(command, selected_id)

    def _replace_hotspot_polygon(
        self,
        interaction_id: object,
        polygon_index: int,
        polygon: object,
    ) -> None:
        if (
            self._active_hotspot_candidate() is not None
            and isinstance(interaction_id, UUID)
            and isinstance(polygon, Polygon)
        ):
            self._edit_hotspot_candidate(
                lambda: self.hotspot_generation_workflow.replace_polygon(
                    interaction_id,
                    polygon_index,
                    polygon,
                )
            )
            self.inspector.select_interaction(interaction_id)
            self.card_canvas.select_interaction(interaction_id)
            return
        context = self._active_hotspot_context()
        if (
            context is None
            or not isinstance(interaction_id, UUID)
            or not isinstance(polygon, Polygon)
        ):
            return
        card_id, revision_id, _hotspot_set = context
        self._execute_hotspot_canvas_command(
            ReplacePolygonCommand(
                card_id=card_id,
                revision_id=revision_id,
                interaction_id=interaction_id,
                polygon_index=polygon_index,
                polygon=polygon,
            ),
            interaction_id,
        )

    def _delete_hotspot_polygon(
        self,
        interaction_id: object,
        polygon_index: int,
    ) -> None:
        if (
            self._active_hotspot_candidate() is not None
            and isinstance(interaction_id, UUID)
        ):
            self._edit_hotspot_candidate(
                lambda: self.hotspot_generation_workflow.delete_polygon(
                    interaction_id,
                    polygon_index,
                )
            )
            self.inspector.select_interaction(interaction_id)
            self.card_canvas.select_interaction(interaction_id)
            return
        context = self._active_hotspot_context()
        if context is None or not isinstance(interaction_id, UUID):
            return
        card_id, revision_id, _hotspot_set = context
        self._execute_hotspot_canvas_command(
            DeletePolygonCommand(
                card_id=card_id,
                revision_id=revision_id,
                interaction_id=interaction_id,
                polygon_index=polygon_index,
            ),
            interaction_id,
        )

    def _delete_hotspot_interaction(self, interaction_id: object) -> None:
        if (
            self._active_hotspot_candidate() is not None
            and isinstance(interaction_id, UUID)
        ):
            self._edit_hotspot_candidate(
                lambda: self.hotspot_generation_workflow.delete_interaction(
                    interaction_id
                )
            )
            return
        context = self._active_hotspot_context()
        if context is None or not isinstance(interaction_id, UUID):
            return
        card_id, revision_id, _hotspot_set = context
        self._execute_hotspot_canvas_command(
            DeleteInteractionCommand(
                card_id=card_id,
                revision_id=revision_id,
                interaction_id=interaction_id,
            ),
            None,
        )

    def _execute_hotspot_canvas_command(
        self,
        command: DocumentCommand,
        selected_interaction_id: UUID | None,
    ) -> None:
        try:
            changed = self.controller.execute(command)
        except (CommandError, ValidationError) as error:
            self.inspector.set_hotspot_error(str(error))
            self.render_document()
            return
        self.inspector.set_hotspot_error("")
        self.render_document(changed)
        self.inspector.select_interaction(selected_interaction_id)
        self.card_canvas.select_interaction(selected_interaction_id)

    def _active_hotspot_context(
        self,
    ) -> tuple[UUID, UUID, HotspotSet] | None:
        card = next(
            (
                card
                for card in self.controller.document.cards
                if card.id == self._selected_card_id
            ),
            None,
        )
        if card is None or card.active_revision_id is None:
            self.inspector.set_hotspot_error(
                "Apply a background before editing hotspots."
            )
            return None
        revision = next(
            revision
            for revision in card.image_revisions
            if revision.id == card.active_revision_id
        )
        return (
            card.id,
            revision.id,
            revision.hotspot_set or HotspotSet(),
        )

    def _background_generation_settings(self) -> BackgroundGenerationSettings:
        values = load_machine_settings(self.settings)
        return BackgroundGenerationSettings(
            mflux_model=values.mflux_model,
            step_count=values.step_count,
            quantization=values.quantization,
            random_seed=values.random_seed,
            fixed_seed=values.fixed_seed,
        )

    def _ollama_settings(self) -> OllamaSettings:
        values = load_machine_settings(self.settings)
        return OllamaSettings(
            endpoint=values.ollama_endpoint,
            model=values.ollama_model,
        )

    def _action_diagnostic(self, adapter: AdapterKind) -> str:
        if self._availability[adapter] is True:
            return f"{adapter.value} is available"
        return self._diagnostic_messages.get(
            adapter,
            f"Waiting for asynchronous {adapter.value} availability check",
        )

    def _overlay_changed(self, index: int) -> None:
        if self._rendering:
            return
        mode = self.overlay_selector.itemData(index)
        try:
            overlay_mode = RunOverlayMode(mode)
        except (TypeError, ValueError):
            return
        changed = self.controller.execute(SetRunOverlayModeCommand(mode=overlay_mode))
        self.render_document(changed)

    def _mode_changed(self, index: int) -> None:
        if self._rendering:
            return
        should_run = self.mode_selector.itemText(index) == "Run"
        if should_run == self._is_running:
            return
        if should_run:
            self.inspector.commit_card_metadata()
            self.card_canvas.cancel_drawing()
            self._cancel_ai_activity_for_run()
            self._is_running = True
            state = self._run_session.start(
                self.controller.document,
                self._selected_card_id,
            )
            self._selected_card_id = state.current_card_id
        else:
            self._is_running = False
            self._run_session.clear()
            state = None
        self._apply_mode_chrome()
        self._set_run_warning("")
        self.render_document()
        if state is not None and state.warning is not None:
            self._set_run_warning(state.warning)
        if not should_run:
            self.run_availability_checks()

    def _run_interaction_activated(self, interaction_id: object) -> None:
        if not self._is_running or not isinstance(interaction_id, UUID):
            return
        state = self._run_session.navigate(
            self.controller.document,
            interaction_id,
        )
        self._apply_run_state(state)

    def _run_back(self) -> None:
        if self._is_running:
            self._apply_run_state(self._run_session.back())

    def _run_restart(self) -> None:
        if self._is_running:
            self._apply_run_state(self._run_session.restart())

    def _apply_run_state(self, state: RunSessionState) -> None:
        self._selected_card_id = state.current_card_id
        self._set_run_warning("")
        self.render_document()
        if state.warning is not None:
            self._set_run_warning(state.warning)

    def _apply_mode_chrome(self) -> None:
        authoring = not self._is_running
        self.card_sidebar.setVisible(authoring)
        self.inspector.setVisible(authoring)
        self.fit_canvas_button.setVisible(authoring)
        self.canvas_title.setVisible(authoring)
        self.create_first_card_button.setVisible(authoring)
        self.empty_canvas_title.setText(
            "Create your first card" if authoring else "No cards to run"
        )
        self.empty_canvas_description.setText(
            "Cards are the scenes readers visit. Start with one, then add its background and links."
            if authoring
            else "Switch to Author mode to create the first card."
        )
        self.overlay_label.setVisible(self._is_running)
        self.overlay_selector.setVisible(self._is_running)
        for action in self.player_navigation_actions:
            action.setVisible(self._is_running)
        self.service_status_label.setVisible(authoring)
        pending = any(
            available is None
            for available in self._availability.values()
        )
        unavailable = any(
            available is False
            for available in self._availability.values()
        )
        self.check_services_button.setVisible(
            authoring
            and not self._diagnostic_operations
            and (pending or unavailable)
        )
        self.review_settings_button.setVisible(
            authoring and unavailable
        )
        self.run_status_label.setVisible(self._is_running)

    def _update_run_actions(self) -> None:
        state = self._run_session.state
        self.back_action.setEnabled(
            self._is_running and bool(state.history)
        )
        self.restart_action.setEnabled(
            self._is_running and state.current_card_id is not None
        )

    def _set_run_warning(self, message: str) -> None:
        self.run_status_label.setText(message)
        self.run_status_label.setToolTip(message)

    def _cancel_ai_activity_for_run(self) -> None:
        self._cancel_diagnostics()
        self.scene_enrichment_workflow.cancel()
        self.hotspot_generation_workflow.cancel()
        if self.background_workflow is not None and self.background_workflow.busy:
            self.background_workflow.cancel()

    def _cancel_diagnostics(self) -> None:
        self._diagnostic_generation += 1
        for operation in tuple(self._diagnostic_operations):
            operation.cancel()

    def _restart_availability_checks(self) -> None:
        self._cancel_diagnostics()
        self.run_availability_checks()

    def open_advanced_settings(self) -> None:
        dialog = self._settings_dialog_factory(self.settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._restart_availability_checks()

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._confirm_drafts_discard("closing the stack"):
            event.ignore()
            return
        if not self._confirm_generation_cancel("closing the stack"):
            event.ignore()
            return
        self.inspector.commit_card_metadata()
        if self.document_session is not None and not self.document_session.flush():
            answer = QMessageBox.warning(
                self,
                "Stack Not Saved",
                (self.document_session.state.error or "The stack could not be saved.")
                + "\n\nRetry saving before closing?",
                QMessageBox.StandardButton.Retry | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Retry,
            )
            if answer == QMessageBox.StandardButton.Retry:
                if not self.document_session.flush():
                    event.ignore()
                    return
            else:
                event.ignore()
                return
        if self.background_workflow is not None:
            self.background_workflow.close()
        self.scene_enrichment_workflow.close()
        self.hotspot_generation_workflow.close()
        if self._owns_workers:
            self.workers.shutdown(wait_milliseconds=100)
        super().closeEvent(event)


__all__ = [
    "AvailabilityChecks",
    "AvailabilityChecksFactory",
    "MainWindow",
    "SettingsDialogFactory",
]
