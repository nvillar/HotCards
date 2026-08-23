"""Restrained three-pane HyperGen application shell."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import ValidationError
from PySide6.QtCore import QSettings, QSignalBlocker, Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QToolBar,
    QToolButton,
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
    RenameCardCommand,
    ReplacePolygonCommand,
    SetRunOverlayModeCommand,
)
from hypergen.application.document_controller import DocumentController, UndoToken
from hypergen.application.document_session import (
    DocumentSession,
    DocumentSessionError,
    DocumentSessionState,
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
    Card,
    HotspotSet,
    Interaction,
    NavigateAction,
    Polygon,
    ResolvedCardReference,
    RunOverlayMode,
    Stack,
    UnresolvedCardReference,
)
from hypergen.generation.ollama_client import OllamaSettings
from hypergen.storage.stack_store import StackStoreError
from hypergen.ui.card_canvas import CardCanvas
from hypergen.ui.card_sidebar import CardSidebar
from hypergen.ui.inspector import Inspector
from hypergen.ui.new_stack_dialog import NewStackDialog
from hypergen.ui.notification_bar import (
    Notification,
    NotificationAction,
    NotificationBar,
    NotificationKind,
)
from hypergen.ui.project_paths import bundle_path, default_project_directory
from hypergen.ui.settings_dialog import (
    MFLUX_MODEL_KEY,
    MFLUX_MODEL_OPTIONS,
    OLLAMA_MODEL_KEY,
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
        project_directory: Path | None = None,
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
        self.project_directory = project_directory or default_project_directory()
        self._availability_checks = dict(availability_checks or {})
        self._availability_checks_factory = availability_checks_factory
        self._settings_dialog_factory = settings_dialog_factory
        self._owns_workers = owns_workers
        self._selected_card_id = (
            controller.document.cards[0].id if controller.document.cards else None
        )
        self._rendered_card_id: UUID | None = None
        self._availability: dict[AdapterKind, bool | None] = {
            AdapterKind.OLLAMA: None,
            AdapterKind.MFLUX: None,
        }
        self._diagnostic_messages: dict[AdapterKind, str] = {}
        self._diagnostic_operations: list[WorkerOperation] = []
        self._diagnostic_generation = 0
        self._service_notification_dismissed = False
        self._undo_notification_token: UndoToken | None = None
        self._card_name_commit_failed = False
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

        self.overlay_label = QLabel("Hotspots")
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

    def _build_panes(self) -> None:
        self.card_sidebar = CardSidebar(self.controller)
        self.card_sidebar.card_selected.connect(self.select_card)
        self.card_sidebar.document_changed.connect(self.render_document)
        self.card_sidebar.delete_requested.connect(self._delete_card)

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
        self.card_header = QHBoxLayout()
        self.canvas_card_name = QLineEdit()
        self.canvas_card_name.setObjectName("canvasCardName")
        self.canvas_card_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas_card_name.setPlaceholderText("Card name")
        self.canvas_card_name.setAccessibleName("Card name")
        self.card_header.addWidget(self.canvas_card_name, 1)
        self.revision_combo = QComboBox()
        self.revision_combo.setObjectName("cardRevisionCombo")
        self.revision_combo.setAccessibleName("Active revision")
        self.revision_combo.setToolTip("Select the active card revision")
        self.card_header.addWidget(self.revision_combo)
        self.add_revision_button = QToolButton()
        self.add_revision_button.setObjectName("addRevisionButton")
        self.add_revision_button.setText("+")
        self.add_revision_button.setAccessibleName("Duplicate revision")
        self.add_revision_button.setToolTip("Duplicate the active revision")
        self.card_header.addWidget(self.add_revision_button)
        self.delete_revision_button = QToolButton()
        self.delete_revision_button.setObjectName("deleteRevisionButton")
        self.delete_revision_button.setText("−")
        self.delete_revision_button.setAccessibleName("Delete revision")
        self.delete_revision_button.setToolTip("Delete the active revision")
        self.card_header.addWidget(self.delete_revision_button)
        canvas_layout.addLayout(self.card_header)
        self.canvas_card_name_error = QLabel()
        self.canvas_card_name_error.setObjectName("canvasCardNameError")
        self.canvas_card_name_error.setWordWrap(True)
        self.canvas_card_name_error.setVisible(False)
        canvas_layout.addWidget(self.canvas_card_name_error)
        self.card_canvas = CardCanvas()
        canvas_layout.addWidget(self.card_canvas, 1)
        self.canvas_fit_controls = QHBoxLayout()
        self.canvas_fit_controls.addStretch(1)
        self.fit_canvas_button = QToolButton()
        self.fit_canvas_button.setObjectName("fitCanvasButton")
        self.fit_canvas_button.setText("⛶")
        self.fit_canvas_button.setAccessibleName("Fit image to window")
        self.fit_canvas_button.setToolTip("Fit image to window")
        self.canvas_fit_controls.addWidget(self.fit_canvas_button)
        canvas_layout.addLayout(self.canvas_fit_controls)
        self.canvas_pages.addWidget(card_canvas)
        self.fit_canvas_button.clicked.connect(self.card_canvas.fit_to_window)
        self.canvas_card_name.editingFinished.connect(self._commit_canvas_card_name)
        self.canvas_card_name.textEdited.connect(self._card_name_edited)
        self.revision_combo.currentIndexChanged.connect(self._revision_selection_changed)
        self.add_revision_button.clicked.connect(self._duplicate_revision)
        self.delete_revision_button.clicked.connect(self._delete_active_revision)

        self.inspector = Inspector(self.controller)
        self.inspector.document_changed.connect(self.render_document)
        self.inspector.render_inputs_changed.connect(self._authoring_inputs_changed)
        self.inspector.enrich_scene_requested.connect(self._enrich_scene)
        self.inspector.generate_background_requested.connect(self._generate_background)
        self.inspector.clear_background_requested.connect(self._clear_background)
        self.inspector.change_applied.connect(self._show_undo_notification)
        self.inspector.hotspot_selected.connect(self.card_canvas.select_interaction)
        self.card_canvas.interaction_selected.connect(self.inspector.select_interaction)
        self.card_canvas.empty_area_requested.connect(self._begin_implicit_hotspot_area)
        self.card_canvas.polygon_created.connect(self._create_hotspot_polygon)
        self.card_canvas.polygon_changed.connect(self._replace_hotspot_polygon)
        self.card_canvas.polygon_deletion_requested.connect(self._delete_hotspot_polygon)
        self.card_canvas.interaction_deletion_requested.connect(self._delete_hotspot_interaction)
        self.card_canvas.editing_error.connect(self.inspector.set_hotspot_error)
        self.card_canvas.interaction_activated.connect(self._run_interaction_activated)
        if self.background_workflow is not None:
            self.background_workflow.busy_changed.connect(
                lambda _busy: self._update_generation_actions()
            )
            self.background_workflow.progress_changed.connect(self._background_progress_changed)
            self.background_workflow.failed.connect(self._background_failed)
            self.background_workflow.document_changed.connect(self.render_document)
            self.background_workflow.change_applied.connect(self._show_undo_notification)
        self.scene_enrichment_workflow.busy_changed.connect(
            lambda _busy: self._update_generation_actions()
        )
        self.scene_enrichment_workflow.progress_changed.connect(
            self._scene_enrichment_progress_changed
        )
        self.scene_enrichment_workflow.failed.connect(self._scene_enrichment_failed)
        self.scene_enrichment_workflow.document_changed.connect(self.render_document)
        self.scene_enrichment_workflow.change_applied.connect(self._show_undo_notification)
        self.pane_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.pane_splitter.setObjectName("threePaneSplitter")
        self.pane_splitter.addWidget(self.card_sidebar)
        self.pane_splitter.addWidget(self.canvas_pages)
        self.pane_splitter.addWidget(self.inspector)
        self.pane_splitter.setStretchFactor(0, 0)
        self.pane_splitter.setStretchFactor(1, 1)
        self.pane_splitter.setStretchFactor(2, 0)
        self.pane_splitter.setSizes([220, 700, 280])

        self.notification_bar = NotificationBar()
        self.notification_bar.action_requested.connect(self._notification_action_requested)
        self.notification_bar.notification_dismissed.connect(self._notification_dismissed)
        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.notification_bar)
        central_layout.addWidget(self.pane_splitter, 1)
        self.setCentralWidget(central_widget)

        self.document_status_label = QLabel()
        self.document_status_label.setObjectName("documentStatusLabel")
        self.statusBar().addWidget(self.document_status_label, 1)
        values = load_machine_settings(self.settings)
        self.llm_model_label = QLabel("LLM")
        self.llm_model_combo = QComboBox()
        self.llm_model_combo.setObjectName("llmModelCombo")
        self.llm_model_combo.setAccessibleName("LLM model")
        self.llm_model_combo.addItem(values.ollama_model, values.ollama_model)
        self.image_model_label = QLabel("Image")
        self.image_model_combo = QComboBox()
        self.image_model_combo.setObjectName("imageModelCombo")
        self.image_model_combo.setAccessibleName("Image model")
        for label, model in MFLUX_MODEL_OPTIONS:
            self.image_model_combo.addItem(label, model)
        self.image_model_combo.setCurrentIndex(
            max(0, self.image_model_combo.findData(values.mflux_model))
        )
        self.statusBar().addPermanentWidget(self.llm_model_label)
        self.statusBar().addPermanentWidget(self.llm_model_combo)
        self.statusBar().addPermanentWidget(self.image_model_label)
        self.statusBar().addPermanentWidget(self.image_model_combo)
        self.llm_model_combo.currentIndexChanged.connect(
            self._llm_model_changed
        )
        self.image_model_combo.currentIndexChanged.connect(
            self._image_model_changed
        )
        self._service_status_detail = "Local AI availability checks pending"
        self.create_first_card_button.clicked.connect(self._primary_empty_action)

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
        snapshot = self.controller.document
        previous_card_id = self._rendered_card_id
        preserve_card_name = self.canvas_card_name.hasFocus()
        card_name_draft = self.canvas_card_name.text()
        if (
            self._undo_notification_token is not None
            and self.controller.current_undo_token != self._undo_notification_token
        ):
            self._undo_notification_token = None
            self.notification_bar.clear_notification("undo")
        card_ids = {card.id for card in snapshot.cards}
        if self._is_running:
            self._selected_card_id = self._run_session.state.current_card_id
        elif self._selected_card_id not in card_ids:
            self._selected_card_id = snapshot.cards[0].id if snapshot.cards else None
        self._rendering = True
        try:
            self.card_sidebar.render(
                snapshot,
                self._selected_card_id,
                draft_card_ids=(),
            )
            self.inspector.render(snapshot, self._selected_card_id)
            selected_card = next(
                (card for card in snapshot.cards if card.id == self._selected_card_id),
                None,
            )
            if selected_card is None:
                self._rendered_card_id = None
                self._card_name_commit_failed = False
                self.canvas_pages.setCurrentIndex(0)
                self.canvas_card_name.clear()
                self._set_canvas_card_name_error("")
                with QSignalBlocker(self.revision_combo):
                    self.revision_combo.clear()
                self.add_revision_button.setEnabled(False)
                self.delete_revision_button.setEnabled(False)
            else:
                if previous_card_id != selected_card.id:
                    self._card_name_commit_failed = False
                    self._set_canvas_card_name_error("")
                self._rendered_card_id = selected_card.id
                self.canvas_pages.setCurrentIndex(1)
                self.canvas_card_name.setText(
                    card_name_draft
                    if preserve_card_name and previous_card_id == selected_card.id
                    else selected_card.name
                )
                active_revision_index = next(
                    index
                    for index, revision in enumerate(selected_card.revisions)
                    if revision.id == selected_card.active_revision_id
                )
                with QSignalBlocker(self.revision_combo):
                    self.revision_combo.clear()
                    for index, revision in enumerate(
                        selected_card.revisions,
                        start=1,
                    ):
                        self.revision_combo.addItem(str(index), revision.id)
                    self.revision_combo.setCurrentIndex(active_revision_index)
                self.add_revision_button.setEnabled(not self._is_running)
                self.delete_revision_button.setEnabled(
                    not self._is_running and len(selected_card.revisions) > 1
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
        self._selected_card_id = selected_card_id
        if self.card_sidebar.selected_card_id != self._selected_card_id:
            self.card_sidebar.select_card(self._selected_card_id)
        self.render_document()

    def _commit_canvas_card_name(self) -> bool:
        if self._rendering or self._selected_card_id is None or self._is_running:
            return True
        if self._card_name_commit_failed:
            self._card_name_commit_failed = False
            return False
        card = next(
            (
                candidate
                for candidate in self.controller.document.cards
                if candidate.id == self._selected_card_id
            ),
            None,
        )
        if card is None:
            return True
        name = self.canvas_card_name.text()
        if name == card.name:
            return True
        try:
            changed = self.controller.execute(RenameCardCommand(card_id=card.id, name=name))
        except (CommandError, ValidationError) as error:
            self._card_name_commit_failed = True
            self._set_canvas_card_name_error(str(error))
            self.canvas_card_name.setText(card.name)
            return False
        self._card_name_commit_failed = False
        self._set_canvas_card_name_error("")
        self.render_document(changed)
        return True

    def _card_name_edited(self) -> None:
        self._card_name_commit_failed = False
        self._set_canvas_card_name_error("")

    def _set_canvas_card_name_error(self, message: str) -> None:
        self.canvas_card_name_error.setText(message)
        self.canvas_card_name_error.setVisible(bool(message))

    def _commit_authoring_metadata(self) -> bool:
        if not self._commit_canvas_card_name():
            return False
        return self.inspector.commit_card_metadata()

    def _revision_selection_changed(self, index: int) -> None:
        if self._rendering or self._selected_card_id is None or index < 0:
            return
        if not self._commit_authoring_metadata():
            self.render_document()
            return
        revision_id = self.revision_combo.itemData(index)
        if isinstance(revision_id, UUID):
            self._activate_revision(revision_id)

    def _duplicate_revision(self) -> None:
        workflow = self.background_workflow
        card_id = self._selected_card_id
        revision_id = self.revision_combo.currentData()
        if (
            workflow is None
            or card_id is None
            or not isinstance(revision_id, UUID)
            or not self._commit_authoring_metadata()
        ):
            return
        self.notification_bar.clear_notification("background-error")
        self.scene_enrichment_workflow.cancel()
        try:
            workflow.duplicate_revision(card_id, revision_id)
        except (BackgroundWorkflowError, CommandError, ValidationError) as error:
            self._show_error(
                "background-error",
                "Could not duplicate revision",
                detail=str(error),
            )

    def _delete_active_revision(self) -> None:
        revision_id = self.revision_combo.currentData()
        if isinstance(revision_id, UUID):
            self._delete_revision(revision_id)

    def _show_undo_notification(self, message: str, token: object) -> None:
        if self._is_running or not isinstance(token, UndoToken):
            return
        self._undo_notification_token = token
        self.notification_bar.show_notification(
            "undo",
            Notification(
                message=message,
                kind=NotificationKind.SUCCESS,
                primary_action=NotificationAction("undo", "Undo"),
                priority=3,
            ),
        )

    def _undo_notification(self) -> None:
        if self._is_running:
            return
        self.scene_enrichment_workflow.cancel()
        token = self._undo_notification_token
        self._undo_notification_token = None
        self.notification_bar.clear_notification("undo")
        if token is not None and self.controller.undo_if_current(token):
            self.render_document()

    def _notification_action_requested(self, action_id: str) -> None:
        if action_id == "undo" and not self._is_running:
            self._undo_notification()
        elif action_id == "open-settings" and not self._is_running:
            self.open_advanced_settings()
        elif action_id == "check-services" and not self._is_running:
            self.run_availability_checks()

    def _notification_dismissed(self, key: str) -> None:
        if key == "undo":
            self._undo_notification_token = None
        elif key == "ai-services":
            self._service_notification_dismissed = True

    def _show_error(
        self,
        key: str,
        message: str,
        *,
        detail: str = "",
    ) -> None:
        self.notification_bar.show_notification(
            key,
            Notification(
                message=message,
                kind=NotificationKind.ERROR,
                detail=detail,
            ),
        )

    def _show_warning(
        self,
        key: str,
        message: str,
        *,
        detail: str = "",
    ) -> None:
        self.notification_bar.show_notification(
            key,
            Notification(
                message=message,
                kind=NotificationKind.WARNING,
                detail=detail,
            ),
        )

    def _show_info(self, key: str, message: str, *, detail: str = "") -> None:
        self.notification_bar.show_notification(
            key,
            Notification(
                message=message,
                kind=NotificationKind.INFO,
                detail=detail,
            ),
        )

    def new_stack(self) -> None:
        """Create and bind a new stack before exposing its initial card."""
        if self.document_session is None:
            return
        self._commit_authoring_metadata()
        dialog = NewStackDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        stack = dialog.stack()
        suggested_path = self.project_directory / f"{stack.name}.hypergen"
        selected_path, _filter = QFileDialog.getSaveFileName(
            self,
            "Create HyperGen Stack",
            str(suggested_path),
            "HyperGen Stack (*.hypergen)",
        )
        if not selected_path:
            return
        try:
            self.document_session.create(stack, self._bundle_path(selected_path))
        except DocumentSessionError as error:
            self._show_document_error("Could Not Create Stack", str(error))

    def open_stack(self) -> None:
        """Open a validated bundle without replacing the current session on failure."""
        if self.document_session is None:
            return
        self._commit_authoring_metadata()
        selected_path = QFileDialog.getExistingDirectory(
            self,
            "Open HyperGen Stack",
            str(self.project_directory),
        )
        if not selected_path:
            return
        try:
            self.document_session.open(Path(selected_path))
        except DocumentSessionError as error:
            self._show_document_error("Could Not Open Stack", str(error))

    def save_document(self) -> bool:
        """Flush accepted mutations and keep a failed save visible."""
        if self.document_session is None:
            return True
        self._commit_authoring_metadata()
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
        self._commit_authoring_metadata()
        selected_path, _filter = QFileDialog.getSaveFileName(
            self,
            "Save HyperGen Stack As",
            str(self.project_directory / f"{self.controller.document.name}.hypergen"),
            "HyperGen Stack (*.hypergen)",
        )
        if not selected_path:
            return
        try:
            self.document_session.save_as(self._bundle_path(selected_path))
        except DocumentSessionError as error:
            self._show_document_error("Could Not Save Stack As", str(error))
            return
        self._cancel_background_generation()
        self.scene_enrichment_workflow.cancel()
        self._undo_notification_token = None
        self.notification_bar.clear_notification("undo")
        self._update_document_actions()

    def undo(self) -> None:
        self.scene_enrichment_workflow.cancel()
        self._undo_notification_token = None
        self.notification_bar.clear_notification("undo")
        if self.controller.undo():
            self.render_document()

    def redo(self) -> None:
        self.scene_enrichment_workflow.cancel()
        self._undo_notification_token = None
        self.notification_bar.clear_notification("undo")
        if self.controller.redo():
            self.render_document()

    def _authoring_inputs_changed(self) -> None:
        self.scene_enrichment_workflow.cancel()
        self._update_generation_actions()

    def _primary_empty_action(self) -> None:
        if self._is_running:
            return
        if self.document_session is not None and self.document_session.store is None:
            self.new_stack()
        else:
            self.card_sidebar.add_card()

    def _delete_card(self, card_id: object) -> None:
        if not isinstance(card_id, UUID):
            return
        card = next(
            (candidate for candidate in self.controller.document.cards if candidate.id == card_id),
            None,
        )
        if card is None:
            return
        was_start_card = self.controller.document.start_card_id == card.id
        inbound_link_count = sum(
            1
            for candidate in self.controller.document.cards
            if candidate.id != card.id
            for revision in candidate.revisions
            if revision.hotspot_set is not None
            for interaction in revision.hotspot_set.interactions
            if isinstance(interaction.action.target, ResolvedCardReference)
            and interaction.action.target.target_card_id == card.id
        )
        if self.background_workflow is not None and self.background_workflow.is_generating_for(
            card.id
        ):
            self._cancel_background_generation()
        previous_token = self.controller.current_undo_token
        self.scene_enrichment_workflow.cancel()
        self.card_sidebar.delete_card(card.id)
        consequences: list[str] = []
        if inbound_link_count:
            noun = "link is" if inbound_link_count == 1 else "links are"
            consequences.append(f"{inbound_link_count} inbound {noun} now unresolved")
        if was_start_card:
            consequences.append("the start card was cleared")
        message = "Card deleted"
        if consequences:
            message += "; " + " and ".join(consequences)
        token = self.controller.current_undo_token
        if token is not None and token != previous_token:
            self._show_undo_notification(message, token)

    def _cancel_background_generation(self) -> None:
        if self.background_workflow is not None and self.background_workflow.busy:
            self.background_workflow.cancel()

    def _document_replaced(self, _document: object) -> None:
        self._cancel_background_generation()
        self.scene_enrichment_workflow.cancel()
        self._undo_notification_token = None
        self._rendered_card_id = None
        self._card_name_commit_failed = False
        self._set_canvas_card_name_error("")
        self.inspector.reset_context()
        self.notification_bar.clear_all()
        if self._is_running:
            state = self._run_session.start(self.controller.document)
            self._selected_card_id = state.current_card_id
        else:
            self._selected_card_id = None
        self.render_document()

    def _session_state_changed(self, state: object) -> None:
        if not isinstance(state, DocumentSessionState):
            return
        if state.error is None:
            self.notification_bar.clear_notification("document-error")
        bound = state.bundle_path is not None
        self.card_sidebar.set_document_editable(self.document_session is None or bound)
        if self.document_session is not None and not bound:
            self.create_first_card_button.setText("Create New Stack")
            self.document_status_label.setText("Create or open a stack")
            self.document_status_label.setToolTip("")
        elif state.error is not None:
            self.create_first_card_button.setText("Create Your First Card")
            self.document_status_label.setText("Save failed")
            self.document_status_label.setToolTip(state.error)
            self._show_error(
                "document-error",
                "Stack could not be saved",
                detail=state.error,
            )
        elif state.dirty:
            self.create_first_card_button.setText("Create Your First Card")
            self.document_status_label.setText("Unsaved changes")
            self.document_status_label.setToolTip("Autosave is pending")
        else:
            self.create_first_card_button.setText("Create Your First Card")
            label = (
                state.bundle_path.name if state.bundle_path is not None else "In-memory document"
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
        self.undo_action.setEnabled(bound and not self._is_running and self.controller.can_undo)
        self.redo_action.setEnabled(bound and not self._is_running and self.controller.can_redo)
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
        return bundle_path(selected_path)

    def _show_document_error(self, title: str, message: str) -> None:
        self._show_error("document-error", title, detail=message)

    def run_availability_checks(self) -> None:
        """Submit injected service checks without blocking the UI thread."""
        if self._is_running or self._diagnostic_operations:
            return
        self._service_notification_dismissed = False
        self.notification_bar.clear_notification("ai-services")
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
        self.notification_bar.clear_notification("background-error")
        self.notification_bar.clear_notification("background-warning")
        self.notification_bar.clear_notification("background-cancelled")
        if not self._commit_authoring_metadata():
            return
        try:
            workflow.generate(card_id)
        except BackgroundWorkflowError as error:
            self._show_error(
                "background-error",
                "Could not generate image",
                detail=str(error),
            )
        self._update_generation_actions()

    def _enrich_scene(self) -> None:
        card_id = self._selected_card_id
        if card_id is None or not self._commit_authoring_metadata():
            return
        self.notification_bar.clear_notification("enrichment-error")
        try:
            self.scene_enrichment_workflow.start(card_id)
        except SceneEnrichmentWorkflowError as error:
            self._show_error(
                "enrichment-error",
                "Could not enrich Description",
                detail=str(error),
            )
        self._update_generation_actions()

    def _scene_enrichment_progress_changed(self, _message: str) -> None:
        self.notification_bar.clear_notification("enrichment-error")
        self._update_generation_actions()

    def _scene_enrichment_failed(self, failure: object) -> None:
        detail = failure.message if isinstance(failure, WorkerFailure) else str(failure)
        self._show_error(
            "enrichment-error",
            "Description enrichment failed",
            detail=detail,
        )
        self._update_generation_actions()

    def _clear_background(self) -> None:
        workflow = self.background_workflow
        card_id = self._selected_card_id
        if workflow is None or card_id is None:
            return
        self.notification_bar.clear_notification("background-error")
        self.scene_enrichment_workflow.cancel()
        try:
            workflow.clear_background(card_id)
        except (
            BackgroundWorkflowError,
            CommandError,
            ValidationError,
        ) as error:
            self._show_error(
                "background-error",
                "Could not clear image",
                detail=str(error),
            )

    def _activate_revision(self, revision_id: object) -> None:
        if (
            self.background_workflow is None
            or self._selected_card_id is None
            or not isinstance(revision_id, UUID)
        ):
            return
        self.notification_bar.clear_notification("background-error")
        self.scene_enrichment_workflow.cancel()
        try:
            self.background_workflow.activate_revision(
                self._selected_card_id,
                revision_id,
            )
        except (BackgroundWorkflowError, CommandError) as error:
            self._show_error(
                "background-error",
                "Could not activate revision",
                detail=str(error),
            )

    def _delete_revision(self, revision_id: object) -> None:
        if (
            self.background_workflow is None
            or self._selected_card_id is None
            or not isinstance(revision_id, UUID)
        ):
            return
        was_generating = self.background_workflow.is_generating_for(self._selected_card_id)
        self.notification_bar.clear_notification("background-error")
        self.scene_enrichment_workflow.cancel()
        try:
            self.background_workflow.delete_revision(
                self._selected_card_id,
                revision_id,
            )
        except (BackgroundWorkflowError, CommandError) as error:
            self._show_error(
                "background-error",
                "Could not delete revision",
                detail=str(error),
            )
            return
        if was_generating:
            self._cancel_background_generation()

    def _background_progress_changed(self, message: str) -> None:
        self.notification_bar.clear_notification("background-error")
        self.notification_bar.clear_notification("background-warning")
        if message == "Generation cancelled":
            self._show_info("background-cancelled", message)
        else:
            self.notification_bar.clear_notification("background-cancelled")
        self._update_generation_actions()

    def _background_failed(self, failure: object) -> None:
        if isinstance(failure, WorkerFailure):
            self._show_error(
                "background-error",
                "Image generation failed",
                detail=failure.message,
            )
        else:
            self._show_error(
                "background-error",
                "Image generation failed",
                detail=str(failure),
            )
        self._update_generation_actions()

    def _availability_check_succeeded(
        self,
        adapter: AdapterKind,
        generation: int,
        result: object,
    ) -> None:
        if generation != self._diagnostic_generation:
            return
        if adapter is AdapterKind.OLLAMA and isinstance(result, (list, tuple)):
            models = tuple(
                model for model in result if isinstance(model, str) and model
            )
            self._set_installed_ollama_models(models)
            selected = load_machine_settings(self.settings).ollama_model
            if selected not in models:
                self.apply_availability_diagnostic(
                    AvailabilityDiagnostic(
                        adapter=adapter,
                        available=False,
                        message=(
                            f"Ollama model {selected!r} is not installed. "
                            f"Run `ollama pull {selected}` or select an installed model."
                        ),
                    )
                )
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
        ollama_available = self._availability[AdapterKind.OLLAMA] is True
        workflow_busy = (
            self.background_workflow.busy if self.background_workflow is not None else False
        )
        selected_card = next(
            (card for card in self.controller.document.cards if card.id == self._selected_card_id),
            None,
        )
        active_revision = selected_card.active_revision if selected_card is not None else None
        has_image = active_revision is not None and active_revision.background is not None
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
            generate_reason = "Enter a Description before generating"
        elif not mflux_available:
            generate_reason = self._action_diagnostic(AdapterKind.MFLUX)
        self.inspector.set_background_capabilities(
            can_generate=(has_card and has_render_prompt and mflux_available),
            generate_reason=generate_reason,
            can_clear=has_card and has_image,
            clear_reason=("Clear the current image" if has_image else "This revision has no image"),
            busy=workflow_busy,
            generating=workflow_busy,
        )
        enrichment_busy = self.scene_enrichment_workflow.busy
        has_enrichment_input = self.inspector.has_description_input()
        can_enrich = has_card and has_enrichment_input and ollama_available and not enrichment_busy
        enrich_reason = "Ready to enrich Description"
        if not has_card:
            enrich_reason = "Select a card in a saved stack"
        elif not has_enrichment_input:
            enrich_reason = "Enter a Description before enriching"
        elif enrichment_busy:
            enrich_reason = "Description enrichment is running"
        elif not ollama_available:
            enrich_reason = self._action_diagnostic(AdapterKind.OLLAMA)
        self.inspector.set_scene_enrichment_capabilities(
            can_enrich=can_enrich,
            reason=enrich_reason,
            busy=enrichment_busy,
        )
        authoring = not self._is_running
        self.llm_model_combo.setEnabled(authoring and not enrichment_busy)
        self.image_model_combo.setEnabled(authoring and not workflow_busy)
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
        self._service_status_detail = "\n".join(
            self._diagnostic_messages.get(
                adapter,
                f"{adapter.value} availability check pending",
            )
            for adapter in AdapterKind
        )
        self.llm_model_combo.setToolTip(self._service_status_detail)
        image_license = (
            "\nFLUX.2 Klein 9B KV is licensed for non-commercial use."
            if self.image_model_combo.currentData() == "flux2-klein-9b-kv"
            else ""
        )
        self.image_model_combo.setToolTip(
            self._service_status_detail + image_license
        )
        if (
            unavailable
            and not pending
            and not self._is_running
            and not self._service_notification_dismissed
        ):
            self.notification_bar.show_notification(
                "ai-services",
                Notification(
                    message=summary,
                    kind=NotificationKind.WARNING,
                    detail=self._service_status_detail,
                    primary_action=NotificationAction(
                        "open-settings",
                        "Settings",
                    ),
                    secondary_action=NotificationAction(
                        "check-services",
                        "Check Again",
                    ),
                ),
            )
        elif not unavailable:
            self._service_notification_dismissed = False
            self.notification_bar.clear_notification("ai-services")
        elif self._is_running:
            self.notification_bar.clear_notification("ai-services")

    def _set_installed_ollama_models(self, models: tuple[str, ...]) -> None:
        selected = load_machine_settings(self.settings).ollama_model
        with QSignalBlocker(self.llm_model_combo):
            self.llm_model_combo.clear()
            for model in models:
                self.llm_model_combo.addItem(model, model)
            if selected not in models:
                self.llm_model_combo.addItem(
                    f"{selected} (not installed)",
                    selected,
                )
            self.llm_model_combo.setCurrentIndex(
                self.llm_model_combo.findData(selected)
            )

    def _llm_model_changed(self, index: int) -> None:
        model = self.llm_model_combo.itemData(index)
        if not isinstance(model, str) or not model:
            return
        if model == load_machine_settings(self.settings).ollama_model:
            return
        self.scene_enrichment_workflow.cancel()
        self.settings.setValue(OLLAMA_MODEL_KEY, model)
        self.settings.sync()
        self._restart_availability_checks()

    def _image_model_changed(self, index: int) -> None:
        model = self.image_model_combo.itemData(index)
        if not isinstance(model, str) or not model:
            return
        if model == load_machine_settings(self.settings).mflux_model:
            return
        self._cancel_background_generation()
        self.settings.setValue(MFLUX_MODEL_KEY, model)
        self.settings.sync()
        self._restart_availability_checks()

    def _render_card_canvas(self, card: object) -> None:
        if not isinstance(card, Card):
            return
        self.card_canvas.set_canvas_size(self.controller.document.canvas)
        revision = card.active_revision
        if revision.background is None:
            self.card_canvas.show_message("No image")
            if self._is_running:
                self._set_run_warning(f'"{card.name}" has no image in this revision.')
            return
        if self.document_session is None or self.document_session.store is None:
            self.card_canvas.show_message("Background bundle is unavailable")
            if self._is_running:
                self._set_run_warning(f'The background for "{card.name}" is unavailable.')
            return
        try:
            asset_path = self.document_session.store.asset_path(revision.background.image_path)
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
            if revision.hotspot_set is None or not revision.hotspot_set.interactions:
                self._set_run_warning(f'"{card.name}" has no hotspots to navigate.')
            return
        self.card_canvas.set_hotspots(
            revision.hotspot_set,
            self.inspector.selected_interaction_id,
            editable=True,
            context_id=revision.id,
        )

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
        context = self._active_hotspot_context()
        if context is None or not isinstance(polygon, Polygon):
            return
        card_id, revision_id, hotspot_set = context
        if isinstance(interaction_id, UUID):
            interaction = next(
                (
                    interaction
                    for interaction in hotspot_set.interactions
                    if interaction.id == interaction_id
                ),
                None,
            )
            if interaction is None:
                self.inspector.set_hotspot_error("The hotspot being edited no longer exists.")
                self.render_document()
                return
            command = AddPolygonCommand(
                card_id=card_id,
                revision_id=revision_id,
                interaction_id=interaction_id,
                polygon=polygon,
            )
            selected_id = interaction_id
            selected_polygon_index = len(interaction.polygons)
        else:
            interaction = Interaction(
                action=NavigateAction(target=UnresolvedCardReference()),
                polygons=(polygon,),
            )
            command = AddInteractionCommand(
                card_id=card_id,
                revision_id=revision_id,
                interaction=interaction,
            )
            selected_id = interaction.id
            selected_polygon_index = 0
        self._execute_hotspot_canvas_command(
            command,
            selected_id,
            selected_polygon_index=selected_polygon_index,
            undo_message="Hotspot area added",
        )

    def _begin_implicit_hotspot_area(self, point: object) -> None:
        from hypergen.domain.models import Point

        if not isinstance(point, Point):
            return
        context = self._active_hotspot_context()
        if context is None:
            return
        card_id, revision_id, hotspot_set = context
        interaction_id = self.inspector.selected_interaction_id
        if interaction_id is None:
            interaction = Interaction(
                action=NavigateAction(target=UnresolvedCardReference()),
            )
            interaction_id = interaction.id
            self._execute_hotspot_canvas_command(
                AddInteractionCommand(
                    card_id=card_id,
                    revision_id=revision_id,
                    interaction=interaction,
                ),
                interaction_id,
            )
        self.card_canvas.begin_polygon(
            interaction_id,
            initial_point=point,
        )

    def _replace_hotspot_polygon(
        self,
        interaction_id: object,
        polygon_index: int,
        polygon: object,
    ) -> None:
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
            undo_message="Hotspot geometry updated",
        )

    def _delete_hotspot_polygon(
        self,
        interaction_id: object,
        polygon_index: int,
    ) -> None:
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
            undo_message="Hotspot area deleted",
        )

    def _delete_hotspot_interaction(self, interaction_id: object) -> None:
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
            undo_message="Hotspot deleted",
        )

    def _execute_hotspot_canvas_command(
        self,
        command: DocumentCommand,
        selected_interaction_id: UUID | None,
        *,
        selected_polygon_index: int | None = None,
        undo_message: str | None = None,
    ) -> None:
        previous_undo_token = self.controller.current_undo_token
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
        if selected_interaction_id is not None and selected_polygon_index is not None:
            self.card_canvas.select_polygon(
                selected_interaction_id,
                selected_polygon_index,
            )
        undo_token = self.controller.current_undo_token
        if (
            undo_message is not None
            and undo_token is not None
            and undo_token != previous_undo_token
        ):
            self._show_undo_notification(undo_message, undo_token)

    def _active_hotspot_context(
        self,
    ) -> tuple[UUID, UUID, HotspotSet] | None:
        card = next(
            (card for card in self.controller.document.cards if card.id == self._selected_card_id),
            None,
        )
        if card is None:
            self.inspector.set_hotspot_error("Select a card before editing hotspots.")
            return None
        revision = card.active_revision
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
            if not self._commit_authoring_metadata():
                with QSignalBlocker(self.mode_selector):
                    self.mode_selector.setCurrentText("Author")
                return
            self.card_canvas.cancel_drawing()
            self._cancel_ai_activity_for_run()
            self._undo_notification_token = None
            self.notification_bar.clear_all()
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
        self.canvas_card_name.setReadOnly(not authoring)
        self.revision_combo.setEnabled(authoring)
        self.add_revision_button.setVisible(authoring)
        self.delete_revision_button.setVisible(authoring)
        self.card_sidebar.setVisible(authoring)
        self.inspector.setVisible(authoring)
        self.fit_canvas_button.setVisible(authoring)
        self.create_first_card_button.setVisible(authoring)
        self.empty_canvas_title.setText(
            "Create your first card" if authoring else "No cards to run"
        )
        self.empty_canvas_description.setText(
            "Cards are the scenes readers visit. Start with one, then add its background and links."
            if authoring
            else "Switch to Author mode to create the first card."
        )
        self.overlay_label.setVisible(True)
        self.overlay_selector.setVisible(True)
        for action in self.player_navigation_actions:
            action.setVisible(self._is_running)

    def _update_run_actions(self) -> None:
        state = self._run_session.state
        self.back_action.setEnabled(self._is_running and bool(state.history))
        self.restart_action.setEnabled(self._is_running and state.current_card_id is not None)

    def _set_run_warning(self, message: str) -> None:
        if message:
            self._show_warning("run-warning", message)
        else:
            self.notification_bar.clear_notification("run-warning")

    def _cancel_ai_activity_for_run(self) -> None:
        self._cancel_diagnostics()
        self.scene_enrichment_workflow.cancel()
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

    def _ask_retry_failed_close_save(self, message: str) -> bool:
        dialog = QMessageBox(
            QMessageBox.Icon.Critical,
            "Stack Not Saved",
            message,
            parent=self,
        )
        retry_button = dialog.addButton(
            "Retry Saving",
            QMessageBox.ButtonRole.AcceptRole,
        )
        keep_working_button = dialog.addButton(
            "Keep Working",
            QMessageBox.ButtonRole.RejectRole,
        )
        dialog.setDefaultButton(retry_button)
        dialog.setEscapeButton(keep_working_button)
        dialog.exec()
        return dialog.clickedButton() is retry_button

    def closeEvent(self, event: QCloseEvent) -> None:
        self._commit_authoring_metadata()
        if self.document_session is not None and not self.document_session.flush():
            if self._ask_retry_failed_close_save(
                (self.document_session.state.error or "The stack could not be saved.")
                + "\n\nRetry saving before closing?",
            ):
                if not self.document_session.flush():
                    event.ignore()
                    return
            else:
                event.ignore()
                return
        self._cancel_background_generation()
        if self.background_workflow is not None:
            self.background_workflow.close()
        self.scene_enrichment_workflow.close()
        if self._owns_workers:
            self.workers.shutdown(wait_milliseconds=100)
        super().closeEvent(event)


__all__ = [
    "AvailabilityChecks",
    "AvailabilityChecksFactory",
    "MainWindow",
    "SettingsDialogFactory",
]
