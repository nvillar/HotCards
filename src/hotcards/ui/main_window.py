"""Restrained three-pane HotCards application shell."""

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
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStyle,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from hotcards.application.background_workflow import (
    BackgroundGenerationSettings,
    BackgroundWorkflow,
    BackgroundWorkflowError,
)
from hotcards.application.card_duplication import (
    CardDuplicationError,
    CardDuplicationWorkflow,
)
from hotcards.application.commands import (
    AddInteractionCommand,
    CommandError,
    CreateGeneratedRevisionCommand,
    DeleteInteractionCommand,
    DocumentCommand,
    RenameCardCommand,
    ReplacePolygonCommand,
    SetRunOverlayModeCommand,
)
from hotcards.application.document_controller import (
    PENDING_DURABILITY_MESSAGE,
    DocumentController,
    DocumentMutationBlockedError,
    UndoToken,
)
from hotcards.application.document_session import (
    DocumentSession,
    DocumentSessionError,
    DocumentSessionState,
)
from hotcards.application.generated_revision_change import (
    EditedRevisionChange,
    GeneratedRevisionChange,
)
from hotcards.application.run_session import RunSession, RunSessionState
from hotcards.application.sound_player import QtSoundPlayer, SoundPlayer
from hotcards.application.sound_workflow import SoundWorkflow
from hotcards.application.workers import (
    AdapterKind,
    AdapterWorkers,
    AvailabilityDiagnostic,
    WorkerFailure,
    WorkerOperation,
)
from hotcards.domain.models import (
    Card,
    CurrentSourceSize,
    HotspotSet,
    Interaction,
    Polygon,
    PresetOutputSize,
    RefineTransformation,
    ResolvedCardReference,
    RunOverlayMode,
    Stack,
)
from hotcards.storage.stack_store import StackStoreError
from hotcards.ui.card_canvas import CardCanvas
from hotcards.ui.card_sidebar import CardSidebar
from hotcards.ui.inspector import Inspector
from hotcards.ui.new_stack_dialog import NewStackDialog
from hotcards.ui.notification_bar import (
    Notification,
    NotificationAction,
    NotificationBar,
    NotificationKind,
)
from hotcards.ui.project_paths import bundle_path, default_project_directory
from hotcards.ui.settings_dialog import (
    SettingsDialog,
    SettingsStore,
    load_machine_settings,
)
from hotcards.ui.utility_windows import (
    UTILITY_WINDOW_HEIGHT,
    UTILITY_WINDOW_WIDTH,
    KeyManagerWindow,
    SoundManagerWindow,
    StyleManagerWindow,
)

AvailabilityChecks = Mapping[AdapterKind, Callable[[], Any]]
AvailabilityChecksFactory = Callable[[], AvailabilityChecks]
SettingsDialogFactory = Callable[[SettingsStore, QWidget], QDialog]


class MainWindow(QMainWindow):
    """Application shell whose panes render one DocumentController."""

    _STYLE_MANAGER_GEOMETRY_KEY = "windows/style_manager_geometry"
    _SOUND_MANAGER_GEOMETRY_KEY = "windows/sound_manager_geometry"
    _KEY_MANAGER_GEOMETRY_KEY = "windows/key_manager_geometry"

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
        sound_workflow: SoundWorkflow | None = None,
        sound_player: SoundPlayer | None = None,
        card_duplication_workflow: CardDuplicationWorkflow | None = None,
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
        self.sound_workflow = sound_workflow
        self.sound_player = sound_player or QtSoundPlayer(parent=self)
        self.card_duplication_workflow = card_duplication_workflow
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
            AdapterKind.MFLUX: None,
        }
        self._diagnostic_messages: dict[AdapterKind, str] = {}
        self._diagnostic_operations: list[WorkerOperation] = []
        self._diagnostic_generation = 0
        self._service_notification_dismissed = False
        self._undo_notification_token: UndoToken | None = None
        self._generated_revision_change: GeneratedRevisionChange | None = None
        self._edit_undo_instructions: dict[UndoToken, str] = {}
        self._card_selection_history: dict[
            UndoToken,
            tuple[UUID | None, UUID | None],
        ] = {}
        self._card_name_commit_failed = False
        self._rendering = False
        self._is_running = False
        self._background_step_progress: tuple[int, int] | None = None
        self._background_progress_message = ""
        self._last_session_mutation_blocked = False
        self._run_session = RunSession()
        self.style_manager_window: StyleManagerWindow | None = None
        self.sound_manager_window: SoundManagerWindow | None = None
        self.key_manager_window: KeyManagerWindow | None = None
        if self.background_workflow is None and self.document_session is not None:
            self.background_workflow = BackgroundWorkflow(
                controller,
                self.document_session,
                workers,
                self._background_generation_settings,
                parent=self,
            )
        if self.card_duplication_workflow is None and self.document_session is not None:
            self.card_duplication_workflow = CardDuplicationWorkflow(
                controller,
                self.document_session,
            )
        self.setWindowTitle(f"HotCards — {controller.document.name}")
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
        self.authoring_toolbar = toolbar
        self.addToolBar(toolbar)

        self.toolbar_leading_spacer = QWidget()
        self.toolbar_leading_spacer.setFixedWidth(8)
        toolbar.addWidget(self.toolbar_leading_spacer)
        self.mode_button = QPushButton("Run")
        self.mode_button.setObjectName("modeButton")
        self.mode_button.setAccessibleName("Switch mode")
        self.mode_button.setToolTip("Switch to Run mode")
        self.mode_button.clicked.connect(self._toggle_mode)
        self.mode_button_action = toolbar.addWidget(self.mode_button)
        self.run_controls_separator = toolbar.addSeparator()

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
        self.back_button = QPushButton("Back")
        self.back_button.setObjectName("runBackButton")
        self.back_button.setToolTip(self.back_action.toolTip())
        self.back_button.clicked.connect(self.back_action.trigger)
        self.back_button.setEnabled(False)
        self.back_button_action = toolbar.addWidget(self.back_button)
        self.back_button_action.setVisible(False)
        self.restart_button = QPushButton("Restart")
        self.restart_button.setObjectName("runRestartButton")
        self.restart_button.setToolTip(self.restart_action.toolTip())
        self.restart_button.clicked.connect(self.restart_action.trigger)
        self.restart_button.setEnabled(False)
        self.restart_button_action = toolbar.addWidget(self.restart_button)
        self.restart_button_action.setVisible(False)
        self.run_overlay_separator = toolbar.addSeparator()
        self.run_overlay_separator.setVisible(False)

        self.overlay_label = QLabel("Hotspots")
        self.overlay_label_action = toolbar.addWidget(self.overlay_label)
        self.overlay_selector = QComboBox()
        self.overlay_selector.setObjectName("overlaySelector")
        for mode, label in (
            (RunOverlayMode.HIDDEN, "Hidden"),
            (RunOverlayMode.HOVER, "On hover"),
            (RunOverlayMode.VISIBLE, "Visible"),
        ):
            self.overlay_selector.addItem(label, mode)
        self.overlay_selector.currentIndexChanged.connect(self._overlay_changed)
        self.overlay_selector_action = toolbar.addWidget(self.overlay_selector)
        self.run_controls_separator.setVisible(False)
        self.overlay_label_action.setVisible(False)
        self.overlay_selector_action.setVisible(False)

        self.author_action_spacer = QWidget()
        self.author_action_spacer.setObjectName("authorActionSpacer")
        self.author_action_spacer.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self.author_action_spacer_action = toolbar.addWidget(self.author_action_spacer)
        self.styles_button = QPushButton("Styles")
        self.styles_button.setObjectName("stylesManagerButton")
        self.styles_button.setAccessibleName("Open Styles manager")
        self.styles_button.setToolTip("Open the stack Styles manager")
        self.styles_button.clicked.connect(self._show_style_manager)
        self.styles_button_action = toolbar.addWidget(self.styles_button)
        self.sounds_button = QPushButton("Sounds")
        self.sounds_button.setObjectName("soundsManagerButton")
        self.sounds_button.setAccessibleName("Open Sounds manager")
        self.sounds_button.setToolTip("Open the stack Sounds manager")
        self.sounds_button.clicked.connect(self._show_sound_manager)
        self.sounds_button_action = toolbar.addWidget(self.sounds_button)
        self.keys_button = QPushButton("Keys")
        self.keys_button.setObjectName("keysManagerButton")
        self.keys_button.setAccessibleName("Open Keys manager")
        self.keys_button.setToolTip("Open the stack Keys manager")
        self.keys_button.clicked.connect(self._show_key_manager)
        self.keys_button_action = toolbar.addWidget(self.keys_button)
        self.toolbar_trailing_spacer = QWidget()
        self.toolbar_trailing_spacer.setFixedWidth(self.toolbar_leading_spacer.width())
        self.toolbar_trailing_spacer_action = toolbar.addWidget(self.toolbar_trailing_spacer)

    def _show_style_manager(self) -> None:
        if self._is_running:
            return
        window = self.style_manager_window
        if window is None:
            window = StyleManagerWindow(self.controller, self)
            self.style_manager_window = window
            self._attach_utility_actions(window)
            window.document_changed.connect(
                lambda document, source=window: self.render_document(
                    document,
                    utility_source=source,
                )
            )
            window.change_applied.connect(self._show_undo_notification)
            window.inputs_changed.connect(self._authoring_inputs_changed)
            window.closing.connect(
                lambda geometry: self.settings.setValue(
                    self._STYLE_MANAGER_GEOMETRY_KEY,
                    geometry,
                )
            )
            window.destroyed.connect(
                lambda _object=None, source=window: self._utility_destroyed(source)
            )
            geometry = self.settings.value(self._STYLE_MANAGER_GEOMETRY_KEY)
            if geometry is not None:
                window.restoreGeometry(geometry)
            window.resize(UTILITY_WINDOW_WIDTH, UTILITY_WINDOW_HEIGHT)
        window.set_mutation_allowed(not self.controller.mutation_blocked)
        window.render(self.controller.document)
        window.show()
        window.raise_()
        window.activateWindow()

    def _show_sound_manager(self) -> None:
        if self._is_running or self.sound_workflow is None:
            return
        window = self.sound_manager_window
        if window is None:
            window = SoundManagerWindow(
                self.controller,
                self.sound_workflow,
                self.sound_player,
                self._resolve_sound_asset_path,
                self,
            )
            self.sound_manager_window = window
            self._attach_utility_actions(window)
            window.document_changed.connect(
                lambda document, source=window: self.render_document(
                    document,
                    utility_source=source,
                )
            )
            window.change_applied.connect(self._show_undo_notification)
            window.hotspot_usage_requested.connect(self._show_hotspot_usage)
            window.closing.connect(
                lambda geometry: self.settings.setValue(
                    self._SOUND_MANAGER_GEOMETRY_KEY,
                    geometry,
                )
            )
            window.destroyed.connect(
                lambda _object=None, source=window: self._utility_destroyed(source)
            )
            geometry = self.settings.value(self._SOUND_MANAGER_GEOMETRY_KEY)
            if geometry is not None:
                window.restoreGeometry(geometry)
            window.resize(UTILITY_WINDOW_WIDTH, UTILITY_WINDOW_HEIGHT)
        window.set_mutation_allowed(not self.controller.mutation_blocked)
        window.render(self.controller.document)
        window.show()
        window.raise_()
        window.activateWindow()

    def _show_key_manager(self) -> None:
        if self._is_running:
            return
        window = self.key_manager_window
        if window is None:
            window = KeyManagerWindow(self.controller, self)
            self.key_manager_window = window
            self._attach_utility_actions(window)
            window.document_changed.connect(
                lambda document, source=window: self.render_document(
                    document,
                    utility_source=source,
                )
            )
            window.change_applied.connect(self._show_undo_notification)
            window.hotspot_usage_requested.connect(self._show_hotspot_usage)
            window.closing.connect(
                lambda geometry: self.settings.setValue(
                    self._KEY_MANAGER_GEOMETRY_KEY,
                    geometry,
                )
            )
            window.destroyed.connect(
                lambda _object=None, source=window: self._utility_destroyed(source)
            )
            geometry = self.settings.value(self._KEY_MANAGER_GEOMETRY_KEY)
            if geometry is not None:
                window.restoreGeometry(geometry)
            window.resize(UTILITY_WINDOW_WIDTH, UTILITY_WINDOW_HEIGHT)
        window.set_mutation_allowed(not self.controller.mutation_blocked)
        window.render(self.controller.document)
        window.show()
        window.raise_()
        window.activateWindow()

    def _utility_destroyed(self, window: QWidget) -> None:
        if self.style_manager_window is window:
            self.style_manager_window = None
        if self.sound_manager_window is window:
            self.sound_manager_window = None
        if self.key_manager_window is window:
            self.key_manager_window = None

    def _attach_utility_actions(self, window: QWidget) -> None:
        window.addActions(
            [
                self.new_stack_action,
                self.open_stack_action,
                self.save_action,
                self.save_as_action,
                self.undo_action,
                self.redo_action,
                self.duplicate_card_action,
            ]
        )

    def _render_utility_windows(
        self,
        *,
        skip: QWidget | None = None,
    ) -> None:
        for window in (
            self.style_manager_window,
            self.sound_manager_window,
            self.key_manager_window,
        ):
            if window is not None and window is not skip:
                window.render(self.controller.document)

    def _close_utility_windows(self, *, commit_pending: bool = True) -> None:
        for attribute in (
            "style_manager_window",
            "sound_manager_window",
            "key_manager_window",
        ):
            window = getattr(self, attribute)
            if window is None:
                continue
            closed = window.close() if commit_pending else window.close_without_committing()
            if closed:
                setattr(self, attribute, None)

    def _build_panes(self) -> None:
        self.card_sidebar = CardSidebar(
            self.controller,
            image_path_resolver=self._resolve_revision_image_path,
        )
        self.card_sidebar.card_selected.connect(self.select_card)
        self.card_sidebar.document_changed.connect(self.render_document)
        self.card_sidebar.duplicate_requested.connect(self._duplicate_card)
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
        self.revision_label = QLabel("Version:")
        self.revision_label.setObjectName("cardRevisionLabel")
        self.card_header.addWidget(self.revision_label)
        self.revision_combo = QComboBox()
        self.revision_combo.setObjectName("cardRevisionCombo")
        self.revision_combo.setAccessibleName("Active revision")
        self.revision_combo.setToolTip("Select the active card revision")
        revision_width = (
            self.revision_combo.fontMetrics().horizontalAdvance("0000")
            + self.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
            + 24
        )
        self.revision_combo.setFixedWidth(revision_width)
        self.card_header.addWidget(self.revision_combo)
        self.add_revision_button = QToolButton()
        self.add_revision_button.setObjectName("addRevisionButton")
        self.add_revision_button.setText("+")
        self.add_revision_button.setAccessibleName("Duplicate revision")
        self.add_revision_button.setToolTip("Duplicate the active revision")
        self.delete_revision_button = QToolButton()
        self.delete_revision_button.setObjectName("deleteRevisionButton")
        self.delete_revision_button.setText("−")
        self.delete_revision_button.setAccessibleName("Delete revision")
        self.delete_revision_button.setToolTip("Delete the active revision")
        self.card_header.addWidget(self.delete_revision_button)
        self.card_header.addWidget(self.add_revision_button)
        canvas_layout.addLayout(self.card_header)
        self.canvas_card_name_error = QLabel()
        self.canvas_card_name_error.setObjectName("canvasCardNameError")
        self.canvas_card_name_error.setWordWrap(True)
        self.canvas_card_name_error.setVisible(False)
        canvas_layout.addWidget(self.canvas_card_name_error)
        self.card_canvas = CardCanvas()
        canvas_layout.addWidget(self.card_canvas, 1)
        self.canvas_pages.addWidget(card_canvas)
        self.canvas_card_name.editingFinished.connect(self._commit_canvas_card_name)
        self.canvas_card_name.textEdited.connect(self._card_name_edited)
        self.revision_combo.currentIndexChanged.connect(self._revision_selection_changed)
        self.add_revision_button.clicked.connect(self._duplicate_revision)
        self.delete_revision_button.clicked.connect(self._delete_active_revision)

        self.inspector = Inspector(self.controller)
        self.inspector.document_changed.connect(self.render_document)
        self.inspector.render_inputs_changed.connect(self._authoring_inputs_changed)
        self.inspector.inspector_tabs.currentChanged.connect(self._inspector_tab_changed)
        self.inspector.generate_background_requested.connect(self._generate_background)
        self.inspector.refine_background_requested.connect(self._refine_background)
        self.inspector.edit_background_requested.connect(self._edit_background)
        self.inspector.change_applied.connect(self._show_undo_notification)
        self.inspector.hotspot_selected.connect(self.card_canvas.select_interaction)
        self.inspector.hotspot_drawing_requested.connect(self.card_canvas.begin_polygon)
        self.card_canvas.interaction_selected.connect(self.inspector.select_interaction)
        self.card_canvas.polygon_created.connect(self._create_hotspot_polygon)
        self.card_canvas.polygon_changed.connect(self._replace_hotspot_polygon)
        self.card_canvas.interaction_deletion_requested.connect(self._delete_hotspot_interaction)
        self.card_canvas.editing_error.connect(self.inspector.set_hotspot_error)
        self.card_canvas.interaction_activated.connect(self._run_interaction_activated)
        if self.background_workflow is not None:
            self.background_workflow.busy_changed.connect(self._background_activity_changed)
            invocation_active_changed = getattr(
                self.background_workflow,
                "invocation_active_changed",
                None,
            )
            if invocation_active_changed is not None:
                invocation_active_changed.connect(self._background_activity_changed)
            self.background_workflow.progress_changed.connect(self._background_progress_changed)
            self.background_workflow.generation_progress_changed.connect(
                self._background_generation_progress_changed
            )
            self.background_workflow.failed.connect(self._background_failed)
            self.background_workflow.document_changed.connect(self.render_document)
            self.background_workflow.change_applied.connect(self._show_undo_notification)
            self.background_workflow.generation_applied.connect(
                self._show_generated_revision_notification
            )
            edit_applied = getattr(self.background_workflow, "edit_applied", None)
            if edit_applied is not None:
                edit_applied.connect(self._remember_edit_undo_instruction)
            edit_instruction_clear_requested = getattr(
                self.background_workflow,
                "edit_instruction_clear_requested",
                None,
            )
            if edit_instruction_clear_requested is not None:
                edit_instruction_clear_requested.connect(self.inspector.clear_edit_instruction)
        self.pane_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.pane_splitter.setObjectName("threePaneSplitter")
        self.pane_splitter.addWidget(self.card_sidebar)
        self.pane_splitter.addWidget(self.canvas_pages)
        self.pane_splitter.addWidget(self.inspector)
        self.pane_splitter.setStretchFactor(0, 0)
        self.pane_splitter.setStretchFactor(1, 1)
        self.pane_splitter.setStretchFactor(2, 0)
        self.pane_splitter.setSizes([240, 680, 280])

        self.notification_bar = NotificationBar()
        self.notification_bar.action_requested.connect(self._notification_action_requested)
        self.notification_bar.notification_dismissed.connect(self._notification_dismissed)
        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.pane_splitter, 1)
        central_layout.addWidget(self.notification_bar)
        self.setCentralWidget(central_widget)

        self.generation_progress_bar = QProgressBar()
        self.generation_progress_bar.setObjectName("generationProgressBar")
        self.generation_progress_bar.setAccessibleName("Generation progress")
        self.generation_progress_bar.setRange(0, 0)
        self.generation_progress_bar.setTextVisible(False)
        self.generation_progress_container = QWidget()
        self.generation_progress_container.setObjectName("generationProgressContainer")
        self.generation_progress_layout = QHBoxLayout(self.generation_progress_container)
        self.generation_progress_layout.setContentsMargins(8, 0, 8, 0)
        self.generation_step_label = QLabel()
        self.generation_step_label.setObjectName("generationStepLabel")
        self.generation_step_label.setAccessibleName("Current generation step")
        self.generation_progress_layout.addWidget(self.generation_step_label)
        self.generation_progress_layout.addSpacing(8)
        self.generation_progress_layout.addWidget(
            self.generation_progress_bar,
            1,
        )
        self.generation_progress_layout.addSpacing(8)
        self.cancel_generation_button = QPushButton("Cancel")
        self.cancel_generation_button.setObjectName("cancelGenerationButton")
        self.cancel_generation_button.setAccessibleName("Cancel generation")
        self.cancel_generation_button.setToolTip("Cancel image generation")
        self.cancel_generation_button.clicked.connect(self._cancel_generation_activity)
        self.generation_progress_layout.addWidget(self.cancel_generation_button)
        self.generation_progress_container.hide()
        self.statusBar().addWidget(self.generation_progress_container, 1)
        self._service_status_detail = "MFLUX availability check pending"
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
        edit_menu.addSeparator()
        self.duplicate_card_action = QAction("Duplicate Card", self)
        self.duplicate_card_action.setObjectName("duplicateCardAction")
        self.duplicate_card_action.setShortcut(QKeySequence("Ctrl+D"))
        self.duplicate_card_action.triggered.connect(self._duplicate_card)
        edit_menu.addAction(self.duplicate_card_action)

        self.settings_menu = self.menuBar().addMenu("Settings")
        self.settings_menu.setObjectName("settingsMenu")
        self.models_action = QAction("Models…", self)
        self.models_action.setObjectName("modelsAction")
        self.models_action.triggered.connect(self.open_model_settings)
        self.settings_menu.addAction(self.models_action)

    def render_document(
        self,
        _document: Stack | None = None,
        *,
        render_sidebar: bool = True,
        utility_source: QWidget | None = None,
    ) -> None:
        """Refresh all panes from the controller's authoritative snapshot."""
        snapshot = self.controller.document
        previous_card_id = self._rendered_card_id
        preserve_card_name = self.canvas_card_name.hasFocus()
        card_name_draft = self.canvas_card_name.text()
        if (
            self._undo_notification_token is not None
            and self.controller.current_undo_token != self._undo_notification_token
        ):
            self._clear_undo_notification()
        card_ids = {card.id for card in snapshot.cards}
        if self._is_running:
            self._selected_card_id = self._run_session.state.current_card_id
        elif self._selected_card_id not in card_ids:
            self._selected_card_id = snapshot.cards[0].id if snapshot.cards else None
        selected_card = next(
            (card for card in snapshot.cards if card.id == self._selected_card_id),
            None,
        )
        self._rendering = True
        try:
            if render_sidebar:
                self.card_sidebar.render(
                    snapshot,
                    self._selected_card_id,
                    draft_card_ids=(),
                )
            self.inspector.render(
                snapshot,
                self._selected_card_id,
                refine_source_size=self._refine_source_size(selected_card),
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
        self._render_utility_windows(skip=utility_source)

    def select_card(self, card_id: object) -> None:
        if self._is_running:
            return
        selected_card_id = card_id if isinstance(card_id, UUID) else None
        if (
            selected_card_id != self._selected_card_id
            and self.background_workflow is not None
            and self.background_workflow.busy
        ):
            self._cancel_background_generation()
        self._selected_card_id = selected_card_id
        if self.card_sidebar.selected_card_id != self._selected_card_id:
            self.card_sidebar.select_card(self._selected_card_id)
        card_ids = {card.id for card in self.controller.document.cards}
        self.render_document(render_sidebar=self._selected_card_id not in card_ids)

    def _show_hotspot_usage(
        self,
        card_id: object,
        revision_id: object,
        interaction_id: object,
    ) -> None:
        if (
            self._is_running
            or not isinstance(card_id, UUID)
            or not isinstance(revision_id, UUID)
            or not isinstance(interaction_id, UUID)
            or not self._commit_authoring_metadata()
        ):
            return
        self.select_card(card_id)
        card = next(
            (candidate for candidate in self.controller.document.cards if candidate.id == card_id),
            None,
        )
        if card is None:
            return
        if card.active_revision_id != revision_id:
            self._activate_revision(revision_id)
            card = next(
                (
                    candidate
                    for candidate in self.controller.document.cards
                    if candidate.id == card_id
                ),
                None,
            )
        if card is None or card.active_revision_id != revision_id:
            return
        self.inspector.show_interaction(interaction_id)
        self.card_canvas.select_interaction(interaction_id)

    def _commit_canvas_card_name(self, *, render_change: bool = True) -> bool:
        if self._rendering or self._selected_card_id is None or self._is_running:
            return True
        if self.controller.mutation_blocked:
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
        except (CommandError, DocumentMutationBlockedError, ValidationError) as error:
            self._card_name_commit_failed = True
            self._set_canvas_card_name_error(str(error))
            self.canvas_card_name.setText(card.name)
            return False
        self._card_name_commit_failed = False
        self._set_canvas_card_name_error("")
        if render_change:
            self.render_document(changed)
        return True

    def _card_name_edited(self) -> None:
        self._card_name_commit_failed = False
        self._set_canvas_card_name_error("")

    def _set_canvas_card_name_error(self, message: str) -> None:
        self.canvas_card_name_error.setText(message)
        self.canvas_card_name_error.setVisible(bool(message))

    def _commit_authoring_metadata(self) -> bool:
        if self.controller.mutation_blocked:
            self._show_pending_durability_error(PENDING_DURABILITY_MESSAGE)
            return False
        for window in (
            self.style_manager_window,
            self.sound_manager_window,
            self.key_manager_window,
        ):
            if window is not None and not window.commit_pending_edits(render_change=False):
                window.show()
                window.raise_()
                window.activateWindow()
                return False
        if self._selected_card_id is None:
            return True
        if not self.inspector.commit_card_metadata(render_change=False):
            return False
        return self._commit_canvas_card_name(render_change=False)

    def _flush_document(self, *, title: str) -> bool:
        if self.document_session is None:
            return True
        if self.document_session.flush():
            return True
        self._show_document_error(
            title,
            self.document_session.state.error or "The document could not be saved.",
        )
        return False

    def _prepare_authoring_lifecycle(
        self,
        *,
        save_error_title: str,
        flush_after_commit: bool = False,
    ) -> bool:
        durability_was_pending = self.controller.mutation_blocked
        if durability_was_pending and not self._flush_document(title=save_error_title):
            return False
        if not self._commit_authoring_metadata():
            return False
        if durability_was_pending or flush_after_commit:
            return self._flush_document(title=save_error_title)
        return True

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
        if workflow.busy:
            self._cancel_background_generation()
        self.notification_bar.clear_notification("background-error")
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
        self._generated_revision_change = None
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

    def _show_generated_revision_notification(self, change: object) -> None:
        if self._is_running or not isinstance(change, GeneratedRevisionChange):
            return
        self._generated_revision_change = change
        self._undo_notification_token = change.token
        self.notification_bar.show_notification(
            "undo",
            Notification(
                message=f"{change.message} on the current version",
                kind=NotificationKind.SUCCESS,
                primary_action=NotificationAction(
                    "create-generated-revision",
                    "Create New Version",
                ),
                secondary_action=NotificationAction("undo", "Undo"),
                dismiss_label="Keep",
                priority=3,
            ),
        )

    def _remember_edit_undo_instruction(self, change: object) -> None:
        if (
            isinstance(change, EditedRevisionChange)
            and self.controller.current_undo_token == change.token
        ):
            self._edit_undo_instructions[change.token] = change.instruction

    def _restore_edit_instruction_after_undo(self, token: UndoToken | None) -> None:
        if token is None:
            return
        instruction = self._edit_undo_instructions.pop(token, None)
        if instruction is None:
            return
        if self.inspector.edit_instruction_edit.toPlainText():
            return
        self.inspector.set_edit_instruction(instruction)
        self._update_generation_actions()

    def _undo_notification(self) -> None:
        if self._is_running:
            return
        token = self._undo_notification_token
        self._clear_undo_notification()
        if self.background_workflow is not None and self.background_workflow.busy:
            self._cancel_background_generation()
        if token is not None and self.controller.undo_if_current(token):
            self._restore_card_selection(token, undoing=True)
            self.render_document()
            self._restore_edit_instruction_after_undo(token)

    def _create_generated_revision(self) -> None:
        if self._is_running:
            return
        change = self._generated_revision_change
        if change is None or self.controller.current_undo_token != change.token:
            self._clear_undo_notification()
            return
        card = next(
            (
                candidate
                for candidate in self.controller.document.cards
                if candidate.id == change.card_id
            ),
            None,
        )
        revision = (
            next(
                (candidate for candidate in card.revisions if candidate.id == change.revision_id),
                None,
            )
            if card is not None
            else None
        )
        if revision is None:
            self._clear_undo_notification()
            return
        if self.background_workflow is not None and self.background_workflow.busy:
            self._cancel_background_generation()
        self._clear_undo_notification()
        command = CreateGeneratedRevisionCommand(
            card_id=change.card_id,
            revision_id=change.revision_id,
            previous_revision=change.previous_revision,
        )
        try:
            changed = self.controller.execute(command)
        except (CommandError, DocumentMutationBlockedError, ValidationError) as error:
            self._show_error(
                "revision-error",
                "Could not create a new version",
                detail=str(error),
            )
            return
        self._selected_card_id = change.card_id
        self.render_document(changed)
        token = self.controller.current_undo_token
        if token is not None:
            self._show_undo_notification("New version created", token)

    def _clear_undo_notification(self) -> None:
        self._undo_notification_token = None
        self._generated_revision_change = None
        self.notification_bar.clear_notification("undo")

    def _notification_action_requested(self, action_id: str) -> None:
        if action_id == "undo" and not self._is_running:
            self._undo_notification()
        elif action_id == "create-generated-revision" and not self._is_running:
            self._create_generated_revision()
        elif action_id == "open-settings" and not self._is_running:
            self.open_model_settings()
        elif action_id == "check-services" and not self._is_running:
            self.run_availability_checks()

    def _notification_dismissed(self, key: str) -> None:
        if key == "undo":
            self._undo_notification_token = None
            self._generated_revision_change = None
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

    def _show_pending_durability_error(self, detail: str) -> None:
        self._show_error(
            "document-error",
            "Save required before editing",
            detail=detail,
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
        if not self._prepare_authoring_lifecycle(save_error_title="Could Not Save Current Stack"):
            return
        dialog = NewStackDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        stack = dialog.stack()
        suggested_path = self.project_directory / f"{stack.name}.hotcards"
        selected_path, _filter = QFileDialog.getSaveFileName(
            self,
            "Create HotCards Stack",
            str(suggested_path),
            "HotCards Stack (*.hotcards)",
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
        if not self._prepare_authoring_lifecycle(save_error_title="Could Not Save Current Stack"):
            return
        selected_path = QFileDialog.getExistingDirectory(
            self,
            "Open HotCards Stack",
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
        return self._prepare_authoring_lifecycle(
            save_error_title="Could Not Save Stack",
            flush_after_commit=True,
        )

    def save_as(self) -> None:
        """Clone the current bound bundle and rebind future autosaves."""
        if self.document_session is None or self.document_session.store is None:
            return
        if not self._prepare_authoring_lifecycle(save_error_title="Could Not Save Current Stack"):
            return
        selected_path, _filter = QFileDialog.getSaveFileName(
            self,
            "Save HotCards Stack As",
            str(self.project_directory / f"{self.controller.document.name}.hotcards"),
            "HotCards Stack (*.hotcards)",
        )
        if not selected_path:
            return
        try:
            self.document_session.save_as(self._bundle_path(selected_path))
        except DocumentSessionError as error:
            self._show_document_error("Could Not Save Stack As", str(error))
            return
        self._cancel_background_generation()
        self._clear_undo_notification()
        self._update_document_actions()

    def undo(self) -> None:
        self._clear_undo_notification()
        if self.background_workflow is not None and self.background_workflow.busy:
            self._cancel_background_generation()
        token = self.controller.current_undo_token
        try:
            changed = self.controller.undo()
        except DocumentMutationBlockedError as error:
            self._show_pending_durability_error(str(error))
            return
        if changed:
            self._restore_card_selection(token, undoing=True)
            self.render_document()
            self._restore_edit_instruction_after_undo(token)

    def redo(self) -> None:
        self._clear_undo_notification()
        if self.background_workflow is not None and self.background_workflow.busy:
            self._cancel_background_generation()
        token = self.controller.current_redo_token
        try:
            changed = self.controller.redo()
        except DocumentMutationBlockedError as error:
            self._show_pending_durability_error(str(error))
            return
        if changed:
            self._restore_card_selection(token, undoing=False)
            self.render_document()

    def _restore_card_selection(
        self,
        token: UndoToken | None,
        *,
        undoing: bool,
    ) -> None:
        if token is None or token not in self._card_selection_history:
            return
        before, after = self._card_selection_history[token]
        self._selected_card_id = before if undoing else after

    def _authoring_inputs_changed(self) -> None:
        if self.background_workflow is not None and self.background_workflow.busy:
            self._cancel_background_generation()
        self._update_generation_actions()

    def _primary_empty_action(self) -> None:
        if self._is_running:
            return
        if self.document_session is not None and self.document_session.store is None:
            self.new_stack()
        else:
            self.card_sidebar.add_card()

    def _duplicate_card(self, requested_card_id: object = None) -> None:
        if self._is_running or self.card_duplication_workflow is None:
            return
        card_id = (
            requested_card_id if isinstance(requested_card_id, UUID) else self._selected_card_id
        )
        if card_id is None or card_id != self._selected_card_id:
            return
        if not self._commit_authoring_metadata():
            return
        if self.background_workflow is not None and self.background_workflow.busy:
            self._cancel_background_generation()
        self.notification_bar.clear_notification("card-error")
        previous_selection = self._selected_card_id
        try:
            change = self.card_duplication_workflow.duplicate(card_id)
        except CardDuplicationError as error:
            self.render_document()
            if self.controller.mutation_blocked:
                self._show_pending_durability_error(f"{PENDING_DURABILITY_MESSAGE}\n\n{error}")
            else:
                self._show_error(
                    "card-error",
                    "Could not duplicate card",
                    detail=str(error),
                )
            return
        self._selected_card_id = change.duplicate_card_id
        self._card_selection_history[change.token] = (
            previous_selection,
            change.duplicate_card_id,
        )
        self.render_document(change.document)
        self._show_undo_notification("Card duplicated", change.token)

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
            if interaction.action is not None
            and isinstance(interaction.action.target, ResolvedCardReference)
            and interaction.action.target.target_card_id == card.id
        )
        was_generating = (
            self.background_workflow is not None
            and self.background_workflow.is_generating_for(card.id)
        )
        previous_token = self.controller.current_undo_token
        try:
            self.card_sidebar.delete_card(card.id)
        except (CommandError, DocumentMutationBlockedError) as error:
            self._show_error(
                "card-error",
                "Could not delete card",
                detail=str(error),
            )
            return
        if was_generating:
            self._cancel_background_generation()
        self.notification_bar.clear_notification("card-error")
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
        if self.background_workflow is not None:
            self.background_workflow.cancel()

    def _cancel_generation_activity(self) -> None:
        if self.background_workflow is not None and self.background_workflow.busy:
            self._cancel_background_generation()

    def _document_replaced(self, _document: object) -> None:
        self._cancel_background_generation()
        if self.sound_workflow is not None:
            self.sound_workflow.cancel()
        self.sound_player.stop()
        self._close_utility_windows(commit_pending=False)
        self._clear_undo_notification()
        self._edit_undo_instructions.clear()
        self._card_selection_history.clear()
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
        durability_resolved = self._last_session_mutation_blocked and not state.mutation_blocked
        self._last_session_mutation_blocked = state.mutation_blocked
        if (
            state.mutation_blocked
            and self.background_workflow is not None
            and self.background_workflow.busy
        ):
            self._cancel_background_generation()
        if state.mutation_blocked and self.sound_workflow is not None:
            self.sound_workflow.cancel()
        if state.error is None:
            self.notification_bar.clear_notification("document-error")
        bound = state.bundle_path is not None
        mutation_allowed = (self.document_session is None or bound) and not state.mutation_blocked
        for window in (
            self.style_manager_window,
            self.key_manager_window,
        ):
            if window is not None:
                window.set_mutation_allowed(mutation_allowed)
        self.card_sidebar.set_document_editable(self.document_session is None or mutation_allowed)
        if self.document_session is not None and not bound:
            self.create_first_card_button.setText("Create New Stack")
        elif state.mutation_blocked:
            self.create_first_card_button.setText("Create Your First Card")
            detail = PENDING_DURABILITY_MESSAGE
            if state.error is not None:
                detail = f"{detail}\n\n{state.error}"
            self._show_pending_durability_error(detail)
        elif state.error is not None:
            self.create_first_card_button.setText("Create Your First Card")
            self._show_error(
                "document-error",
                "Stack could not be saved",
                detail=state.error,
            )
        else:
            self.create_first_card_button.setText("Create Your First Card")
        self._update_document_actions()
        self.styles_button.setEnabled(mutation_allowed and not self._is_running)
        self.sounds_button.setEnabled(
            mutation_allowed and not self._is_running and self.sound_workflow is not None
        )
        self.keys_button.setEnabled(mutation_allowed and not self._is_running)
        self._update_window_title()
        self._update_generation_actions()
        if durability_resolved:
            self.render_document()

    def _background_activity_changed(self, _active: bool) -> None:
        if self._image_operation_active() and self.card_canvas.drawing:
            self.card_canvas.cancel_drawing()
            self._show_info(
                "hotspot-draft-cancelled",
                "Unfinished hotspot drawing cancelled before image processing",
            )
        card = next(
            (card for card in self.controller.document.cards if card.id == self._selected_card_id),
            None,
        )
        if card is not None and not self._is_running:
            self._render_card_canvas(card)
        self._update_generation_actions()

    def _image_operation_active(self) -> bool:
        workflow = self.background_workflow
        return bool(
            workflow is not None
            and (workflow.busy or bool(getattr(workflow, "invocation_active", False)))
        )

    def _update_document_actions(self) -> None:
        bound = self.document_session is None or self.document_session.store is not None
        mutation_allowed = bound and not self.controller.mutation_blocked
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
            mutation_allowed and not self._is_running and self.controller.can_undo
        )
        self.redo_action.setEnabled(
            mutation_allowed and not self._is_running and self.controller.can_redo
        )
        self.duplicate_card_action.setEnabled(
            mutation_allowed
            and not self._is_running
            and self._selected_card_id is not None
            and self.card_duplication_workflow is not None
        )
        self.models_action.setEnabled(not self._is_running)
        self.mode_button.setEnabled(bound)
        self.overlay_selector.setEnabled(mutation_allowed and self._is_running)
        self.card_sidebar.set_document_editable(mutation_allowed and not self._is_running)
        authoring_enabled = mutation_allowed and not self._is_running
        self.styles_button.setEnabled(authoring_enabled)
        self.sounds_button.setEnabled(authoring_enabled and self.sound_workflow is not None)
        self.keys_button.setEnabled(authoring_enabled)
        self.inspector.setEnabled(authoring_enabled)
        self.canvas_card_name.setReadOnly(not authoring_enabled)
        self.revision_combo.setEnabled(authoring_enabled)
        self.add_revision_button.setEnabled(
            authoring_enabled and self._selected_card_id is not None
        )
        selected_card = next(
            (card for card in self.controller.document.cards if card.id == self._selected_card_id),
            None,
        )
        self.delete_revision_button.setEnabled(
            authoring_enabled and selected_card is not None and len(selected_card.revisions) > 1
        )
        self.create_first_card_button.setEnabled(mutation_allowed)

    def _update_window_title(self) -> None:
        dirty = self.document_session is not None and self.document_session.state.dirty
        suffix = " *" if dirty else ""
        self.setWindowTitle(f"HotCards — {self.controller.document.name}{suffix}")

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
                self.workers.check_mflux(check, emit_diagnostic=False)
                if adapter is AdapterKind.MFLUX
                else self.workers.check_stable_audio(check, emit_diagnostic=False)
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
        self.render_document()
        self._update_generation_actions()

    def _refine_background(
        self,
        transformation: object,
        output_size: object,
    ) -> None:
        workflow = self.background_workflow
        card_id = self._selected_card_id
        if (
            workflow is None
            or card_id is None
            or not isinstance(transformation, RefineTransformation)
            or not isinstance(output_size, (CurrentSourceSize, PresetOutputSize))
        ):
            return
        self.notification_bar.clear_notification("background-error")
        self.notification_bar.clear_notification("background-warning")
        self.notification_bar.clear_notification("background-cancelled")
        if not self._commit_authoring_metadata():
            return
        try:
            workflow.refine(
                card_id,
                transformation=transformation,
                output_size=output_size,
            )
        except BackgroundWorkflowError as error:
            self._show_error(
                "background-error",
                "Could not reinterpret image",
                detail=str(error),
            )
        self.render_document()
        self._update_generation_actions()

    def _edit_background(
        self,
        instruction: object,
        output_size: object,
    ) -> None:
        workflow = self.background_workflow
        card_id = self._selected_card_id
        if (
            workflow is None
            or card_id is None
            or not isinstance(instruction, str)
            or not isinstance(
                output_size,
                (CurrentSourceSize, PresetOutputSize),
            )
        ):
            return
        self.notification_bar.clear_notification("background-error")
        self.notification_bar.clear_notification("background-warning")
        self.notification_bar.clear_notification("background-cancelled")
        if not self._commit_authoring_metadata():
            return
        try:
            workflow.edit(
                card_id,
                instruction=instruction,
                output_size=output_size,
            )
        except (BackgroundWorkflowError, ValidationError) as error:
            self.inspector.set_edit_error(str(error))
            self._show_error(
                "background-error",
                "Could not edit image",
                detail=str(error),
            )
        self.render_document()
        self._update_generation_actions()

    def _activate_revision(self, revision_id: object) -> None:
        if (
            self.background_workflow is None
            or self._selected_card_id is None
            or not isinstance(revision_id, UUID)
        ):
            return
        selected_card = next(
            (card for card in self.controller.document.cards if card.id == self._selected_card_id),
            None,
        )
        if (
            selected_card is not None
            and revision_id != selected_card.active_revision_id
            and self.background_workflow.busy
        ):
            self._cancel_background_generation()
        self.notification_bar.clear_notification("background-error")
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
        if message in {
            "Generation cancelled",
            "Reinterpret cancelled",
            "Edit cancelled",
        }:
            self._show_info("background-cancelled", message)
        else:
            self.notification_bar.clear_notification("background-cancelled")
        self._background_progress_message = message
        self._background_step_progress = None
        self._update_generation_progress()
        self._update_generation_actions()

    def _background_generation_progress_changed(
        self,
        completed_steps: int,
        total_steps: int,
    ) -> None:
        if completed_steps == 0 and total_steps == 0:
            self._background_step_progress = None
            self._update_generation_progress()
            return
        if total_steps <= 0 or not 0 <= completed_steps <= total_steps:
            return
        self._background_step_progress = (completed_steps, total_steps)
        self._update_generation_progress()

    def _update_generation_progress(self) -> None:
        background_busy = (
            self.background_workflow.busy if self.background_workflow is not None else False
        )
        if not background_busy:
            self.generation_progress_container.hide()
            self.generation_step_label.clear()
            return
        self.generation_step_label.setText(
            self._background_progress_message or "Generating image..."
        )
        if background_busy and self._background_step_progress is not None:
            completed_steps, total_steps = self._background_step_progress
            self.generation_progress_bar.setRange(0, total_steps)
            self.generation_progress_bar.setValue(completed_steps)
        else:
            self.generation_progress_bar.setRange(0, 0)
        self.generation_progress_container.show()

    def _background_failed(self, failure: object) -> None:
        if self._background_progress_message == "Image reinterpretation failed":
            title = "Image reinterpretation failed"
        elif self._background_progress_message == "Image editing failed":
            title = "Image editing failed"
        else:
            title = "Image generation failed"
        detail = failure.message if isinstance(failure, WorkerFailure) else str(failure)
        if title == "Image editing failed":
            self.inspector.set_edit_error(detail)
        message = f"{title}: {detail}" if detail else title
        if isinstance(failure, WorkerFailure):
            self._show_error(
                "background-error",
                message,
                detail=detail,
            )
        else:
            self._show_error(
                "background-error",
                message,
                detail=detail,
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
        from hotcards.application.workers import WorkerFailure

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
        mutation_allowed = bound and not self.controller.mutation_blocked
        workflow_available = self.background_workflow is not None
        has_card = (
            workflow_available
            and mutation_allowed
            and self._selected_card_id is not None
            and not self._is_running
        )
        mflux_available = self._availability[AdapterKind.MFLUX] is True
        has_description_input = self.inspector.has_description_input()
        selected_card = next(
            (card for card in self.controller.document.cards if card.id == self._selected_card_id),
            None,
        )
        workflow_busy = self._image_operation_active()
        active_operation = (
            getattr(self.background_workflow, "active_operation", None)
            if self.background_workflow is not None
            else None
        )
        active_revision = selected_card.active_revision if selected_card is not None else None
        has_image = active_revision is not None and active_revision.background is not None
        generate_reason = "Ready to generate"
        if self.controller.mutation_blocked:
            generate_reason = PENDING_DURABILITY_MESSAGE
        elif not has_card:
            generate_reason = "Select a card in a saved stack"
        elif workflow_busy:
            generate_reason = (
                "Background generation is running for this card"
                if active_operation == "generate"
                and self.background_workflow is not None
                and self._selected_card_id is not None
                and self.background_workflow.is_generating_for(self._selected_card_id)
                else ("An image operation is running; MFLUX runs one job at a time")
            )
        elif not has_description_input:
            generate_reason = "Enter a Description before generating"
        elif not mflux_available:
            generate_reason = self._action_diagnostic(AdapterKind.MFLUX)
        self.inspector.set_background_capabilities(
            can_generate=(has_card and has_description_input and mflux_available),
            generate_reason=generate_reason,
            has_image=has_image,
            busy=workflow_busy,
            generating=workflow_busy and active_operation == "generate",
        )
        refine_output_size = self.inspector.refine_resolution_combo.currentData()
        has_refine_output_size = isinstance(
            refine_output_size,
            (CurrentSourceSize, PresetOutputSize),
        )
        refine_reason = "Ready to reinterpret"
        if self.controller.mutation_blocked:
            refine_reason = PENDING_DURABILITY_MESSAGE
        elif not has_card:
            refine_reason = "Select a card in a saved stack"
        elif workflow_busy:
            refine_reason = (
                "Reinterpret is running for this card"
                if self.background_workflow is not None
                and self._selected_card_id is not None
                and getattr(
                    self.background_workflow,
                    "is_refining_for",
                    lambda _card_id: False,
                )(self._selected_card_id)
                else ("An image operation is running; MFLUX runs one job at a time")
            )
        elif not has_description_input:
            refine_reason = "Enter a Description before reinterpreting"
        elif not has_image:
            refine_reason = "Generate an image before reinterpreting"
        elif not has_refine_output_size:
            refine_reason = self.inspector.refine_error.text() or "Select a Reinterpret resolution"
        elif not mflux_available:
            refine_reason = self._action_diagnostic(AdapterKind.MFLUX)
        self.inspector.set_refine_capabilities(
            can_refine=(
                has_card
                and has_description_input
                and has_image
                and has_refine_output_size
                and mflux_available
            ),
            refine_reason=refine_reason,
            busy=workflow_busy,
            refining=workflow_busy and active_operation == "refine",
        )
        has_edit_instruction = self.inspector.has_edit_instruction_input()
        has_edit_output_size = self.inspector.selected_edit_output_size() is not None
        edit_reason = "Ready to edit"
        if self.controller.mutation_blocked:
            edit_reason = PENDING_DURABILITY_MESSAGE
        elif not has_card:
            edit_reason = "Select a card in a saved stack"
        elif workflow_busy:
            edit_reason = (
                "Edit is running for this card"
                if self.background_workflow is not None
                and self._selected_card_id is not None
                and getattr(
                    self.background_workflow,
                    "is_editing_for",
                    lambda _card_id: False,
                )(self._selected_card_id)
                else "An image operation is running; MFLUX runs one job at a time"
            )
        elif not has_image:
            edit_reason = "Generate an image before editing"
        elif not has_edit_instruction:
            edit_reason = "Enter an Edit Instruction"
        elif not has_edit_output_size:
            edit_reason = (
                self.inspector.edit_output_error.text()
                or "Select the current size or a higher Edit output tier"
            )
        elif not mflux_available:
            edit_reason = self._action_diagnostic(AdapterKind.MFLUX)
        self.inspector.set_edit_capabilities(
            can_edit=(
                has_card
                and has_image
                and has_edit_instruction
                and has_edit_output_size
                and mflux_available
            ),
            edit_reason=edit_reason,
            busy=workflow_busy,
            editing=workflow_busy and active_operation == "edit",
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
            summary = "MFLUX unavailable"
        else:
            summary = "Local AI services ready"
        self._service_status_detail = "\n".join(
            self._diagnostic_messages.get(
                adapter,
                f"{adapter.value} availability check pending",
            )
            for adapter in AdapterKind
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

    def _refine_source_size(
        self,
        card: Card | None,
    ) -> tuple[int, int] | None:
        if (
            card is None
            or card.active_revision.background is None
            or self.document_session is None
            or self.document_session.store is None
        ):
            return None
        try:
            background = card.active_revision.background
            return self.document_session.store.image_asset_dimensions(
                background.image_path,
                card_id=card.id,
                asset_id=background.id,
            )
        except StackStoreError:
            return None

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
                self._run_session.active_hotspot_set(self.controller.document),
                self.controller.document.run_overlay_mode,
            )
            return
        self.card_canvas.set_hotspots(
            revision.hotspot_set,
            self.inspector.selected_interaction_id,
            editable=(
                self.inspector.hotspots_active
                and not self.controller.mutation_blocked
                and not self._image_operation_active()
            ),
            context_id=revision.id,
        )

    def _inspector_tab_changed(self, _index: int) -> None:
        if not self._rendering and not self._is_running:
            self.render_document()

    def _resolve_revision_image_path(self, image_path: str) -> Path | None:
        if self.document_session is None or self.document_session.store is None:
            return None
        try:
            return self.document_session.store.asset_path(image_path)
        except StackStoreError:
            return None

    def _resolve_sound_asset_path(self, audio_path: str) -> Path:
        if self.document_session is None or self.document_session.store is None:
            raise StackStoreError("save the stack before playing a Sound")
        return self.document_session.store.asset_path(audio_path)

    def _reference_image_path(self, image_path: str) -> Path:
        if self.document_session is None or self.document_session.store is None:
            raise BackgroundWorkflowError("save the stack before generating with a Reference image")
        return self.document_session.store.asset_path(image_path)

    def _create_hotspot_polygon(
        self,
        polygon: object,
    ) -> None:
        context = self._active_hotspot_context()
        if context is None or not isinstance(polygon, Polygon):
            return
        card_id, revision_id, _hotspot_set = context
        interaction = Interaction(polygons=(polygon,))
        self._execute_hotspot_canvas_command(
            AddInteractionCommand(
                card_id=card_id,
                revision_id=revision_id,
                interaction=interaction,
            ),
            interaction.id,
            selected_polygon_index=0,
            undo_message="Hotspot created",
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
        except (CommandError, DocumentMutationBlockedError, ValidationError) as error:
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
        try:
            changed = self.controller.execute(SetRunOverlayModeCommand(mode=overlay_mode))
        except DocumentMutationBlockedError as error:
            self._show_pending_durability_error(str(error))
            return
        self.render_document(changed)

    def _toggle_mode(self) -> None:
        if self._rendering:
            return
        should_run = not self._is_running
        if should_run == self._is_running:
            return
        if should_run:
            if not self._commit_authoring_metadata():
                return
            self.card_canvas.cancel_drawing()
            self._cancel_ai_activity_for_run()
            self._generated_revision_change = None
            self._undo_notification_token = None
            self.notification_bar.clear_all()
            self._is_running = True
            state = self._run_session.start(
                self.controller.document,
                self._selected_card_id,
            )
            self._selected_card_id = state.current_card_id
        else:
            self.sound_player.stop()
            self._is_running = False
            self._run_session.clear()
            state = None
        self._apply_mode_chrome()
        self._set_run_warning("")
        self._set_run_notice("")
        self.render_document()
        if state is not None and state.warning is not None:
            self._set_run_warning(state.warning)
        if not should_run:
            self.run_availability_checks()

    def _run_interaction_activated(self, interaction_id: object) -> None:
        if not self._is_running or not isinstance(interaction_id, UUID):
            return
        document = self.controller.document
        sound_id = self._run_session.activation_sound_id(document, interaction_id)
        if sound_id is not None or self._run_session.activation_has_navigation(
            document,
            interaction_id,
        ):
            self.sound_player.stop()
        state = self._run_session.activate(
            document,
            interaction_id,
        )
        self._apply_run_state(state)
        if sound_id is not None:
            self._play_run_sound(sound_id)

    def _run_back(self) -> None:
        if self._is_running:
            self.sound_player.stop()
            self._apply_run_state(self._run_session.back())

    def _run_restart(self) -> None:
        if self._is_running:
            self.sound_player.stop()
            self._apply_run_state(self._run_session.restart())

    def _play_run_sound(self, sound_id: UUID) -> None:
        try:
            sound = self.controller.document.sound_by_id(sound_id)
        except StopIteration:
            self._set_run_warning("The selected Sound is no longer available.")
            return
        if sound.generated is None:
            self._set_run_warning(f'"{sound.name}" has not been generated yet.')
            return
        try:
            path = self._resolve_sound_asset_path(sound.generated.audio_path)
        except StackStoreError as error:
            self._set_run_warning(str(error))
            return
        self.sound_player.play(path)

    def _apply_run_state(self, state: RunSessionState) -> None:
        self._selected_card_id = state.current_card_id
        self._set_run_warning("")
        self._set_run_notice("")
        self.render_document()
        if state.warning is not None:
            self._set_run_warning(state.warning)
        elif state.notice is not None:
            self._set_run_notice(state.notice)

    def _apply_mode_chrome(self) -> None:
        authoring = not self._is_running
        authoring_enabled = authoring and not self.controller.mutation_blocked
        self.mode_button.setText("Run" if authoring else "Author")
        self.mode_button.setToolTip("Switch to Run mode" if authoring else "Switch to Author mode")
        self.canvas_card_name.setReadOnly(not authoring_enabled)
        self.canvas_card_name.setVisible(authoring)
        self.canvas_card_name_error.setVisible(
            authoring and bool(self.canvas_card_name_error.text())
        )
        self.revision_label.setVisible(authoring)
        self.revision_combo.setVisible(authoring)
        self.revision_combo.setEnabled(authoring_enabled)
        self.add_revision_button.setVisible(authoring)
        self.delete_revision_button.setVisible(authoring)
        self.card_sidebar.setVisible(authoring)
        self.inspector.setVisible(authoring)
        self.create_first_card_button.setVisible(authoring)
        self.empty_canvas_title.setText(
            "Create your first card" if authoring else "No cards to run"
        )
        self.empty_canvas_description.setText(
            "Cards are the scenes readers visit. Start with one, then add its background and links."
            if authoring
            else "Switch to Author mode to create the first card."
        )
        self.run_controls_separator.setVisible(self._is_running)
        self.run_overlay_separator.setVisible(self._is_running)
        self.overlay_label_action.setVisible(self._is_running)
        self.overlay_selector_action.setVisible(self._is_running)
        self.overlay_label.setVisible(self._is_running)
        self.overlay_selector.setVisible(self._is_running)
        for action in self.player_navigation_actions:
            action.setVisible(self._is_running)
        self.back_button_action.setVisible(self._is_running)
        self.restart_button_action.setVisible(self._is_running)
        self.styles_button_action.setVisible(authoring)
        self.sounds_button_action.setVisible(authoring)
        self.keys_button_action.setVisible(authoring)
        self.styles_button.setVisible(authoring)
        self.sounds_button.setVisible(authoring)
        self.keys_button.setVisible(authoring)
        self.styles_button.setEnabled(authoring_enabled)
        self.sounds_button.setEnabled(authoring_enabled and self.sound_workflow is not None)
        self.keys_button.setEnabled(authoring_enabled)
        if self._is_running:
            if self.style_manager_window is not None:
                self.style_manager_window.hide()
            if self.sound_manager_window is not None:
                self.sound_manager_window.hide()
            if self.key_manager_window is not None:
                self.key_manager_window.hide()

    def _update_run_actions(self) -> None:
        state = self._run_session.state
        can_go_back = self._is_running and bool(state.history)
        can_restart = self._is_running and state.current_card_id is not None
        self.back_action.setEnabled(can_go_back)
        self.restart_action.setEnabled(can_restart)
        self.back_button.setEnabled(can_go_back)
        self.restart_button.setEnabled(can_restart)

    def _set_run_warning(self, message: str) -> None:
        if message:
            self._show_warning("run-warning", message)
        else:
            self.notification_bar.clear_notification("run-warning")

    def _set_run_notice(self, message: str) -> None:
        if message:
            self._show_info("run-notice", message)
        else:
            self.notification_bar.clear_notification("run-notice")

    def _cancel_ai_activity_for_run(self) -> None:
        self._cancel_diagnostics()
        if self.background_workflow is not None:
            self.background_workflow.cancel()
        if self.sound_workflow is not None:
            self.sound_workflow.cancel()
        self.sound_player.stop()

    def _cancel_diagnostics(self) -> None:
        self._diagnostic_generation += 1
        for operation in tuple(self._diagnostic_operations):
            operation.cancel()

    def _restart_availability_checks(self) -> None:
        self._cancel_diagnostics()
        self.run_availability_checks()

    def open_model_settings(self) -> None:
        previous = load_machine_settings(self.settings)
        dialog = self._settings_dialog_factory(self.settings, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        current = load_machine_settings(self.settings)
        if current.mflux_model != previous.mflux_model:
            self._cancel_background_generation()
            self._restart_availability_checks()
        if (
            current.stable_audio_model != previous.stable_audio_model
            and self.sound_workflow is not None
        ):
            self.sound_workflow.cancel()

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
        if (
            self.document_session is not None
            and self.controller.mutation_blocked
            and not self._flush_for_close()
        ):
            event.ignore()
            return
        if not self._commit_authoring_metadata():
            event.ignore()
            return
        if self.document_session is not None and not self._flush_for_close():
            event.ignore()
            return
        if self.document_session is not None and not self.document_session.close_history():
            self._show_document_error(
                "Could Not Clean Up Stack",
                self.document_session.state.error
                or "Duplicate-owned assets could not be cleaned up.",
            )
        if self.background_workflow is not None:
            self.background_workflow.close()
        if self.sound_workflow is not None:
            self.sound_workflow.cancel()
        self.sound_player.stop()
        self._close_utility_windows()
        if self._owns_workers:
            self.workers.shutdown(wait_milliseconds=100)
        super().closeEvent(event)

    def _flush_for_close(self) -> bool:
        if self.document_session is None or self.document_session.flush():
            return True
        if not self._ask_retry_failed_close_save(
            (self.document_session.state.error or "The stack could not be saved.")
            + "\n\nRetry saving before closing?",
        ):
            return False
        return self.document_session.flush()


__all__ = [
    "AvailabilityChecks",
    "AvailabilityChecksFactory",
    "MainWindow",
    "SettingsDialogFactory",
]
