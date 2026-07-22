"""Restrained three-pane HyperGen application shell."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any
from uuid import UUID

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
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

from hypergen.application.commands import SetRunOverlayModeCommand
from hypergen.application.document_controller import DocumentController
from hypergen.application.document_session import (
    DocumentSession,
    DocumentSessionError,
    DocumentSessionState,
)
from hypergen.application.workers import (
    AdapterKind,
    AdapterWorkers,
    AvailabilityDiagnostic,
    WorkerOperation,
)
from hypergen.domain.models import RunOverlayMode, Stack
from hypergen.ui.card_sidebar import CardSidebar
from hypergen.ui.inspector import Inspector
from hypergen.ui.new_stack_dialog import NewStackDialog
from hypergen.ui.settings_dialog import SettingsDialog, SettingsStore

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
        start_diagnostics: bool = True,
        owns_workers: bool = False,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.workers = workers
        self.settings = settings if settings is not None else QSettings()
        self.document_session = document_session
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
        toolbar.addWidget(self.mode_selector)
        toolbar.addSeparator()

        self.generate_background_action = QAction("Generate Background", self)
        self.generate_background_action.setObjectName("generateBackgroundAction")
        self.generate_hotspots_action = QAction("Generate Hotspots", self)
        self.generate_hotspots_action.setObjectName("generateHotspotsAction")
        toolbar.addAction(self.generate_background_action)
        toolbar.addAction(self.generate_hotspots_action)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("Overlay"))
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

        self.first_card_action = QAction("First", self)
        self.previous_card_action = QAction("Previous", self)
        self.next_card_action = QAction("Next", self)
        self.last_card_action = QAction("Last", self)
        self.player_navigation_actions = (
            self.first_card_action,
            self.previous_card_action,
            self.next_card_action,
            self.last_card_action,
        )
        for action in self.player_navigation_actions:
            action.setEnabled(False)
            action.setToolTip("Available in Run mode")
            toolbar.addAction(action)

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
        empty_title = QLabel("Create your first card")
        empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_title.setStyleSheet("font-size: 20px; font-weight: 600;")
        empty_layout.addWidget(empty_title)
        empty_description = QLabel(
            "Cards are the scenes readers visit. Start with one, then add its background and links."
        )
        empty_description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_description.setWordWrap(True)
        empty_layout.addWidget(empty_description)
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
        card_canvas.setObjectName("cardCanvas")
        canvas_layout = QVBoxLayout(card_canvas)
        canvas_layout.addStretch(1)
        canvas_title = QLabel("Card Canvas")
        canvas_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        canvas_title.setStyleSheet("font-size: 18px; font-weight: 600;")
        canvas_layout.addWidget(canvas_title)
        self.canvas_card_name = QLabel()
        self.canvas_card_name.setObjectName("canvasCardName")
        self.canvas_card_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        canvas_layout.addWidget(self.canvas_card_name)
        canvas_hint = QLabel("Use the generation actions above to add content to this card.")
        canvas_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        canvas_hint.setWordWrap(True)
        canvas_layout.addWidget(canvas_hint)
        canvas_layout.addStretch(1)
        self.canvas_pages.addWidget(card_canvas)

        self.inspector = Inspector(self.controller)
        self.inspector.document_changed.connect(self.render_document)

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
        self.review_settings_button = QPushButton("Review Settings…")
        self.review_settings_button.setObjectName("reviewSettingsButton")
        self.review_settings_button.setVisible(False)
        self.review_settings_button.clicked.connect(self.open_advanced_settings)
        self.statusBar().addPermanentWidget(self.review_settings_button)
        self.create_first_card_button.clicked.connect(self._primary_empty_action)

        self.document_status_label = QLabel()
        self.document_status_label.setObjectName("documentStatusLabel")
        self.statusBar().addWidget(self.document_status_label, 1)

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
        card_ids = {card.id for card in snapshot.cards}
        if self._selected_card_id not in card_ids:
            self._selected_card_id = snapshot.cards[0].id if snapshot.cards else None
        self._rendering = True
        try:
            self.card_sidebar.render(snapshot, self._selected_card_id)
            self.inspector.render(snapshot, self._selected_card_id)
            selected_card = next(
                (card for card in snapshot.cards if card.id == self._selected_card_id),
                None,
            )
            if selected_card is None:
                self.canvas_pages.setCurrentIndex(0)
                self.canvas_card_name.clear()
            else:
                self.canvas_pages.setCurrentIndex(1)
                self.canvas_card_name.setText(selected_card.name)
            overlay_index = self.overlay_selector.findData(snapshot.run_overlay_mode)
            self.overlay_selector.setCurrentIndex(overlay_index)
        finally:
            self._rendering = False
        self._update_document_actions()
        self._update_window_title()
        self._update_generation_actions()

    def select_card(self, card_id: object) -> None:
        self._selected_card_id = card_id if isinstance(card_id, UUID) else None
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
        try:
            self.document_session.create(stack, self._bundle_path(selected_path))
        except DocumentSessionError as error:
            self._show_document_error("Could Not Create Stack", str(error))

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
        try:
            self.document_session.open(Path(selected_path))
        except DocumentSessionError as error:
            self._show_document_error("Could Not Open Stack", str(error))

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
            self.card_sidebar.delete_card(card.id)

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
        self.undo_action.setEnabled(bound and self.controller.can_undo)
        self.redo_action.setEnabled(bound and self.controller.can_redo)
        self.mode_selector.setEnabled(bound)
        self.overlay_selector.setEnabled(bound)

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
        has_card = bound and self._selected_card_id is not None
        mflux_available = self._availability[AdapterKind.MFLUX] is True
        ollama_available = self._availability[AdapterKind.OLLAMA] is True
        self.generate_background_action.setEnabled(
            has_card and mflux_available and ollama_available
        )
        self.generate_hotspots_action.setEnabled(has_card and ollama_available)
        self.generate_background_action.setToolTip(
            " · ".join(
                self._action_diagnostic(adapter)
                for adapter in (AdapterKind.OLLAMA, AdapterKind.MFLUX)
            )
        )
        self.generate_hotspots_action.setToolTip(self._action_diagnostic(AdapterKind.OLLAMA))
        pending = [
            adapter for adapter, available in self._availability.items() if available is None
        ]
        unavailable = [
            adapter for adapter, available in self._availability.items() if available is False
        ]
        if pending:
            summary = "Checking local AI services…"
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
        self.review_settings_button.setVisible(bool(unavailable))

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

    def open_advanced_settings(self) -> None:
        dialog = self._settings_dialog_factory(self.settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.run_availability_checks()

    def closeEvent(self, event: QCloseEvent) -> None:
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
        if self._owns_workers:
            self.workers.shutdown(wait_milliseconds=100)
        super().closeEvent(event)


__all__ = [
    "AvailabilityChecks",
    "AvailabilityChecksFactory",
    "MainWindow",
    "SettingsDialogFactory",
]
