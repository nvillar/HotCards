"""Restrained three-pane HyperGen application shell."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any
from uuid import UUID

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QLabel,
    QMainWindow,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from hypergen.application.commands import SetRunOverlayModeCommand
from hypergen.application.document_controller import DocumentController
from hypergen.application.workers import (
    AdapterKind,
    AdapterWorkers,
    AvailabilityDiagnostic,
    WorkerOperation,
)
from hypergen.domain.models import RunOverlayMode, Stack
from hypergen.ui.card_sidebar import CardSidebar
from hypergen.ui.inspector import Inspector
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
        start_diagnostics: bool = True,
        owns_workers: bool = False,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.workers = workers
        self.settings = settings if settings is not None else QSettings()
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
        self.render_document(controller.document)
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
        self.mode_selector.setToolTip("Run behavior will be implemented in issue #14")
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
            action.setToolTip("Player navigation is a placeholder for issue #14")
            toolbar.addAction(action)

    def _build_panes(self) -> None:
        self.card_sidebar = CardSidebar(self.controller)
        self.card_sidebar.card_selected.connect(self.select_card)
        self.card_sidebar.document_changed.connect(self.render_document)

        canvas = QWidget()
        canvas.setObjectName("canvasPlaceholder")
        canvas_layout = QVBoxLayout(canvas)
        canvas_label = QLabel(
            "Canvas preview\n\nAuthoring canvas behavior will be added in issue #12."
        )
        canvas_label.setObjectName("canvasPlaceholderLabel")
        canvas_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        canvas_layout.addWidget(canvas_label)

        self.inspector = Inspector(self.controller)
        self.inspector.document_changed.connect(self.render_document)

        self.pane_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.pane_splitter.setObjectName("threePaneSplitter")
        self.pane_splitter.addWidget(self.card_sidebar)
        self.pane_splitter.addWidget(canvas)
        self.pane_splitter.addWidget(self.inspector)
        self.pane_splitter.setStretchFactor(0, 0)
        self.pane_splitter.setStretchFactor(1, 1)
        self.pane_splitter.setStretchFactor(2, 0)
        self.pane_splitter.setSizes([220, 700, 280])
        self.setCentralWidget(self.pane_splitter)

        self.service_status_label = QLabel("Checking local AI services…")
        self.service_status_label.setObjectName("serviceStatusLabel")
        self.statusBar().addPermanentWidget(self.service_status_label)

    def _build_menu(self) -> None:
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
            overlay_index = self.overlay_selector.findData(snapshot.run_overlay_mode)
            self.overlay_selector.setCurrentIndex(overlay_index)
            self.setWindowTitle(f"HyperGen — {snapshot.name}")
        finally:
            self._rendering = False

    def select_card(self, card_id: object) -> None:
        self._selected_card_id = card_id if isinstance(card_id, UUID) else None
        if self.card_sidebar.selected_card_id != self._selected_card_id:
            self.card_sidebar.select_card(self._selected_card_id)
        self.inspector.render(self.controller.document, self._selected_card_id)

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
        mflux_available = self._availability[AdapterKind.MFLUX] is True
        ollama_available = self._availability[AdapterKind.OLLAMA] is True
        self.generate_background_action.setEnabled(mflux_available and ollama_available)
        self.generate_hotspots_action.setEnabled(ollama_available)
        self.generate_background_action.setToolTip(
            " · ".join(
                self._action_diagnostic(adapter)
                for adapter in (AdapterKind.OLLAMA, AdapterKind.MFLUX)
            )
        )
        self.generate_hotspots_action.setToolTip(self._action_diagnostic(AdapterKind.OLLAMA))
        unavailable = [
            self._diagnostic_messages.get(adapter, f"{adapter.value} check pending")
            for adapter, available in self._availability.items()
            if available is not True
        ]
        self.service_status_label.setText(
            "Local AI services available"
            if not unavailable
            else "Generation unavailable: " + " · ".join(unavailable)
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

    def open_advanced_settings(self) -> None:
        dialog = self._settings_dialog_factory(self.settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.run_availability_checks()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._owns_workers:
            self.workers.shutdown(wait_milliseconds=100)
        super().closeEvent(event)


__all__ = [
    "AvailabilityChecks",
    "AvailabilityChecksFactory",
    "MainWindow",
    "SettingsDialogFactory",
]
