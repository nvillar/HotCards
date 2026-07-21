"""Offscreen tests for the issue #11 application shell."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QModelIndex, QObject, Signal
from PySide6.QtWidgets import QApplication

from hypergen.application.document_controller import DocumentController
from hypergen.application.workers import (
    AdapterKind,
    WorkerFailure,
    WorkerFailureKind,
)
from hypergen.domain.models import (
    Card,
    HotspotSet,
    ImageOrigin,
    ImageRevision,
    Interaction,
    NavigateAction,
    Point,
    Polygon,
    Stack,
    UnresolvedCardReference,
)
from hypergen.main import build_availability_checks, build_main_window
from hypergen.ui.main_window import MainWindow
from hypergen.ui.settings_dialog import (
    MachineSettings,
    SettingsDialog,
    load_machine_settings,
)


class FakeSettings:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.synced = False

    def value(self, key: str, default_value: Any = None) -> Any:
        return self.values.get(key, default_value)

    def setValue(self, key: str, value: Any) -> None:
        self.values[key] = value

    def sync(self) -> None:
        self.synced = True


class FakeOperation(QObject):
    succeeded = Signal(object)
    failed = Signal(object)
    finished = Signal()


class FakeWorkers(QObject):
    availability_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.ollama_checks: list[object] = []
        self.mflux_checks: list[object] = []
        self.ollama_operations: list[FakeOperation] = []
        self.mflux_operations: list[FakeOperation] = []
        self.shutdown_calls = 0

    def check_ollama(
        self,
        check: object,
        *,
        emit_diagnostic: bool = True,
    ) -> FakeOperation:
        assert not emit_diagnostic
        self.ollama_checks.append(check)
        operation = FakeOperation()
        self.ollama_operations.append(operation)
        return operation

    def check_mflux(
        self,
        check: object,
        *,
        emit_diagnostic: bool = True,
    ) -> FakeOperation:
        assert not emit_diagnostic
        self.mflux_checks.append(check)
        operation = FakeOperation()
        self.mflux_operations.append(operation)
        return operation

    def shutdown(self, *, wait_milliseconds: int = 0) -> None:
        self.shutdown_calls += 1


@pytest.fixture(scope="module")
def application() -> QApplication:
    return QApplication.instance() or QApplication([])


def loaded_stack() -> Stack:
    hotspot = Interaction(
        label="Door",
        action=NavigateAction(target=UnresolvedCardReference(target_name="Hall")),
        polygons=(
            Polygon(
                points=(
                    Point(x=0.1, y=0.1),
                    Point(x=0.3, y=0.1),
                    Point(x=0.2, y=0.4),
                )
            ),
        ),
    )
    revision = ImageRevision(
        image_path="assets/cards/foyer/background.png",
        origin=ImageOrigin.IMPORTED,
        hotspot_set=HotspotSet(interactions=(hotspot,)),
        created_at=datetime.now(UTC),
    )
    foyer = Card(
        name="Foyer",
        scene_description="A quiet entrance",
        image_revisions=(revision,),
        active_revision_id=revision.id,
    )
    hall = Card(name="Hall")
    return Stack(name="Demo", cards=(foyer, hall), start_card_id=foyer.id)


def make_window(
    stack: Stack | None = None,
    *,
    start_diagnostics: bool = False,
) -> tuple[MainWindow, DocumentController, FakeWorkers, FakeSettings]:
    controller = DocumentController(stack or loaded_stack())
    workers = FakeWorkers()
    settings = FakeSettings()
    window = MainWindow(
        controller,
        workers,  # type: ignore[arg-type]
        settings,
        availability_checks={
            AdapterKind.OLLAMA: lambda: None,
            AdapterKind.MFLUX: lambda: None,
        },
        start_diagnostics=start_diagnostics,
    )
    return window, controller, workers, settings


def test_three_panes_render_loaded_stack_in_sidebar_and_inspector(
    application: QApplication,
) -> None:
    window, _controller, _workers, _settings = make_window()

    assert window.pane_splitter.count() == 3
    assert window.card_sidebar.card_list.count() == 2
    assert window.card_sidebar.card_list.item(0).text() == "★  Foyer"
    assert window.inspector.card_name_edit.text() == "Foyer"
    assert window.inspector.background_value.text().endswith("background.png")
    assert window.inspector.hotspot_list.item(0).text() == "Door"
    assert window.card_sidebar.findChild(QObject, "addCardButton") is not None
    window.close()


def test_add_rename_and_start_card_mutations_use_controller(
    application: QApplication,
) -> None:
    window, controller, _workers, _settings = make_window()

    created_id = window.card_sidebar.add_card("Library")
    assert [card.name for card in controller.document.cards] == ["Foyer", "Hall", "Library"]
    assert window.card_sidebar.selected_card_id == created_id
    assert window.inspector.selected_card_id == created_id
    window.inspector.card_name_edit.setText("Archive")
    window.inspector.commit_card_metadata()
    assert controller.document.cards[-1].name == "Archive"
    window.inspector.start_card_check.setChecked(True)
    assert controller.document.start_card_id == created_id
    assert window.card_sidebar.card_list.item(2).text() == "★  Archive"
    window.close()


def test_drag_reorder_is_one_undoable_controller_command(
    application: QApplication,
) -> None:
    base = loaded_stack()
    stack = base.model_copy(update={"cards": (*base.cards, Card(name="Tower"))})
    window, controller, _workers, _settings = make_window(stack)

    assert window.card_sidebar.card_list.model().moveRow(
        QModelIndex(),
        0,
        QModelIndex(),
        3,
    )
    assert [card.name for card in controller.document.cards] == ["Hall", "Tower", "Foyer"]
    assert controller.undo()
    assert [card.name for card in controller.document.cards] == ["Foyer", "Hall", "Tower"]
    window.close()


def test_invalid_card_name_is_rejected_and_inspector_is_restored(
    application: QApplication,
) -> None:
    window, controller, _workers, _settings = make_window()

    window.inspector.card_name_edit.clear()
    window.inspector.commit_card_metadata()

    assert controller.document.cards[0].name == "Foyer"
    assert window.inspector.card_name_edit.text() == "Foyer"
    assert not window.inspector.validation_error.isHidden()
    window.close()


def test_inspector_sections_progressively_disclose(application: QApplication) -> None:
    window, _controller, _workers, _settings = make_window()

    assert window.inspector.card_section.isChecked()
    assert not window.inspector.background_section.isChecked()
    assert not window.inspector.hotspots_section.isChecked()
    window.inspector.background_section.setChecked(True)
    assert window.inspector.background_section.isChecked()
    window.close()


def test_settings_round_trip_excludes_credentials_and_stack(
    application: QApplication,
) -> None:
    settings = FakeSettings()
    assert load_machine_settings(settings) == MachineSettings()
    stack = loaded_stack()
    before = stack.model_dump_json()
    dialog = SettingsDialog(settings)
    dialog.ollama_endpoint_edit.setText("http://example.test:11434")
    dialog.ollama_model_edit.setText("local-model:latest")
    dialog.mflux_model_edit.setText("flux2-klein-9b")
    dialog.step_count_spin.setValue(12)
    dialog.quantization_combo.setCurrentIndex(dialog.quantization_combo.findData(8))
    dialog.random_seed_check.setChecked(False)
    dialog.fixed_seed_spin.setValue(8675309)
    dialog.save()

    values = load_machine_settings(settings)
    assert values.ollama_endpoint == "http://example.test:11434"
    assert values.ollama_model == "local-model:latest"
    assert values.mflux_model == "flux2-klein-9b"
    assert values.step_count == 12
    assert values.quantization == 8
    assert not values.random_seed
    assert values.fixed_seed == 8675309
    assert settings.synced
    assert all(
        forbidden not in key.casefold()
        for key in settings.values
        for forbidden in ("credential", "token", "password", "hugging")
    )
    assert stack.model_dump_json() == before
    assert "never asks for or stores" in dialog.findChild(
        QObject, "externalAuthenticationNote"
    ).property("text")
    dialog.ollama_endpoint_edit.setText("http://user:secret@example.test:11434")
    with pytest.raises(ValueError, match="without credentials"):
        dialog.save()
    assert settings.values["services/ollama_endpoint"] == "http://example.test:11434"
    dialog.close()


def test_toolbar_has_modes_overlay_and_disabled_navigation_placeholders(
    application: QApplication,
) -> None:
    window, controller, _workers, _settings = make_window()

    assert [window.mode_selector.itemText(index) for index in range(2)] == [
        "Author",
        "Run",
    ]
    before = controller.document
    window.mode_selector.setCurrentText("Run")
    assert controller.document == before
    assert all(not action.isEnabled() for action in window.player_navigation_actions)
    window.overlay_selector.setCurrentText("Visible")
    assert controller.document.run_overlay_mode.value == "visible"
    window.close()


def test_service_diagnostics_toggle_only_generation_actions(
    application: QApplication,
) -> None:
    window, _controller, workers, _settings = make_window(start_diagnostics=True)
    assert len(workers.ollama_checks) == 1
    assert len(workers.mflux_checks) == 1
    assert not window.generate_background_action.isEnabled()
    assert not window.generate_hotspots_action.isEnabled()
    assert window.card_sidebar.add_button.isEnabled()

    unavailable = WorkerFailure(
        adapter=AdapterKind.MFLUX,
        stage="checking MFLUX model availability",
        kind=WorkerFailureKind.MODEL_UNAVAILABLE,
        message="MFLUX model is unavailable",
    )
    workers.mflux_operations[0].failed.emit(unavailable)
    workers.ollama_operations[0].succeeded.emit(None)
    assert not window.generate_background_action.isEnabled()
    assert window.generate_hotspots_action.isEnabled()
    assert window.card_sidebar.add_button.isEnabled()

    workers.mflux_operations[0].succeeded.emit(None)
    assert window.generate_background_action.isEnabled()
    window.close()


def test_service_rechecks_use_latest_completion_and_reenable_actions(
    application: QApplication,
) -> None:
    window, _controller, workers, _settings = make_window(start_diagnostics=True)
    old_ollama = workers.ollama_operations[0]
    old_mflux = workers.mflux_operations[0]
    old_ollama.succeeded.emit(None)
    old_mflux.succeeded.emit(None)
    assert window.generate_background_action.isEnabled()

    window.run_availability_checks()
    assert not window.generate_background_action.isEnabled()
    assert "check pending" in window.service_status_label.text()
    assert "is available" not in window.service_status_label.text()
    workers.ollama_operations[1].succeeded.emit(None)
    workers.mflux_operations[1].succeeded.emit(None)
    assert window.generate_background_action.isEnabled()

    stale_failure = WorkerFailure(
        adapter=AdapterKind.MFLUX,
        stage="old check",
        kind=WorkerFailureKind.MODEL_UNAVAILABLE,
        message="stale failure",
    )
    old_mflux.failed.emit(stale_failure)
    assert window.generate_background_action.isEnabled()
    window.close()


def test_availability_checks_capture_settings_before_worker_execution() -> None:
    settings = FakeSettings()
    settings.values["services/ollama_endpoint"] = "http://localhost:11434"
    checks = build_availability_checks(settings)
    settings.values["services/ollama_endpoint"] = "http://changed.invalid"

    assert checks[AdapterKind.OLLAMA].__closure__ is not None
    captured = [
        cell.cell_contents
        for cell in checks[AdapterKind.OLLAMA].__closure__
        if isinstance(cell.cell_contents, MachineSettings)
    ]
    assert captured[0].ollama_endpoint == "http://localhost:11434"


def test_bootstrap_construction_uses_injected_services_without_live_clients(
    application: QApplication,
) -> None:
    controller = DocumentController(Stack(name="Injected"))
    workers = FakeWorkers()
    window = build_main_window(
        controller=controller,
        workers=workers,  # type: ignore[arg-type]
        settings=FakeSettings(),
    )

    assert window.controller is controller
    assert window.workers is workers
    assert not workers.ollama_checks
    assert not workers.mflux_checks
    assert window.windowTitle() == "HyperGen — Injected"
    window.close()
