"""HyperGen application entry point."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from hypergen.application.document_controller import DocumentController
from hypergen.application.document_session import DocumentSession, DocumentSessionError
from hypergen.application.workers import AdapterKind, AdapterWorkers
from hypergen.domain.models import Stack
from hypergen.generation.errors import ModelUnavailableError
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings
from hypergen.ui.main_window import AvailabilityChecksFactory, MainWindow
from hypergen.ui.project_paths import default_project_directory
from hypergen.ui.settings_dialog import SettingsStore, load_machine_settings
from hypergen.ui.welcome_dialog import WelcomeDialog, WelcomeSelection


def build_availability_checks(
    settings: SettingsStore,
) -> Mapping[AdapterKind, Callable[[], Any]]:
    """Build lazy checks whose adapter objects are constructed only on worker threads."""
    values = load_machine_settings(settings)

    def check_ollama() -> None:
        OllamaRuntime(
            OllamaSettings(
                endpoint=values.ollama_endpoint,
                model=values.ollama_model,
            )
        ).require_model()

    def check_mflux() -> None:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        from mflux.models.common.config import ModelConfig
        from mflux.models.flux2.weights.flux2_weight_definition import (
            Flux2KleinWeightDefinition,
        )

        configurations = {
            "flux2-klein-4b": ModelConfig.flux2_klein_4b,
            "flux2-klein-9b": ModelConfig.flux2_klein_9b,
        }
        configuration_factory = configurations.get(values.mflux_model)
        if configuration_factory is None:
            raise ModelUnavailableError(
                f"MFLUX model tag {values.mflux_model!r} is unsupported. "
                "Choose flux2-klein-4b or flux2-klein-9b."
            )
        repository = configuration_factory().model_name
        try:
            snapshot_download(
                repo_id=repository,
                allow_patterns=Flux2KleinWeightDefinition.get_download_patterns(),
                local_files_only=True,
            )
        except LocalEntryNotFoundError as error:
            raise ModelUnavailableError(
                f"MFLUX model {values.mflux_model!r} is not available in the local "
                "Hugging Face cache. Download or authenticate with Hugging Face "
                "outside HyperGen, then retry."
            ) from error

    return {
        AdapterKind.OLLAMA: check_ollama,
        AdapterKind.MFLUX: check_mflux,
    }


def build_main_window(
    *,
    controller: DocumentController | None = None,
    workers: AdapterWorkers | None = None,
    settings: SettingsStore | None = None,
    availability_checks: Mapping[AdapterKind, Callable[[], Any]] | None = None,
    availability_checks_factory: AvailabilityChecksFactory | None = None,
    document_session: DocumentSession | None = None,
    project_directory: Path | None = None,
    start_diagnostics: bool = False,
) -> MainWindow:
    """Construct an injectable shell without creating live model clients."""
    document_controller = (
        controller if controller is not None else DocumentController(Stack(name="Untitled Stack"))
    )
    adapter_workers = workers if workers is not None else AdapterWorkers()
    machine_settings = settings if settings is not None else QSettings()
    active_session = (
        document_session
        if document_session is not None
        else (DocumentSession(document_controller) if controller is None else None)
    )
    return MainWindow(
        document_controller,
        adapter_workers,
        machine_settings,
        availability_checks=availability_checks,
        availability_checks_factory=availability_checks_factory,
        document_session=active_session,
        project_directory=project_directory,
        start_diagnostics=start_diagnostics,
        owns_workers=workers is None,
    )


def main() -> int:
    """Run the HyperGen application."""
    application = QApplication.instance() or QApplication(sys.argv)
    application.setOrganizationName("HyperGen")
    application.setApplicationName("HyperGen")
    settings = QSettings()
    project_directory = default_project_directory()
    controller = DocumentController(Stack(name="Welcome"))
    session = DocumentSession(controller)
    while session.store is None:
        welcome = WelcomeDialog(project_directory)
        if welcome.exec() != QDialog.DialogCode.Accepted:
            return 0
        selection = welcome.selection
        if selection is None:
            continue
        try:
            _apply_welcome_selection(session, selection)
        except DocumentSessionError as error:
            QMessageBox.critical(
                None,
                (
                    "Could Not Open Project"
                    if selection.stack is None
                    else "Could Not Create Project"
                ),
                str(error),
            )
    window = build_main_window(
        controller=controller,
        settings=settings,
        availability_checks_factory=lambda: build_availability_checks(settings),
        document_session=session,
        project_directory=project_directory,
        start_diagnostics=True,
    )
    window.show()
    return application.exec()


def _apply_welcome_selection(
    session: DocumentSession,
    selection: WelcomeSelection,
) -> None:
    """Bind the startup selection before constructing the authoring shell."""
    if selection.stack is None:
        session.open(selection.bundle_path)
    else:
        session.create(selection.stack, selection.bundle_path)
