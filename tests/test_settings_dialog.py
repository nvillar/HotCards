"""Tests for machine-local settings and model selector ownership."""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from hypergen.ui.settings_dialog import (
    MFLUX_MODEL_KEY,
    OLLAMA_MODEL_KEY,
    SettingsDialog,
    load_machine_settings,
)


class FakeSettings:
    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self.values = dict(values or {})

    def value(self, key: str, default_value: Any = None) -> Any:
        return self.values.get(key, default_value)

    def setValue(self, key: str, value: Any) -> None:
        self.values[key] = value

    def sync(self) -> None:
        pass


@pytest.fixture(scope="module")
def application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_legacy_9b_selection_maps_to_9b_kv() -> None:
    settings = FakeSettings({MFLUX_MODEL_KEY: "flux2-klein-9b"})

    assert load_machine_settings(settings).mflux_model == "flux2-klein-9b-kv"


def test_advanced_settings_preserve_bottom_bar_model_selections(
    application: QApplication,
) -> None:
    settings = FakeSettings(
        {
            OLLAMA_MODEL_KEY: "llama3.2:latest",
            MFLUX_MODEL_KEY: "flux2-klein-9b-kv",
        }
    )
    dialog = SettingsDialog(settings)

    assert not hasattr(dialog, "ollama_model_edit")
    assert not hasattr(dialog, "mflux_model_edit")
    dialog.ollama_endpoint_edit.setText("http://localhost:11435")
    saved = dialog.save()

    assert saved.ollama_model == "llama3.2:latest"
    assert saved.mflux_model == "flux2-klein-9b-kv"
    assert settings.values[OLLAMA_MODEL_KEY] == "llama3.2:latest"
    assert settings.values[MFLUX_MODEL_KEY] == "flux2-klein-9b-kv"
