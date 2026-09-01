"""Tests for machine-local settings and model selector ownership."""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QLabel

from hotcards.ui.settings_dialog import (
    MFLUX_MODEL_KEY,
    STABLE_AUDIO_MODEL_KEY,
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


def test_settings_dialog_contains_only_image_and_sound_models(
    application: QApplication,
) -> None:
    settings = FakeSettings(
        {
            MFLUX_MODEL_KEY: "flux2-klein-9b-kv",
        }
    )
    dialog = SettingsDialog(settings)

    assert dialog.windowTitle() == "Model Settings"
    assert dialog.objectName() == "modelSettingsDialog"
    assert not hasattr(dialog, "models_group")
    assert dialog.image_model_combo.currentText() == "FLUX.2 Klein 9B KV"
    assert dialog.sound_model_combo.currentText() == "Stable Audio 3 Small-SFX"
    assert not hasattr(dialog, "step_count_spin")
    assert not hasattr(dialog, "quantization_combo")
    assert not hasattr(dialog, "random_seed_check")
    assert not hasattr(dialog, "fixed_seed_spin")
    assert dialog.findChild(QLabel, "externalAuthenticationNote") is None
    assert all(
        "Hugging Face" not in label.text()
        for label in dialog.findChildren(QLabel)
    )


def test_settings_dialog_persists_model_selections(
    application: QApplication,
) -> None:
    settings = FakeSettings()
    dialog = SettingsDialog(settings)
    dialog.image_model_combo.setCurrentIndex(
        dialog.image_model_combo.findData("flux2-klein-9b-kv")
    )

    saved = dialog.save()

    assert saved.mflux_model == "flux2-klein-9b-kv"
    assert saved.stable_audio_model == "stabilityai/stable-audio-3-small-sfx"
    assert settings.values[MFLUX_MODEL_KEY] == "flux2-klein-9b-kv"
    assert (
        settings.values[STABLE_AUDIO_MODEL_KEY]
        == "stabilityai/stable-audio-3-small-sfx"
    )


def test_hidden_generation_tuning_settings_use_fixed_defaults() -> None:
    settings = FakeSettings(
        {
            "generation/step_count": 99,
            "generation/quantization": 4,
            "generation/random_seed": False,
            "generation/fixed_seed": 7,
        }
    )

    values = load_machine_settings(settings)

    assert values.step_count == 4
    assert values.quantization is None
    assert values.random_seed
    assert values.fixed_seed == 42
