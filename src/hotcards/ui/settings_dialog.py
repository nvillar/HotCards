"""Machine-local model settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QVBoxLayout,
    QWidget,
)

MFLUX_MODEL_KEY = "generation/mflux_model"
STABLE_AUDIO_MODEL_KEY = "generation/stable_audio_model"
MFLUX_MODEL_OPTIONS = (
    ("FLUX.2 Klein 4B", "flux2-klein-4b"),
    ("FLUX.2 Klein 9B KV", "flux2-klein-9b-kv"),
)
STABLE_AUDIO_MODEL_OPTIONS = (
    (
        "Stable Audio 3 Small-SFX",
        "stabilityai/stable-audio-3-small-sfx",
    ),
)


class SettingsStore(Protocol):
    """Small QSettings surface used by the dialog and bootstrap."""

    def value(self, key: str, default_value: Any = None) -> Any: ...

    def setValue(self, key: str, value: Any) -> None: ...

    def sync(self) -> None: ...


@dataclass(frozen=True, slots=True)
class MachineSettings:
    """Non-portable generation configuration stored outside stack documents."""

    mflux_model: str = "flux2-klein-4b"
    stable_audio_model: str = "stabilityai/stable-audio-3-small-sfx"
    step_count: int = 4
    quantization: int | None = None
    random_seed: bool = True
    fixed_seed: int = 42


def load_machine_settings(settings: SettingsStore) -> MachineSettings:
    """Read machine-local settings with safe defaults."""
    defaults = MachineSettings()
    mflux_model = str(settings.value(MFLUX_MODEL_KEY, defaults.mflux_model)).strip()
    if mflux_model == "flux2-klein-9b":
        mflux_model = "flux2-klein-9b-kv"
    supported_mflux_models = {value for _label, value in MFLUX_MODEL_OPTIONS}
    if mflux_model not in supported_mflux_models:
        mflux_model = defaults.mflux_model
    stable_audio_model = str(
        settings.value(
            STABLE_AUDIO_MODEL_KEY,
            defaults.stable_audio_model,
        )
    ).strip()
    supported_stable_audio_models = {value for _label, value in STABLE_AUDIO_MODEL_OPTIONS}
    if stable_audio_model not in supported_stable_audio_models:
        stable_audio_model = defaults.stable_audio_model
    return MachineSettings(
        mflux_model=mflux_model,
        stable_audio_model=stable_audio_model,
    )


class SettingsDialog(QDialog):
    """Edit machine-local model selections."""

    def __init__(
        self,
        settings: SettingsStore | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings if settings is not None else QSettings()
        self.setWindowTitle("Model Settings")
        self.setObjectName("modelSettingsDialog")
        self.setMinimumWidth(440)

        models_form = QFormLayout()
        self.image_model_combo = QComboBox()
        self.image_model_combo.setObjectName("settingsImageModelCombo")
        self.image_model_combo.setAccessibleName("Image model")
        for label, model in MFLUX_MODEL_OPTIONS:
            self.image_model_combo.addItem(label, model)
        models_form.addRow("Image", self.image_model_combo)
        self.sound_model_combo = QComboBox()
        self.sound_model_combo.setObjectName("settingsSoundModelCombo")
        self.sound_model_combo.setAccessibleName("Sound model")
        for label, model in STABLE_AUDIO_MODEL_OPTIONS:
            self.sound_model_combo.addItem(label, model)
        models_form.addRow("Sound", self.sound_model_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(models_form)
        layout.addWidget(buttons)
        self.load()

    def load(self) -> MachineSettings:
        """Load persisted values into the controls."""
        values = load_machine_settings(self.settings)
        self.image_model_combo.setCurrentIndex(
            max(0, self.image_model_combo.findData(values.mflux_model))
        )
        self.sound_model_combo.setCurrentIndex(
            max(
                0,
                self.sound_model_combo.findData(values.stable_audio_model),
            )
        )
        return values

    def machine_settings(self) -> MachineSettings:
        """Return the values currently displayed by the dialog."""
        return MachineSettings(
            mflux_model=str(self.image_model_combo.currentData()),
            stable_audio_model=str(self.sound_model_combo.currentData()),
        )

    def save(self) -> MachineSettings:
        """Persist non-sensitive values to the injected QSettings-compatible store."""
        values = self.machine_settings()
        self.settings.setValue(MFLUX_MODEL_KEY, values.mflux_model)
        self.settings.setValue(
            STABLE_AUDIO_MODEL_KEY,
            values.stable_audio_model,
        )
        self.settings.sync()
        return values

    def accept(self) -> None:
        self.save()
        super().accept()


__all__ = [
    "MFLUX_MODEL_KEY",
    "MFLUX_MODEL_OPTIONS",
    "MachineSettings",
    "STABLE_AUDIO_MODEL_KEY",
    "STABLE_AUDIO_MODEL_OPTIONS",
    "SettingsDialog",
    "SettingsStore",
    "load_machine_settings",
]
