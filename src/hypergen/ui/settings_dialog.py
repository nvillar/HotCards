"""Machine-local model and service settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

OLLAMA_ENDPOINT_KEY = "services/ollama_endpoint"
OLLAMA_MODEL_KEY = "services/ollama_model"
MFLUX_MODEL_KEY = "generation/mflux_model"
STEP_COUNT_KEY = "generation/step_count"
QUANTIZATION_KEY = "generation/quantization"
RANDOM_SEED_KEY = "generation/random_seed"
FIXED_SEED_KEY = "generation/fixed_seed"


class SettingsStore(Protocol):
    """Small QSettings surface used by the dialog and bootstrap."""

    def value(self, key: str, default_value: Any = None) -> Any: ...

    def setValue(self, key: str, value: Any) -> None: ...

    def sync(self) -> None: ...


@dataclass(frozen=True, slots=True)
class MachineSettings:
    """Non-portable generation configuration stored outside stack documents."""

    ollama_endpoint: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:9b"
    mflux_model: str = "flux2-klein-4b"
    step_count: int = 4
    quantization: int | None = None
    random_seed: bool = True
    fixed_seed: int = 42


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"true", "1", "yes"}:
            return True
        if value.lower() in {"false", "0", "no"}:
            return False
    if isinstance(value, int):
        return bool(value)
    return default


def load_machine_settings(settings: SettingsStore) -> MachineSettings:
    """Read machine-local settings with safe defaults."""
    defaults = MachineSettings()
    quantization_value = settings.value(QUANTIZATION_KEY, "")
    quantization = _int_value(quantization_value, 0) if str(quantization_value).strip() else None
    return MachineSettings(
        ollama_endpoint=str(settings.value(OLLAMA_ENDPOINT_KEY, defaults.ollama_endpoint)).strip(),
        ollama_model=str(settings.value(OLLAMA_MODEL_KEY, defaults.ollama_model)).strip(),
        mflux_model=str(settings.value(MFLUX_MODEL_KEY, defaults.mflux_model)).strip(),
        step_count=max(
            1,
            _int_value(
                settings.value(STEP_COUNT_KEY, defaults.step_count),
                defaults.step_count,
            ),
        ),
        quantization=quantization,
        random_seed=_bool_value(
            settings.value(RANDOM_SEED_KEY, defaults.random_seed),
            defaults.random_seed,
        ),
        fixed_seed=_int_value(
            settings.value(FIXED_SEED_KEY, defaults.fixed_seed),
            defaults.fixed_seed,
        ),
    )


class SettingsDialog(QDialog):
    """Edit only machine-local settings; credentials remain external."""

    def __init__(
        self,
        settings: SettingsStore | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings if settings is not None else QSettings()
        self.setWindowTitle("Advanced Settings")
        self.setObjectName("advancedSettingsDialog")
        self.setMinimumWidth(440)

        self.ollama_endpoint_edit = QLineEdit()
        self.ollama_endpoint_edit.setObjectName("ollamaEndpointEdit")
        self.ollama_model_edit = QLineEdit()
        self.ollama_model_edit.setObjectName("ollamaModelEdit")
        self.mflux_model_edit = QLineEdit()
        self.mflux_model_edit.setObjectName("mfluxModelEdit")

        self.step_count_spin = QSpinBox()
        self.step_count_spin.setObjectName("stepCountSpin")
        self.step_count_spin.setRange(1, 100)
        self.quantization_combo = QComboBox()
        self.quantization_combo.setObjectName("quantizationCombo")
        self.quantization_combo.addItem("None", None)
        for bits in (8, 6, 4):
            self.quantization_combo.addItem(f"{bits}-bit", bits)

        self.random_seed_check = QCheckBox("Choose a new seed for each generation")
        self.random_seed_check.setObjectName("randomSeedCheck")
        self.fixed_seed_spin = QSpinBox()
        self.fixed_seed_spin.setObjectName("fixedSeedSpin")
        self.fixed_seed_spin.setRange(0, 2_147_483_647)
        self.random_seed_check.toggled.connect(
            lambda checked: self.fixed_seed_spin.setEnabled(not checked)
        )

        form = QFormLayout()
        form.addRow("Ollama endpoint", self.ollama_endpoint_edit)
        form.addRow("Ollama model tag", self.ollama_model_edit)
        form.addRow("MFLUX model tag", self.mflux_model_edit)
        form.addRow("Inference steps", self.step_count_spin)
        form.addRow("Quantization", self.quantization_combo)
        form.addRow("Seed behavior", self.random_seed_check)
        form.addRow("Fixed seed", self.fixed_seed_spin)

        auth_note = QLabel(
            "Hugging Face authentication is managed outside HyperGen. "
            "This application never asks for or stores tokens or credentials."
        )
        auth_note.setObjectName("externalAuthenticationNote")
        auth_note.setWordWrap(True)
        self.validation_error = QLabel()
        self.validation_error.setObjectName("settingsValidationError")
        self.validation_error.setWordWrap(True)
        self.validation_error.setVisible(False)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(auth_note)
        layout.addWidget(self.validation_error)
        layout.addWidget(buttons)
        self.load()

    def load(self) -> MachineSettings:
        """Load persisted values into the controls."""
        values = load_machine_settings(self.settings)
        self.ollama_endpoint_edit.setText(values.ollama_endpoint)
        self.ollama_model_edit.setText(values.ollama_model)
        self.mflux_model_edit.setText(values.mflux_model)
        self.step_count_spin.setValue(values.step_count)
        quantization_index = self.quantization_combo.findData(values.quantization)
        self.quantization_combo.setCurrentIndex(max(0, quantization_index))
        self.random_seed_check.setChecked(values.random_seed)
        self.fixed_seed_spin.setValue(values.fixed_seed)
        self.fixed_seed_spin.setEnabled(not values.random_seed)
        return values

    def machine_settings(self) -> MachineSettings:
        """Return the values currently displayed by the dialog."""
        return MachineSettings(
            ollama_endpoint=_validated_ollama_endpoint(self.ollama_endpoint_edit.text().strip()),
            ollama_model=self.ollama_model_edit.text().strip(),
            mflux_model=self.mflux_model_edit.text().strip(),
            step_count=self.step_count_spin.value(),
            quantization=self.quantization_combo.currentData(),
            random_seed=self.random_seed_check.isChecked(),
            fixed_seed=self.fixed_seed_spin.value(),
        )

    def save(self) -> MachineSettings:
        """Persist non-sensitive values to the injected QSettings-compatible store."""
        values = self.machine_settings()
        self.settings.setValue(OLLAMA_ENDPOINT_KEY, values.ollama_endpoint)
        self.settings.setValue(OLLAMA_MODEL_KEY, values.ollama_model)
        self.settings.setValue(MFLUX_MODEL_KEY, values.mflux_model)
        self.settings.setValue(STEP_COUNT_KEY, values.step_count)
        self.settings.setValue(
            QUANTIZATION_KEY,
            "" if values.quantization is None else values.quantization,
        )
        self.settings.setValue(RANDOM_SEED_KEY, values.random_seed)
        self.settings.setValue(FIXED_SEED_KEY, values.fixed_seed)
        self.settings.sync()
        return values

    def accept(self) -> None:
        try:
            self.save()
        except ValueError as error:
            self.validation_error.setText(str(error))
            self.validation_error.setVisible(True)
            return
        self.validation_error.clear()
        self.validation_error.setVisible(False)
        super().accept()


def _validated_ollama_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Ollama endpoint must be an HTTP(S) service URL without credentials, "
            "query parameters, or fragments."
        )
    return value


__all__ = [
    "FIXED_SEED_KEY",
    "MFLUX_MODEL_KEY",
    "MachineSettings",
    "OLLAMA_ENDPOINT_KEY",
    "OLLAMA_MODEL_KEY",
    "QUANTIZATION_KEY",
    "RANDOM_SEED_KEY",
    "STEP_COUNT_KEY",
    "SettingsDialog",
    "SettingsStore",
    "load_machine_settings",
]
