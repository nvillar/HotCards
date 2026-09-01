"""Run-mode and author-preview sound playback boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PySide6.QtCore import QObject, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer


class SoundPlayer(Protocol):
    """Minimal replace-current playback interface used by the UI."""

    def play(self, path: Path) -> None:
        """Stop any current sound and start the given WAV."""

    def stop(self) -> None:
        """Stop current playback."""


class QtSoundPlayer(QObject):
    """Play one sound at a time through QtMultimedia."""

    def __init__(self, *, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._audio_output = QAudioOutput(self)
        self._player = QMediaPlayer(self)
        self._player.setAudioOutput(self._audio_output)

    @property
    def is_playing(self) -> bool:
        return self._player.playbackState() is QMediaPlayer.PlaybackState.PlayingState

    def play(self, path: Path) -> None:
        self._player.stop()
        self._player.setSource(QUrl.fromLocalFile(str(path)))
        self._player.play()

    def stop(self) -> None:
        self._player.stop()


__all__ = ["QtSoundPlayer", "SoundPlayer"]
