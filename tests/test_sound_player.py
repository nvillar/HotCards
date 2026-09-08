"""Native playback errors reach the notification-facing boundary."""

from PySide6.QtMultimedia import QMediaPlayer

from hotcards.application.sound_player import QtSoundPlayer


def test_native_playback_error_is_reported_without_a_manager_window() -> None:
    player = QtSoundPlayer()
    failures: list[str] = []
    player.playback_failed.connect(failures.append)
    try:
        player._player.errorOccurred.emit(QMediaPlayer.Error.ResourceError, "  Device lost  ")
        player._player.errorOccurred.emit(QMediaPlayer.Error.FormatError, "")
        player._player.errorOccurred.emit(QMediaPlayer.Error.NoError, "")
        assert failures == ["Device lost", "Sound playback failed (FormatError)"]
    finally:
        player.stop()
        player.deleteLater()
