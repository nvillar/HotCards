from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from hotcards.generation.stable_audio import (
    STABLE_AUDIO_CHANNELS,
    STABLE_AUDIO_SAMPLE_RATE,
    StableAudioCancellation,
    StableAudioCancelled,
    StableAudioError,
    StableAudioFailureKind,
    StableAudioGenerator,
    StableAudioRequest,
)


def test_request_trims_prompt_and_validates_duration_and_seed() -> None:
    request = StableAudioRequest("  wooden door closes  ", duration_seconds=2, seed=17)

    assert request.prompt == "wooden door closes"
    with pytest.raises(ValueError, match="must not be empty"):
        StableAudioRequest(" ")
    with pytest.raises(ValueError, match="between 1 and 30"):
        StableAudioRequest("wind", duration_seconds=31)
    with pytest.raises(ValueError, match="between 0 and 4294967295"):
        StableAudioRequest("wind", seed=-1)


def test_availability_check_only_reads_the_local_hugging_face_cache(tmp_path: Path) -> None:
    calls: list[tuple[str, str, bool]] = []

    def download(*, repo_id: str, filename: str, local_files_only: bool) -> str:
        calls.append((repo_id, filename, local_files_only))
        return str(tmp_path / Path(filename).name)

    StableAudioGenerator(download=download).check_available()

    assert len(calls) == 3
    assert all(local_files_only for _, _, local_files_only in calls)


def test_availability_failure_has_a_typed_kind() -> None:
    def unavailable(**_kwargs: object) -> str:
        raise FileNotFoundError("not cached")

    with pytest.raises(StableAudioError) as caught:
        StableAudioGenerator(download=unavailable).check_available()

    assert caught.value.kind is StableAudioFailureKind.MODEL_UNAVAILABLE


def test_precancelled_generation_never_loads_models(tmp_path: Path) -> None:
    cancellation = StableAudioCancellation()
    cancellation.request()

    def download(*, filename: str, **_kwargs: object) -> str:
        return str(tmp_path / Path(filename).name)

    with pytest.raises(StableAudioCancelled):
        StableAudioGenerator(download=download).generate(
            StableAudioRequest("glass tap"),
            tmp_path / "sound.wav",
            cancellation=cancellation,
        )

    assert not (tmp_path / "sound.wav").exists()


def test_wav_writer_creates_stereo_44100hz_pcm(tmp_path: Path) -> None:
    output_path = tmp_path / "sound.wav"
    audio = np.zeros((STABLE_AUDIO_CHANNELS, STABLE_AUDIO_SAMPLE_RATE), dtype=np.float32)

    StableAudioGenerator._write_wav(output_path, audio)

    with wave.open(str(output_path), "rb") as generated:
        assert generated.getnchannels() == 2
        assert generated.getsampwidth() == 2
        assert generated.getframerate() == 44_100
        assert generated.getnframes() == 44_100
