from __future__ import annotations

import wave
from collections.abc import Callable
from pathlib import Path
from threading import current_thread
from types import SimpleNamespace

import numpy as np
import pytest

import hotcards.generation.stable_audio as audio_module
from hotcards.generation.mflux_generator import run_model_invocation
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


def test_weights_missing_after_availability_never_trigger_a_download(tmp_path: Path) -> None:
    cache_only: list[bool] = []

    def download(*, filename: str, local_files_only: bool, **_kwargs: object) -> str:
        cache_only.append(local_files_only)
        if len(cache_only) > 3:
            raise FileNotFoundError("weights were removed from the cache")
        return str(tmp_path / Path(filename).name)

    generator = StableAudioGenerator(download=download)
    generator.check_available()
    with pytest.raises(StableAudioError) as caught:
        generator.generate(StableAudioRequest("glass tap"), tmp_path / "sound.wav")

    assert caught.value.kind is StableAudioFailureKind.MODEL_UNAVAILABLE
    assert cache_only == [True, True, True, True]
    assert not (tmp_path / "sound.wav").exists()


def test_precancelled_generation_never_loads_models(tmp_path: Path) -> None:
    cancellation = StableAudioCancellation()
    cancellation.request()

    def download(*, filename: str, **_kwargs: object) -> str:
        raise AssertionError("precancelled work must not resolve weights")

    with pytest.raises(StableAudioCancelled):
        StableAudioGenerator(download=download).generate(
            StableAudioRequest("glass tap"),
            tmp_path / "sound.wav",
            cancellation=cancellation,
        )

    assert not (tmp_path / "sound.wav").exists()


def fake_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    cancellation: StableAudioCancellation | None = None,
    cancel_at: str | None = None,
    fail_at: str | None = None,
) -> tuple[StableAudioGenerator, list[str], list[object]]:
    calls: list[str] = []
    threads: list[object] = []

    def stage(name: str) -> None:
        calls.append(name)
        threads.append(current_thread())
        if name == fail_at:
            raise OSError(f"injected {name} failure")
        if name == cancel_at:
            assert cancellation is not None
            cancellation.request()

    def download(*, filename: str, **_kwargs: object) -> str:
        assert _kwargs["local_files_only"] is True
        stage(f"resolve:{audio_module._WEIGHT_FILENAMES.index(filename)}")
        return str(tmp_path / Path(filename).name)

    class Encoder:
        @staticmethod
        def from_npz(_path: str) -> Encoder:
            stage("encoder-load")
            return Encoder()

        def encode(self, prompts: list[str], *, max_len: int) -> tuple[np.ndarray, np.ndarray]:
            stage("encode")
            assert prompts == ["glass tap"]
            assert max_len == 256
            return np.zeros((1, 2, 4)), np.ones((1, 2))

    def seconds(_duration: float) -> np.ndarray:
        stage("seconds")
        return np.zeros((1, 1, 4))

    def conditioner(*_args: object, **_kwargs: object) -> tuple[np.ndarray, object]:
        stage("conditioner-load")
        return np.zeros((1, 2, 4)), seconds

    def padding(embeddings: np.ndarray, *_args: object) -> np.ndarray:
        stage("padding")
        return embeddings

    def load_dit(*_args: object, **kwargs: object) -> object:
        stage("dit-load")
        assert kwargs["num_steps"] == audio_module.STABLE_AUDIO_STEPS
        return lambda latents, *_args, **_kwargs: latents

    def noise(shape: tuple[int, ...], **_kwargs: object) -> np.ndarray:
        stage("noise")
        return np.zeros(shape)

    def schedule(steps: int, **_kwargs: object) -> np.ndarray:
        stage("schedule")
        assert steps == audio_module.STABLE_AUDIO_STEPS
        return np.linspace(1, 0, steps + 1)

    def sample(
        model: Callable[..., np.ndarray],
        noise: np.ndarray,
        sigmas: np.ndarray,
        *,
        on_step: Callable[[int, int], None],
        before_step: Callable[[int], None],
        **_kwargs: object,
    ) -> np.ndarray:
        for index in range(len(sigmas) - 1):
            before_step(index)
            stage(f"sample:{index + 1}")
            model(noise, sigmas[index])
            on_step(index + 1, len(sigmas) - 1)
        return noise

    decode_count = 0

    def decode(latents: np.ndarray) -> np.ndarray:
        nonlocal decode_count
        stage(f"decode:{decode_count}")
        decode_count += 1
        return np.zeros((1, 512, latents.shape[-1] * 16))

    def load_decoder(*_args: object, **_kwargs: object) -> object:
        stage("decoder-load")
        return decode

    def chunked(
        decoder: Callable[..., np.ndarray],
        latents: np.ndarray,
        *_args: object,
    ) -> np.ndarray:
        midpoint = latents.shape[-1] // 2
        return np.concatenate(
            (decoder(latents[..., :midpoint]), decoder(latents[..., midpoint:])),
            axis=-1,
        )

    def patched(patches: np.ndarray, **_kwargs: object) -> np.ndarray:
        stage("patched-decode")
        return np.zeros((1, 2, patches.shape[-1] * 256))

    monkeypatch.setattr(
        audio_module,
        "mx",
        SimpleNamespace(
            float16=np.float16,
            float32=np.float32,
            eval=lambda *_args: None,
            concatenate=np.concatenate,
            random=SimpleNamespace(normal=noise, key=lambda seed: seed),
        ),
    )
    monkeypatch.setattr(audio_module, "T5Gemma", Encoder)
    monkeypatch.setattr(audio_module, "load_conditioner_from_npz", conditioner)
    monkeypatch.setattr(audio_module, "apply_prompt_padding", padding)
    monkeypatch.setattr(audio_module, "load_dit", load_dit)
    monkeypatch.setattr(audio_module, "build_pingpong_schedule", schedule)
    monkeypatch.setattr(audio_module, "sample_flow_pingpong", sample)
    monkeypatch.setattr(audio_module, "load_same_s_decoder", load_decoder)
    monkeypatch.setattr(audio_module, "decode_chunked", chunked)
    monkeypatch.setattr(audio_module, "patched_decode", patched)
    monkeypatch.setattr(audio_module, "_free_mlx_memory", lambda: stage("cleanup"))
    return StableAudioGenerator(download=download), calls, threads


@pytest.mark.parametrize(
    "stage",
    (
        "resolve:0",
        "resolve:1",
        "resolve:2",
        "encoder-load",
        "encode",
        "conditioner-load",
        "padding",
        "seconds",
        "dit-load",
        "noise",
        "schedule",
        *(f"sample:{step}" for step in range(1, 9)),
        "decoder-load",
        "decode:0",
        "decode:1",
        "patched-decode",
    ),
)
def test_cancellation_stops_at_every_audio_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    cancellation = StableAudioCancellation()
    generator, calls, _threads = fake_pipeline(
        monkeypatch, tmp_path, cancellation=cancellation, cancel_at=stage
    )
    output = tmp_path / "cancelled.wav"

    with pytest.raises(StableAudioCancelled):
        generator.generate(
            StableAudioRequest("glass tap", seed=17),
            output,
            cancellation=cancellation,
        )

    meaningful_calls = [call for call in calls if call != "cleanup"]
    assert meaningful_calls[-1] == stage
    assert generator._active_cancellation is None
    assert not output.exists()


@pytest.mark.parametrize(
    ("stage", "kind"),
    (
        ("resolve:1", StableAudioFailureKind.MODEL_UNAVAILABLE),
        ("encoder-load", StableAudioFailureKind.MODEL_LOAD),
        ("encode", StableAudioFailureKind.GENERATION),
        ("conditioner-load", StableAudioFailureKind.MODEL_LOAD),
        ("padding", StableAudioFailureKind.GENERATION),
        ("dit-load", StableAudioFailureKind.MODEL_LOAD),
        ("sample:3", StableAudioFailureKind.GENERATION),
        ("decoder-load", StableAudioFailureKind.MODEL_LOAD),
        ("decode:0", StableAudioFailureKind.GENERATION),
        ("patched-decode", StableAudioFailureKind.GENERATION),
        ("cleanup", StableAudioFailureKind.GENERATION),
    ),
)
def test_audio_failures_are_categorized_by_stage_not_retained_locals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    kind: StableAudioFailureKind,
) -> None:
    generator, _calls, _threads = fake_pipeline(monkeypatch, tmp_path, fail_at=stage)
    output = tmp_path / "failed.wav"

    with pytest.raises(StableAudioError) as caught:
        generator.generate(StableAudioRequest("glass tap"), output)

    assert caught.value.kind is kind
    assert not output.exists()
    assert generator._active_cancellation is None


def test_fake_audio_pipeline_runs_on_the_shared_native_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator, calls, threads = fake_pipeline(monkeypatch, tmp_path)
    output = tmp_path / "sound.wav"
    progress: list[tuple[int, int]] = []
    result = generator.generate(
        StableAudioRequest("glass tap", seed=17),
        output,
        on_sampling_step=lambda step, total: progress.append((step, total)),
    )

    assert result.seed == 17
    assert result.output_path == output
    assert progress == [(step, 8) for step in range(1, 9)]
    assert calls[-1] == "cleanup"
    owner = run_model_invocation(current_thread)
    assert owner is not current_thread()
    assert all(thread is owner for thread in threads)
    with wave.open(str(output), "rb") as generated:
        assert generated.getnframes() == 2 * STABLE_AUDIO_SAMPLE_RATE


def test_sampling_progress_cancellation_prevents_the_next_model_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancellation = StableAudioCancellation()
    generator, calls, _threads = fake_pipeline(monkeypatch, tmp_path)

    with pytest.raises(StableAudioCancelled):
        generator.generate(
            StableAudioRequest("glass tap"),
            tmp_path / "cancelled.wav",
            cancellation=cancellation,
            on_sampling_step=lambda *_args: cancellation.request(),
        )

    assert [call for call in calls if call.startswith("sample:")] == ["sample:1"]
    assert "decoder-load" not in calls


@pytest.mark.parametrize("cancel", (False, True))
def test_wav_failure_or_cancellation_disposes_only_the_created_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel: bool,
) -> None:
    cancellation = StableAudioCancellation()
    generator, _calls, _threads = fake_pipeline(monkeypatch, tmp_path)
    writeframes = wave.Wave_write.writeframes

    def interrupt(output: wave.Wave_write, data: bytes) -> None:
        writeframes(output, data)
        if cancel:
            cancellation.request()
        else:
            raise OSError("injected WAV serialization failure")

    monkeypatch.setattr(wave.Wave_write, "writeframes", interrupt)
    output = tmp_path / "incomplete.wav"
    with pytest.raises(StableAudioError) as caught:
        generator.generate(
            StableAudioRequest("glass tap"),
            output,
            cancellation=cancellation,
        )

    assert caught.value.kind is (
        StableAudioFailureKind.CANCELLED if cancel else StableAudioFailureKind.OUTPUT
    )
    assert not output.exists()


def test_wav_writer_never_overwrites_a_preexisting_file(tmp_path: Path) -> None:
    output = tmp_path / "existing.wav"
    output.write_bytes(b"foreign output")
    with pytest.raises(FileExistsError):
        StableAudioGenerator._write_wav(output, np.zeros((2, 10)))

    assert output.read_bytes() == b"foreign output"


@pytest.mark.parametrize("outcome", ("cancel", "error", "foreign-error"))
def test_audio_finalization_disposes_output_before_a_result_can_be_returned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    cancellation = StableAudioCancellation()
    generator, _calls, _threads = fake_pipeline(monkeypatch, tmp_path)
    output = tmp_path / "not-returned.wav"
    foreign = tmp_path / "foreign.wav"
    foreign.write_bytes(b"foreign output")
    result_type = audio_module.StableAudioResult

    def finalize(**kwargs: object) -> object:
        assert output.is_file()
        if outcome == "cancel":
            cancellation.request()
            return result_type(**kwargs)
        if outcome == "foreign-error":
            foreign.replace(output)
        raise RuntimeError("injected result finalization failure")

    monkeypatch.setattr(audio_module, "StableAudioResult", finalize)
    with pytest.raises(StableAudioError) as caught:
        generator.generate(
            StableAudioRequest("glass tap"),
            output,
            cancellation=cancellation,
        )

    assert caught.value.kind is (
        StableAudioFailureKind.CANCELLED if outcome == "cancel" else StableAudioFailureKind.OUTPUT
    )
    if outcome == "foreign-error":
        assert output.read_bytes() == b"foreign output"
    else:
        assert not output.exists()
    assert generator._active_cancellation is None


def test_audio_cleanup_failure_does_not_replace_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancellation = StableAudioCancellation()
    generator, _calls, _threads = fake_pipeline(
        monkeypatch,
        tmp_path,
        cancellation=cancellation,
        cancel_at="encoder-load",
        fail_at="cleanup",
    )

    with pytest.raises(StableAudioCancelled) as caught:
        generator.generate(
            StableAudioRequest("glass tap"),
            tmp_path / "cancelled.wav",
            cancellation=cancellation,
        )

    assert "could not release Stable Audio memory" in caught.value.__notes__[0]
    assert not (tmp_path / "cancelled.wav").exists()


def test_wav_writer_creates_stereo_44100hz_pcm(tmp_path: Path) -> None:
    output_path = tmp_path / "sound.wav"
    audio = np.zeros((STABLE_AUDIO_CHANNELS, STABLE_AUDIO_SAMPLE_RATE), dtype=np.float32)

    StableAudioGenerator._write_wav(output_path, audio)

    with wave.open(str(output_path), "rb") as generated:
        assert generated.getnchannels() == 2
        assert generated.getsampwidth() == 2
        assert generated.getframerate() == 44_100
        assert generated.getnframes() == 44_100
