"""Stable Audio 3 Small-SFX generation through the official MLX runtime."""

from __future__ import annotations

import gc
import math
import secrets
import wave
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import Event, Lock
from time import monotonic

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download

from hotcards.vendor.stable_audio_3_mlx.dit_mlx import load_dit
from hotcards.vendor.stable_audio_3_mlx.sa3_pipeline import (
    apply_prompt_padding,
    build_pingpong_schedule,
    load_conditioner_from_npz,
    patched_decode,
    sample_flow_pingpong,
)
from hotcards.vendor.stable_audio_3_mlx.same_s_decoder import (
    decode_chunked,
)
from hotcards.vendor.stable_audio_3_mlx.same_s_decoder import (
    load_model as load_same_s_decoder,
)
from hotcards.vendor.stable_audio_3_mlx.t5gemma_mlx import T5Gemma

STABLE_AUDIO_REPOSITORY = "stabilityai/stable-audio-3-optimized"
STABLE_AUDIO_MODEL = "stabilityai/stable-audio-3-small-sfx"
STABLE_AUDIO_RUNTIME = "stable-audio-3-optimized-mlx"
STABLE_AUDIO_SAMPLER = "pingpong"
STABLE_AUDIO_STEPS = 8
STABLE_AUDIO_CFG = 1.0
STABLE_AUDIO_SAMPLE_RATE = 44_100
STABLE_AUDIO_CHANNELS = 2
STABLE_AUDIO_SAMPLE_WIDTH = 2
_SAMPLES_PER_LATENT = 4_096
_WEIGHT_FILENAMES = (
    "MLX/t5gemma_f16.npz",
    "MLX/dit_sm-sfx_f16.npz",
    "MLX/same_s_decoder_f32.npz",
)


class StableAudioFailureKind(StrEnum):
    """User-facing Stable Audio failure categories."""

    MODEL_UNAVAILABLE = "model-unavailable"
    MODEL_LOAD = "model-load"
    GENERATION = "generation"
    OUTPUT = "output"
    CANCELLED = "cancelled"


class StableAudioError(RuntimeError):
    """A Stable Audio operation failed in a known stage."""

    def __init__(
        self,
        message: str,
        *,
        kind: StableAudioFailureKind,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.__cause__ = cause


class StableAudioCancelled(StableAudioError):
    """Stable Audio work was cancelled cooperatively."""

    def __init__(self) -> None:
        super().__init__(
            "sound generation was cancelled",
            kind=StableAudioFailureKind.CANCELLED,
        )


@dataclass(frozen=True, slots=True)
class StableAudioRequest:
    """Validated inputs for one Small-SFX generation."""

    prompt: str
    duration_seconds: int = 2
    seed: int | None = None

    def __post_init__(self) -> None:
        prompt = self.prompt.strip()
        if not prompt:
            raise ValueError("sound prompt must not be empty")
        if not 1 <= self.duration_seconds <= 30:
            raise ValueError("sound duration must be between 1 and 30 seconds")
        if self.seed is not None and not 0 <= self.seed <= (2**32 - 1):
            raise ValueError("sound seed must be between 0 and 4294967295")
        object.__setattr__(self, "prompt", prompt)


@dataclass(frozen=True, slots=True)
class StableAudioResult:
    """One generated temporary WAV and its exact runtime metadata."""

    output_path: Path
    seed: int
    generation_duration_milliseconds: int


class StableAudioCancellation:
    """Thread-safe cooperative cancellation shared with the adapter worker."""

    def __init__(self) -> None:
        self._event = Event()

    def request(self) -> None:
        self._event.set()

    def raise_if_requested(self) -> None:
        if self._event.is_set():
            raise StableAudioCancelled


class StableAudioGenerator:
    """Generate 16-bit stereo WAV files with Stable Audio 3 Small-SFX."""

    def __init__(
        self,
        *,
        repository: str = STABLE_AUDIO_REPOSITORY,
        download: Callable[..., str] = hf_hub_download,
    ) -> None:
        self._repository = repository
        self._download = download
        self._active_lock = Lock()
        self._active_cancellation: StableAudioCancellation | None = None

    def check_available(self) -> None:
        """Require all model weights to be present in the local Hugging Face cache."""
        self._resolve_weights(local_files_only=True)

    def cancel(self) -> None:
        """Request cancellation of the active generation, if any."""
        with self._active_lock:
            cancellation = self._active_cancellation
        if cancellation is not None:
            cancellation.request()

    def generate(
        self,
        request: StableAudioRequest,
        output_path: Path,
        *,
        cancellation: StableAudioCancellation | None = None,
        on_sampling_step: Callable[[int, int], None] | None = None,
    ) -> StableAudioResult:
        """Generate one exact-duration WAV into a caller-owned temporary path."""
        cancellation = cancellation or StableAudioCancellation()
        with self._active_lock:
            if self._active_cancellation is not None:
                raise RuntimeError("Stable Audio generation is already active")
            self._active_cancellation = cancellation
        started = monotonic()
        try:
            weights = self._resolve_weights(local_files_only=False)
            cancellation.raise_if_requested()
            audio, seed = self._generate_audio(
                request,
                weights=weights,
                cancellation=cancellation,
                on_sampling_step=on_sampling_step,
            )
            cancellation.raise_if_requested()
            self._write_wav(output_path, audio)
            return StableAudioResult(
                output_path=output_path,
                seed=seed,
                generation_duration_milliseconds=round((monotonic() - started) * 1_000),
            )
        except StableAudioError:
            raise
        except OSError as error:
            raise StableAudioError(
                f"could not write generated sound: {error}",
                kind=StableAudioFailureKind.OUTPUT,
                cause=error,
            ) from error
        finally:
            with self._active_lock:
                if self._active_cancellation is cancellation:
                    self._active_cancellation = None

    def _resolve_weights(self, *, local_files_only: bool) -> dict[str, Path]:
        resolved: dict[str, Path] = {}
        try:
            for filename in _WEIGHT_FILENAMES:
                resolved[filename] = Path(
                    self._download(
                        repo_id=self._repository,
                        filename=filename,
                        local_files_only=local_files_only,
                    )
                )
        except Exception as error:
            raise StableAudioError(
                "Stable Audio 3 Small-SFX weights are not available",
                kind=StableAudioFailureKind.MODEL_UNAVAILABLE,
                cause=error,
            ) from error
        return resolved

    @staticmethod
    def _generate_audio(
        request: StableAudioRequest,
        *,
        weights: dict[str, Path],
        cancellation: StableAudioCancellation,
        on_sampling_step: Callable[[int, int], None] | None,
    ) -> tuple[np.ndarray, int]:
        dtype = mx.float16
        duration = float(request.duration_seconds)
        latent_length = max(
            1,
            math.ceil(duration * STABLE_AUDIO_SAMPLE_RATE / _SAMPLES_PER_LATENT),
        )
        try:
            encoder = T5Gemma.from_npz(str(weights["MLX/t5gemma_f16.npz"]))
            embeddings, mask = encoder.encode([request.prompt], max_len=256)
            mx.eval(embeddings, mask)
            cancellation.raise_if_requested()

            padding, seconds_embedder = load_conditioner_from_npz(
                str(weights["MLX/dit_sm-sfx_f16.npz"]),
                prefix="cond.",
            )
            padded = apply_prompt_padding(
                embeddings.astype(dtype),
                mask,
                padding.astype(dtype),
            )
            seconds = seconds_embedder(duration).astype(dtype)
            cross_attention = mx.concatenate([padded, seconds], axis=1)
            global_conditioning = seconds[:, 0, :]
            mx.eval(cross_attention, global_conditioning)
            del encoder, embeddings, mask, padding, padded, seconds, seconds_embedder
            _free_mlx_memory()
            cancellation.raise_if_requested()

            model = load_dit(
                str(weights["MLX/dit_sm-sfx_f16.npz"]),
                T_lat=latent_length,
                dtype=dtype,
                compile_=False,
                num_steps=STABLE_AUDIO_STEPS,
            )
            seed = request.seed if request.seed is not None else secrets.randbits(32)
            noise = mx.random.normal(
                (1, 256, latent_length),
                dtype=dtype,
                key=mx.random.key(seed),
            )
            mx.eval(noise)
            sigmas = build_pingpong_schedule(
                STABLE_AUDIO_STEPS,
                sigma_max=1.0,
                use_logsnr_shift=True,
            )

            def sampling_step(step: int, total: int) -> None:
                cancellation.raise_if_requested()
                if on_sampling_step is not None:
                    on_sampling_step(step, total)

            latents = _sample_latents(
                model,
                cross_attention,
                global_conditioning,
                noise,
                sigmas,
                seed=seed + 1,
                on_step=sampling_step,
            )
            mx.eval(latents)
            del model, noise, sigmas, cross_attention, global_conditioning
            _free_mlx_memory()
            cancellation.raise_if_requested()

            decoder = load_same_s_decoder(
                str(weights["MLX/same_s_decoder_f32.npz"]),
                dtype=mx.float32,
                compile_=False,
            )
            latents = latents.astype(mx.float32)
            if latent_length > 12:
                patches = decode_chunked(decoder, latents, 8, 2)
            elif latent_length % 2 == 0:
                patches = decoder(latents)
            elif latent_length > 6:
                patches = decode_chunked(decoder, latents, 2, 2)
            else:
                even_latents = mx.concatenate([latents, latents[..., -1:]], axis=-1)
                patches = decoder(even_latents)[..., : latent_length * 16]
            mx.eval(patches)
            cancellation.raise_if_requested()

            audio = patched_decode(patches, patch_size=256, channels=STABLE_AUDIO_CHANNELS)
            mx.eval(audio)
            audio_array = np.array(audio.astype(mx.float32))[0]
            sample_count = round(duration * STABLE_AUDIO_SAMPLE_RATE)
            return audio_array[..., :sample_count], seed
        except StableAudioError:
            raise
        except Exception as error:
            kind = (
                StableAudioFailureKind.MODEL_LOAD
                if "model" not in locals()
                else StableAudioFailureKind.GENERATION
            )
            raise StableAudioError(
                f"Stable Audio generation failed: {error}",
                kind=kind,
                cause=error,
            ) from error
        finally:
            _free_mlx_memory()

    @staticmethod
    def _write_wav(path: Path, audio: np.ndarray) -> None:
        if audio.shape[0] != STABLE_AUDIO_CHANNELS:
            raise StableAudioError(
                f"generated sound has {audio.shape[0]} channels; expected stereo",
                kind=StableAudioFailureKind.OUTPUT,
            )
        if not np.isfinite(audio).all():
            raise StableAudioError(
                "generated sound contains non-finite samples",
                kind=StableAudioFailureKind.OUTPUT,
            )
        pcm = (np.clip(audio, -1.0, 1.0) * 32_767.0).astype(np.int16).T
        with wave.open(str(path), "wb") as output:
            output.setnchannels(STABLE_AUDIO_CHANNELS)
            output.setsampwidth(STABLE_AUDIO_SAMPLE_WIDTH)
            output.setframerate(STABLE_AUDIO_SAMPLE_RATE)
            output.writeframes(pcm.tobytes())


def _free_mlx_memory() -> None:
    gc.collect()
    clear_cache = getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", None)
    if clear_cache is not None:
        clear_cache()


def _sample_latents(
    model: object,
    cross_attention: mx.array,
    global_conditioning: mx.array,
    noise: mx.array,
    sigmas: mx.array,
    *,
    seed: int,
    on_step: Callable[[int, int], None],
) -> mx.array:
    def model_function(latents: mx.array, timestep: mx.array) -> mx.array:
        return model(
            latents,
            timestep,
            cross_attention,
            global_conditioning,
            local_add_cond=None,
        )

    return sample_flow_pingpong(
        model_function,
        noise,
        sigmas,
        seed=seed,
        on_step=on_step,
    )
