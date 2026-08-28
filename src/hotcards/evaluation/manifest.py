"""Immutable evaluation run-directory lifecycle and manifest contract."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import monotonic
from typing import Any

import httpx

from hotcards.generation.errors import (
    ImageGenerationError,
    ModelLoadError,
    ModelResponseError,
    ModelUnavailableError,
    ServiceUnavailableError,
)

MANIFEST_VERSION = "eval-run-manifest-v1"
EnvironmentProvider = Callable[[], Mapping[str, Any]]
Clock = Callable[[], datetime]
MonotonicClock = Callable[[], float]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _command_version(command: list[str]) -> str:
    try:
        completed = subprocess.run(  # noqa: S603
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return (completed.stdout or completed.stderr).strip() or "unknown"


def _memory_bytes() -> int | None:
    if platform.system() != "Darwin":
        return None
    try:
        return int(_command_version(["sysctl", "-n", "hw.memsize"]))
    except ValueError:
        return None


def default_environment() -> Mapping[str, Any]:
    """Capture local software and hardware facts without contacting models."""
    try:
        git_sha = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git_sha = "unknown"
    dependencies = {
        name: _package_version(name)
        for name in (
            "httpx",
            "hotcards",
            "mflux",
            "mlx",
            "ollama",
            "pillow",
            "pydantic",
            "pyside6",
        )
    }
    return {
        "git_sha": git_sha,
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "memory_bytes": _memory_bytes(),
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "dependencies": dependencies,
        "ollama": {
            "python_package": dependencies["ollama"],
            "cli": _command_version(["ollama", "--version"]),
        },
        "mflux": {
            "package": dependencies["mflux"],
            "mlx": dependencies["mlx"],
        },
    }


def contract_digest(*values: object) -> str:
    """Hash a prompt/schema contract using canonical JSON."""
    encoded = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    """Durably replace a JSON checkpoint without truncating its predecessor."""
    payload = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def classify_failure(error: BaseException) -> str:
    """Map adapter failures to stable report classifications."""
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, (TimeoutError, httpx.TimeoutException)):
            return "timeout"
        cause = cause.__cause__
    if isinstance(error, ModelResponseError):
        return "structured_output_validation"
    if isinstance(error, (ServiceUnavailableError, ModelUnavailableError)):
        return "transport_or_service"
    if isinstance(error, (ImageGenerationError, ModelLoadError)):
        return "image_generation"
    return "unexpected"


def safe_run_path(run_dir: Path, relative_path: str | Path) -> Path:
    """Resolve a result artifact path and reject traversal outside its run."""
    value = Path(relative_path)
    if value.is_absolute():
        raise ValueError(f"artifact path must be relative to the run directory: {value}")
    root = run_dir.resolve()
    resolved = (run_dir / value).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"artifact path escapes run directory: {value}") from error
    return resolved


class RunLifecycle:
    """Own one run directory from initial manifest through final checksums."""

    def __init__(
        self,
        run_dir: Path,
        manifest: dict[str, Any],
        *,
        clock: Clock,
        monotonic_clock: MonotonicClock,
        started_monotonic: float,
    ) -> None:
        self.run_dir = run_dir
        self.manifest = manifest
        self._clock = clock
        self._monotonic_clock = monotonic_clock
        self._started_monotonic = started_monotonic
        self._finalized = False

    @classmethod
    def create(
        cls,
        *,
        run_dir: Path,
        suite: str,
        settings: Mapping[str, Any],
        models: Mapping[str, object],
        contracts: Mapping[str, object],
        clock: Clock = _utc_now,
        monotonic_clock: MonotonicClock = monotonic,
        environment_provider: EnvironmentProvider = default_environment,
    ) -> RunLifecycle:
        """Create a unique run directory and immediately persist its manifest."""
        run_dir.mkdir(parents=True, exist_ok=False)
        started = clock()
        started_monotonic = monotonic_clock()
        manifest: dict[str, Any] = {
            "manifest_version": MANIFEST_VERSION,
            "suite": suite,
            "run_id": run_dir.name,
            "status": "running",
            "stage": "initializing",
            "failure": None,
            "started_at": started.isoformat(),
            "completed_at": None,
            "duration_seconds": None,
            "environment": {"status": "capture_pending"},
            "models": dict(models),
            "contracts": dict(contracts),
            "settings": dict(settings),
            "completed_stages": [],
            "artifacts": [],
            "raw_output": {"available": False, "paths": []},
            "partial_output": {"available": False, "paths": []},
            "warnings": [],
        }
        lifecycle = cls(
            run_dir,
            manifest,
            clock=clock,
            monotonic_clock=monotonic_clock,
            started_monotonic=started_monotonic,
        )
        lifecycle._write()
        try:
            lifecycle.manifest["environment"] = dict(environment_provider())
        except Exception as error:
            lifecycle.manifest["environment"] = {
                "status": "capture_failed",
                "error_type": type(error).__name__,
                "message": str(error),
            }
            lifecycle.add_warning(f"environment capture failed: {error}")
        lifecycle._write()
        return lifecycle

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def _write(self) -> None:
        atomic_write_json(self.manifest_path, self.manifest)

    def set_stage(self, stage: str) -> None:
        """Persist the currently executing stage."""
        if self._finalized:
            raise RuntimeError("run manifest is already finalized")
        self.manifest["stage"] = stage
        self._write()

    def complete_stage(self, stage: str) -> None:
        """Persist successful completion of a stage."""
        completed: list[str] = self.manifest["completed_stages"]
        if stage not in completed:
            completed.append(stage)
        self.manifest["stage"] = stage
        self._write()

    def update_contract(self, name: str, contract: Mapping[str, object]) -> None:
        """Persist a contract discovered after the immediate manifest write."""
        if self._finalized:
            raise RuntimeError("run manifest is already finalized")
        self.manifest["contracts"][name] = dict(contract)
        self._write()

    def add_warning(self, warning: str) -> None:
        warnings: list[str] = self.manifest["warnings"]
        if warning not in warnings:
            warnings.append(warning)

    def finalize(
        self,
        *,
        status: str,
        failure: Mapping[str, object] | None = None,
        warnings: list[str] | tuple[str, ...] = (),
    ) -> Path:
        """Finalize once with complete artifact inventory and checksums."""
        if self._finalized:
            raise RuntimeError("run manifest is already finalized")
        for warning in warnings:
            self.add_warning(warning)
        artifacts: list[dict[str, object]] = []
        raw_paths: list[str] = []
        partial_paths: list[str] = []
        for path in sorted(self.run_dir.rglob("*")):
            if not path.is_file() or path == self.manifest_path:
                continue
            relative = path.relative_to(self.run_dir).as_posix()
            artifacts.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            if relative.startswith("raw/"):
                raw_paths.append(relative)
                if "partial" in path.name:
                    partial_paths.append(relative)
        self.manifest.update(
            {
                "status": status,
                "completed_at": self._clock().isoformat(),
                "duration_seconds": self._monotonic_clock() - self._started_monotonic,
                "failure": dict(failure) if failure else None,
                "artifacts": artifacts,
                "raw_output": {"available": bool(raw_paths), "paths": raw_paths},
                "partial_output": {
                    "available": bool(partial_paths),
                    "paths": partial_paths,
                },
                "warnings": sorted(self.manifest["warnings"]),
            }
        )
        self._write()
        self._finalized = True
        return self.manifest_path
