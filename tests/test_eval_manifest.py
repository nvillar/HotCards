"""Tests for the shared evaluation run manifest lifecycle."""

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

import hotcards.evaluation.manifest as manifest_module
from hotcards.evaluation.manifest import RunLifecycle, atomic_write_json, safe_run_path


def test_manifest_is_immediate_deterministic_and_finalizes_artifacts(tmp_path: Path) -> None:
    now = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)
    times = iter((10.0, 12.5))
    run_dir = tmp_path / "run"
    lifecycle = RunLifecycle.create(
        run_dir=run_dir,
        suite="fake",
        settings={"seed": 42},
        models={"ollama": "fake", "mflux": "fake"},
        contracts={"prompt": {"version": "v1", "sha256": "abc"}},
        clock=lambda: now,
        monotonic_clock=lambda: next(times),
        environment_provider=lambda: {
            "git_sha": "deadbeef",
            "python": {"version": "3.12.0"},
        },
    )

    initial = json.loads((run_dir / "manifest.json").read_text())
    assert initial["status"] == "running"
    assert initial["environment"]["git_sha"] == "deadbeef"
    raw = run_dir / "raw" / "partial-response.json"
    raw.parent.mkdir()
    raw.write_text('{"partial": true}')
    lifecycle.complete_stage("prompt")
    lifecycle.finalize(
        status="failed",
        failure={"stage": "hotspots", "classification": "timeout"},
        warnings=["geometry clamped"],
    )

    final = json.loads((run_dir / "manifest.json").read_text())
    assert final["duration_seconds"] == 2.5
    assert final["failure"]["classification"] == "timeout"
    assert final["raw_output"]["available"] is True
    assert final["partial_output"]["available"] is True
    artifact = next(
        item for item in final["artifacts"] if item["path"] == "raw/partial-response.json"
    )
    assert artifact["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert final["warnings"] == ["geometry clamped"]


def test_safe_run_path_rejects_absolute_and_traversal_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(ValueError, match="must be relative"):
        safe_run_path(run_dir, tmp_path / "outside.png")
    with pytest.raises(ValueError, match="escapes"):
        safe_run_path(run_dir, "../outside.png")


def test_atomic_json_write_preserves_previous_checkpoint_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "checkpoint.json"
    atomic_write_json(path, {"checkpoint": 1})

    def fail_replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        raise OSError(f"cannot replace {source} with {destination}")

    monkeypatch.setattr(manifest_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="cannot replace"):
        atomic_write_json(path, {"checkpoint": 2})

    assert json.loads(path.read_text()) == {"checkpoint": 1}
    assert not list(tmp_path.glob("*.tmp"))
