"""Local launcher installation tests; no model imports or inference."""

import plistlib
import runpy
import subprocess
from pathlib import Path

import pytest

INSTALLER = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/install_macos_launcher.py")
)
install_launcher = INSTALLER["install_launcher"]
snapshot_source = INSTALLER["snapshot_source"]
ICON_SCRIPT = INSTALLER["ICON_SCRIPT"]


@pytest.fixture
def source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("pyproject.toml", "uv.lock", ".python-version", "LICENSE"):
        (source / name).write_text(name)
    package = source / "src/hotcards"
    package.mkdir(parents=True)
    (package / "main.py").write_text("original")
    (package / "__pycache__").mkdir()
    (package / "__pycache__/main.pyc").write_bytes(b"cache")
    (source / ".env").write_text("do not copy")
    return source


def test_snapshot_is_independent_and_contains_only_installation_inputs(source, tmp_path):
    destination = tmp_path / "snapshot"
    destination.mkdir()
    snapshot_source(source, destination)
    (source / "src/hotcards/main.py").write_text("changed")
    assert (destination / "src/hotcards/main.py").read_text() == "original"
    assert not (destination / ".env").exists()
    assert not (destination / "src/hotcards/__pycache__").exists()
    assert (destination / "uv.lock").read_text() == "uv.lock"


def test_install_publishes_app_with_permanent_runtime(source, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/unrelated/environment")
    monkeypatch.setenv("PYTHONPATH", "/unrelated/packages")

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["check"]
        if command[0] == "/tools/uv" or "-c" in command and ICON_SCRIPT in command:
            assert "UV_PROJECT_ENVIRONMENT" not in kwargs["env"]
            assert "PYTHONPATH" not in kwargs["env"]

    monkeypatch.setattr(subprocess, "run", run)
    app = tmp_path / "Apps with spaces/HotCards.app"
    runtime_root = tmp_path / "Application Support/HotCards"
    assert install_launcher(source, app, runtime_root, uv="/tools/uv") == app
    configuration = plistlib.loads((app / "Contents/Resources/Launcher.plist").read_bytes())
    runtime = Path(configuration["RuntimeDirectory"])
    assert runtime.parent == runtime_root
    assert (runtime / "src/hotcards/main.py").read_text() == "original"
    assert calls[0] == [
        "/tools/uv",
        "sync",
        "--project",
        str(runtime),
        "--locked",
        "--no-dev",
        "--no-editable",
    ]
    info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    assert info["CFBundleExecutable"] == "HotCards"
    assert info["CFBundleIconFile"] == "HotCards.icns"
    assert info["LSUIElement"] is True
    assert calls[-1][-1] == "--check"
    assert list(app.parent.iterdir()) == [app]


@pytest.mark.parametrize("failure_call", [1, 3, 6])
def test_failed_install_removes_only_its_own_runtime(source, tmp_path, monkeypatch, failure_call):
    runtime_root = tmp_path / "runtimes"
    retained = runtime_root / "retained"
    retained.mkdir(parents=True)
    app = tmp_path / "Applications/HotCards.app"
    calls = 0

    def fail(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        install_launcher(source, app, runtime_root, uv="/tools/uv")
    assert list(runtime_root.iterdir()) == [retained]
    assert not app.exists()
    assert list(app.parent.iterdir()) == []


@pytest.mark.parametrize("symlink", [False, True])
def test_existing_app_is_not_overwritten(source, tmp_path, symlink):
    app = tmp_path / "HotCards.app"
    if symlink:
        app.symlink_to(tmp_path / "missing", target_is_directory=True)
    else:
        app.mkdir()
    runtime_root = tmp_path / "runtimes"
    with pytest.raises(FileExistsError):
        install_launcher(source, app, runtime_root, uv="/tools/uv")
    assert not runtime_root.exists()
