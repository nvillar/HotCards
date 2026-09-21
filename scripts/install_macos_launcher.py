"""Install a local macOS launcher with a worktree-independent source snapshot."""

from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
BUNDLE_IDENTIFIER = "io.github.nvillar.HotCards.launcher"
ICON_SCRIPT = """
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from hotcards.ui.branding import application_icon_pixmap

app = QApplication([])
destination = Path(sys.argv[1])
for size in (16, 32, 128, 256, 512):
    for scale in (1, 2):
        suffix = "@2x" if scale == 2 else ""
        path = destination / f"icon_{size}x{size}{suffix}.png"
        if not application_icon_pixmap(size * scale).save(str(path)):
            raise RuntimeError(f"Could not save icon: {path}")
"""


def snapshot_source(source: Path, destination: Path) -> None:
    """Copy only installation inputs, never credentials, caches, or stack data."""
    for name in ("pyproject.toml", "uv.lock", ".python-version", "LICENSE"):
        shutil.copy2(source / name, destination / name)
    shutil.copytree(
        source / "src",
        destination / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def install_launcher(
    source: Path,
    app_path: Path,
    runtime_root: Path,
    *,
    uv: str,
) -> Path:
    app_path = app_path.expanduser().absolute()
    runtime_root = runtime_root.expanduser().absolute()
    if app_path.suffix != ".app":
        raise ValueError("The launcher destination must end in .app.")
    if os.path.lexists(app_path):
        raise FileExistsError(
            f"{app_path} already exists. Move it aside before installing a new launcher."
        )
    app_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)
    # Create the environment at its permanent path: Python entry points are not relocatable.
    runtime = Path(tempfile.mkdtemp(prefix="installation-", dir=runtime_root))
    published = False
    try:
        snapshot_source(source, runtime)
        environment = {
            key: value
            for key, value in os.environ.items()
            if key
            not in (
                "UV_PROJECT_ENVIRONMENT",
                "VIRTUAL_ENV",
                "PYTHONHOME",
                "PYTHONPATH",
                "__PYVENV_LAUNCHER__",
            )
        }
        subprocess.run(
            [uv, "sync", "--project", str(runtime), "--locked", "--no-dev", "--no-editable"],
            check=True,
            env=environment,
        )
        with tempfile.TemporaryDirectory(prefix=".hotcards-build-", dir=app_path.parent) as build:
            build_path = Path(build)
            bundle = build_path / app_path.name
            contents = bundle / "Contents"
            resources = contents / "Resources"
            macos = contents / "MacOS"
            resources.mkdir(parents=True)
            macos.mkdir()
            iconset = build_path / "HotCards.iconset"
            iconset.mkdir()
            subprocess.run(
                [str(runtime / ".venv/bin/python"), "-c", ICON_SCRIPT, str(iconset)],
                check=True,
                env={**environment, "QT_QPA_PLATFORM": "offscreen"},
            )
            subprocess.run(
                [
                    "/usr/bin/iconutil",
                    "-c",
                    "icns",
                    str(iconset),
                    "-o",
                    str(resources / "HotCards.icns"),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "/usr/bin/xcrun",
                    "swiftc",
                    str(source / "scripts/macos/HotCards.swift"),
                    "-o",
                    str(macos / "HotCards"),
                ],
                check=True,
            )
            with (contents / "Info.plist").open("wb") as stream:
                plistlib.dump(
                    {
                        "CFBundleIdentifier": BUNDLE_IDENTIFIER,
                        "CFBundleName": "HotCards",
                        "CFBundleDisplayName": "HotCards",
                        "CFBundleExecutable": "HotCards",
                        "CFBundleIconFile": "HotCards.icns",
                        "CFBundlePackageType": "APPL",
                        "CFBundleVersion": "1",
                        "CFBundleShortVersionString": "0.1.0",
                        "NSHighResolutionCapable": True,
                        # The Qt child owns the visible app; the supervisor has no second Dock tile.
                        "LSUIElement": True,
                    },
                    stream,
                )
            with (resources / "Launcher.plist").open("wb") as stream:
                plistlib.dump({"RuntimeDirectory": str(runtime)}, stream)
            subprocess.run(
                ["/usr/bin/codesign", "--force", "--sign", "-", str(bundle)],
                check=True,
            )
            subprocess.run([str(macos / "HotCards"), "--check"], check=True)
            # Do not overwrite an app created while dependencies were being installed.
            if os.path.lexists(app_path):
                raise FileExistsError(f"{app_path} appeared during installation.")
            bundle.rename(app_path)
            published = True
    finally:
        if not published:
            shutil.rmtree(runtime)
    return app_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-path", type=Path, default=Path.home() / "Applications/HotCards.app")
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path.home() / "Library/Application Support/HotCards/Launcher",
    )
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("The launcher requires macOS.")
    uv = shutil.which("uv")
    if uv is None:
        parser.error("Install uv and make it available on PATH before building the launcher.")
    try:
        app_path = install_launcher(REPOSITORY, args.app_path, args.runtime_root, uv=uv)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Launcher installation failed: {error}", file=sys.stderr)
        return 1
    print(f"Installed {app_path}\nOpen it in Finder, or drag it to the Dock.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
