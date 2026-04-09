"""Build one of the desktop apps into a Windows executable with PyInstaller."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


APP_CONFIG = {
    "parameter_form": {
        "exe_name": "PLDParameterForm",
        "entry_point": "scripts/pyinstaller_entry_parameter_form.py",
        "dependencies": [
            "PyQt5>=5.15.6,<6",
            "pyinstaller>=6.0",
        ],
        "pyinstaller_args": [],
    },
    "plume_manager": {
        "exe_name": "PLDPlumeManager",
        "entry_point": "scripts/pyinstaller_entry_plume_manager.py",
        "dependencies": [
            "PyQt5>=5.15.6,<6",
            "h5py",
            "matplotlib>=3.5",
            "numpy>=1.21",
            "pyinstaller>=6.0",
        ],
        "pyinstaller_args": [],
    },
    "xrd_visualizer": {
        "exe_name": "PLDXRDVisualizer",
        "entry_point": "scripts/pyinstaller_entry_xrd_visualizer.py",
        "dependencies": [
            "PyQt5>=5.15.6,<6",
            "matplotlib>=3.5",
            "XRD-utils",
            "xrayutilities",
            "pyinstaller>=6.0",
        ],
        "pyinstaller_args": [
            "--collect-all",
            "xrd_utils",
            "--collect-all",
            "xrayutilities",
            "--hidden-import",
            "sip",
        ],
    },
    "afm_pfm_visualizer": {
        "exe_name": "PLDAFMPFMVisualizer",
        "entry_point": "scripts/pyinstaller_entry_afm_pfm_visualizer.py",
        "dependencies": [
            "PyQt5>=5.15.6,<6",
            "matplotlib>=3.5",
            "AFM-tools",
            "pyinstaller>=6.0",
        ],
        "pyinstaller_args": [
            "--collect-all",
            "afm_tools",
            "--hidden-import",
            "sip",
        ],
    },
    "rheed_imm_visualizer": {
        "exe_name": "PLDRHEEDIMMVisualizer",
        "entry_point": "scripts/pyinstaller_entry_rheed_imm_visualizer.py",
        "dependencies": [
            "PyQt5>=5.15.6,<6",
            "numpy>=1.21",
            "pyinstaller>=6.0",
        ],
        "pyinstaller_args": [],
    },
}

APP_ALIASES = {
    "parameter": "parameter_form",
    "plume": "plume_manager",
    "visualizer": "xrd_visualizer",
    "afm_visualizer": "afm_pfm_visualizer",
}


def run_step(description: str, args: list[str], cwd: Path) -> None:
    """Run one subprocess step and stop immediately on failure."""
    print(description)
    subprocess.run(args, check=True, cwd=str(cwd))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the build helper."""
    parser = argparse.ArgumentParser(description=__doc__)
    valid_choices = sorted(set(APP_CONFIG) | set(APP_ALIASES))
    parser.add_argument(
        "--app",
        choices=valid_choices,
        default="parameter_form",
        help="App target to build.",
    )
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="Build a single-file executable.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for pip and PyInstaller.",
    )
    return parser.parse_args()


def ensure_clean_output(path: Path) -> None:
    """Remove an old output file or directory when it is not locked."""
    if not path.exists():
        return

    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError as exc:
        raise RuntimeError(
            f"Could not remove old output '{path}'. Close any running PLD app .exe, "
            "close Explorer windows opened inside dist/, and pause Dropbox sync, then retry."
        ) from exc


def clean_legacy_repo_artifacts(repo_root: Path, exe_name: str, onefile: bool) -> None:
    """Delete stale repo-local artifacts that are safe to remove before copying the new build."""
    dist_dir = repo_root / "dist"
    dist_dir.mkdir(exist_ok=True)

    targets = [dist_dir / exe_name]
    if onefile:
        targets.append(dist_dir / f"{exe_name}.exe")

    for target in targets:
        ensure_clean_output(target)


def main() -> int:
    """Install required dependencies and run PyInstaller."""
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    canonical_app = APP_ALIASES.get(args.app, args.app)
    config = APP_CONFIG[canonical_app]
    exe_name = config["exe_name"]
    entry_point = repo_root / config["entry_point"]

    run_step(
        "[1/5] Upgrading pip...",
        [args.python, "-m", "pip", "install", "--upgrade", "pip"],
        cwd=repo_root,
    )
    run_step(
        f"[2/5] Installing build/runtime dependencies for {canonical_app}...",
        [args.python, "-m", "pip", "install", *config["dependencies"]],
        cwd=repo_root,
    )

    clean_legacy_repo_artifacts(repo_root, exe_name, args.onefile)

    with tempfile.TemporaryDirectory(prefix=f"{exe_name}_pyinstaller_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        work_path = temp_dir / "work"
        dist_path = temp_dir / "dist"
        spec_path = temp_dir / "spec"

        pyinstaller_args = [
            args.python,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--windowed",
            "--name",
            exe_name,
            "--paths",
            str(repo_root / "src"),
            "--workpath",
            str(work_path),
            "--distpath",
            str(dist_path),
            "--specpath",
            str(spec_path),
            str(entry_point),
        ]
        pyinstaller_args.extend(config.get("pyinstaller_args", []))
        if args.onefile:
            pyinstaller_args.append("--onefile")

        run_step("[3/5] Building executable with PyInstaller...", pyinstaller_args, cwd=repo_root)

        repo_dist = repo_root / "dist"
        repo_dist.mkdir(exist_ok=True)

        if args.onefile:
            built_output = dist_path / f"{exe_name}.exe"
            final_output = repo_dist / f"{exe_name}.exe"
        else:
            built_output = dist_path / exe_name
            final_output = repo_dist / exe_name

        if not built_output.exists():
            raise RuntimeError(f"PyInstaller completed but expected output was not found: {built_output}")

        print("[4/5] Copying the fresh build back into dist/...")
        if built_output.is_dir():
            shutil.copytree(built_output, final_output)
        else:
            shutil.copy2(built_output, final_output)

    print("[5/5] Done.")
    if args.onefile:
        print(f"Single-file output: dist/{exe_name}.exe")
    else:
        print(f"Output folder: dist/{exe_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
