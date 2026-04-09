"""Remove old build/dist artifacts when they are not locked by another process."""

from __future__ import annotations

import shutil
from pathlib import Path


def remove_path(path: Path) -> tuple[bool, str | None]:
    """Try to remove a file or directory and report lock failures cleanly."""
    if not path.exists():
        return True, None

    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True, None
    except OSError as exc:
        return False, str(exc)


def main() -> int:
    """Clean common build artifacts in the repository."""
    repo_root = Path(__file__).resolve().parents[1]
    targets = [
        repo_root / "build",
        repo_root / "dist" / "PLDParameterForm",
        repo_root / "dist" / "PLDPlumeManager",
        repo_root / "dist" / "PLDXRDVisualizer",
        repo_root / "dist" / "PLDAFMPFMVisualizer",
        repo_root / "dist" / "PLDRHEEDIMMVisualizer",
        repo_root / "dist" / "PLDRawVisualizer",
        repo_root / "dist" / "PLDAFMVisualizer",
        repo_root / "dist" / "PLDParameterForm.exe",
        repo_root / "dist" / "PLDPlumeManager.exe",
        repo_root / "dist" / "PLDXRDVisualizer.exe",
        repo_root / "dist" / "PLDAFMPFMVisualizer.exe",
        repo_root / "dist" / "PLDRHEEDIMMVisualizer.exe",
        repo_root / "dist" / "PLDRawVisualizer.exe",
        repo_root / "dist" / "PLDAFMVisualizer.exe",
    ]

    failed: list[tuple[Path, str]] = []
    for target in targets:
        removed, error = remove_path(target)
        if removed:
            print(f"removed: {target.relative_to(repo_root)}")
        elif error is not None:
            failed.append((target, error))
            print(f"locked:  {target.relative_to(repo_root)}")

    (repo_root / "dist").mkdir(exist_ok=True)

    if failed:
        print("\nSome artifacts could not be removed because they are in use.")
        print("Close any running PLD .exe, close Explorer windows opened inside build/dist,")
        print("and pause Dropbox sync, then run this cleanup script again.")
        for target, error in failed:
            print(f"- {target.relative_to(repo_root)}: {error}")
        return 1

    print("\nBuild artifacts are clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
