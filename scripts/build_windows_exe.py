"""Build Windows executables for PLD apps with PyInstaller.

Usage:
  python scripts/build_windows_exe.py
  python scripts/build_windows_exe.py --onefile
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import subprocess
import sys

 
 
def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def pathlib_backport_present() -> bool:
    module_path = str(getattr(pathlib, "__file__", "") or "")
    normalized = module_path.lower().replace("\\", "/")
    return "site-packages" in normalized and normalized.endswith("/pathlib.py")


def pyinstaller_build(name: str, entry_file: str, onefile: bool) -> None:
    collect_packages = ["PyQt5"]
    for pkg in ("afm_tools", "afm_learn", "xrd_tools", "xrd_learn", "xrayutilities"):
        if importlib.util.find_spec(pkg) is not None:
            collect_packages.append(pkg)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--distpath",
        "dist",
        "--workpath",
        "build",
        "--specpath",
        "build/spec",
        "--paths",
        "src",
        "--name",
        name,
    ]
    for pkg in collect_packages:
        cmd.extend(["--collect-all", pkg])
    if onefile:
        cmd.append("--onefile")
    cmd.append(entry_file)
    print(f"  Collecting packages: {', '.join(collect_packages)}")
    run(cmd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onefile", action="store_true", help="Build single-file .exe outputs")
    args = parser.parse_args()

    print("[1/4] Checking Python environment...")
    if pathlib_backport_present():
        raise RuntimeError(
            "Detected obsolete 'pathlib' backport package in this Python environment.\n"
            "PyInstaller cannot run with that package installed.\n\n"
            "Fix (Conda):\n"
            "  conda remove pathlib\n\n"
            "Fix (pip environment):\n"
            "  python -m pip uninstall pathlib"
        )

    print("[2/4] Installing visualization + build dependencies...")
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", ".[visualization,build]"])
    run([sys.executable, "-m", "pip", "install", "--upgrade", "xrayutilities"])

    print("[3/4] Building PLDParameterForm...")
    pyinstaller_build("PLDParameterForm", "scripts/pyinstaller_entry_pld_form.py", args.onefile)

    print("[4/4] Building PLDRawVisualizer...")
    pyinstaller_build("PLDRawVisualizer", "scripts/pyinstaller_entry_pld_visualizer.py", args.onefile)

    print("Done. Build artifacts are in dist/:")
    if args.onefile:
        print(" - dist/PLDParameterForm.exe")
        print(" - dist/PLDRawVisualizer.exe")
    else:
        print(" - dist/PLDParameterForm/PLDParameterForm.exe")
        print(" - dist/PLDRawVisualizer/PLDRawVisualizer.exe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
