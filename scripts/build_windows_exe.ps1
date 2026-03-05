param(
    [string]$PythonExe = "python",
    [switch]$OneFile
)

$ErrorActionPreference = "Stop"

Write-Host "[1/3] Installing project + build dependencies..."
& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install ".`[build`]"

$pyInstallerArgs = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--windowed",
    "--name", "PLDParameterForm",
    "--paths", "src",
    "src/pld_workflow/app.py"
)

if ($OneFile) {
    $pyInstallerArgs += "--onefile"
}

Write-Host "[2/3] Building executable with PyInstaller..."
& $PythonExe @pyInstallerArgs

Write-Host "[3/3] Done. Output folder: dist/PLDParameterForm"
if ($OneFile) {
    Write-Host "Single-file output: dist/PLDParameterForm.exe"
}
