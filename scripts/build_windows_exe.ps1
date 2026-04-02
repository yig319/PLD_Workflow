param(
    [string]$PythonExe = "python",
    [ValidateSet("parameter", "plume", "visualizer")]
    [string]$App = "parameter",
    [switch]$OneFile
)

$ErrorActionPreference = "Stop"

if ($App -eq "parameter") {
    $exeName = "PLDParameterForm"
    $entryPoint = "src/pld_workflow/app.py"
}
elseif ($App -eq "plume") {
    $exeName = "PLDPlumeManager"
    $entryPoint = "src/pld_workflow/plume_app.py"
}
else {
    $exeName = "PLDRawVisualizer"
    $entryPoint = "src/pld_workflow/visualizer_app.py"
}

Write-Host "[1/3] Installing project + build dependencies for $App..."
& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install ".`[build,analysis,visualization`]"

$pyInstallerArgs = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--windowed",
    "--name", $exeName,
    "--paths", "src",
    $entryPoint
)

if ($OneFile) {
    $pyInstallerArgs += "--onefile"
}

Write-Host "[2/3] Building executable with PyInstaller..."
& $PythonExe @pyInstallerArgs

Write-Host "[3/3] Done. Output folder: dist/$exeName"
if ($OneFile) {
    Write-Host "Single-file output: dist/$exeName.exe"
}
