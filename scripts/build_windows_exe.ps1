param(
    [string]$PythonExe = "python",
    [ValidateSet("parameter_form", "plume_manager", "xrd_visualizer", "afm_pfm_visualizer", "parameter", "plume", "visualizer", "afm_visualizer")]
    [string]$App = "parameter_form",
    [switch]$OneFile
)

$ErrorActionPreference = "Stop"

function Invoke-PythonStep {
    param(
        [string]$Description,
        [string[]]$Args
    )

    Write-Host $Description
    & $PythonExe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $PythonExe $($Args -join ' ')"
    }
}

if ($App -eq "parameter" -or $App -eq "parameter_form") {
    $exeName = "PLDParameterForm"
    $entryPoint = "scripts/pyinstaller_entry_parameter_form.py"
    $dependencies = @(
        "PyQt5>=5.15.6,<6",
        "pyinstaller>=6.0"
    )
}
elseif ($App -eq "plume" -or $App -eq "plume_manager") {
    $exeName = "PLDPlumeManager"
    $entryPoint = "scripts/pyinstaller_entry_plume_manager.py"
    $dependencies = @(
        "PyQt5>=5.15.6,<6",
        "h5py",
        "matplotlib>=3.5",
        "numpy>=1.21",
        "pyinstaller>=6.0"
    )
}
elseif ($App -eq "afm_visualizer" -or $App -eq "afm_pfm_visualizer") {
    $exeName = "PLDAFMPFMVisualizer"
    $entryPoint = "scripts/pyinstaller_entry_afm_pfm_visualizer.py"
    $dependencies = @(
        "PyQt5>=5.15.6,<6",
        "matplotlib>=3.5",
        "AFM-tools",
        "pyinstaller>=6.0"
    )
}
else {
    $exeName = "PLDXRDVisualizer"
    $entryPoint = "scripts/pyinstaller_entry_xrd_visualizer.py"
    $dependencies = @(
        "PyQt5>=5.15.6,<6",
        "matplotlib>=3.5",
        "XRD-utils",
        "xrayutilities",
        "pyinstaller>=6.0"
    )
}

$pyInstallerWorkPath = Join-Path "build" $exeName
$installArgs = @("-m", "pip", "install") + $dependencies

Invoke-PythonStep "[1/4] Upgrading pip..." @("-m", "pip", "install", "--upgrade", "pip")
Invoke-PythonStep "[2/4] Installing build/runtime dependencies for $App..." $installArgs

if (Test-Path -LiteralPath $pyInstallerWorkPath) {
    Remove-Item -LiteralPath $pyInstallerWorkPath -Recurse -Force
}

$pyInstallerArgs = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--windowed",
    "--name", $exeName,
    "--paths", "src",
    "--workpath", $pyInstallerWorkPath,
    $entryPoint
)

if ($OneFile) {
    $pyInstallerArgs += "--onefile"
}

Invoke-PythonStep "[3/4] Building executable with PyInstaller..." $pyInstallerArgs

Write-Host "[4/4] Done."
Write-Host "Output folder: dist/$exeName"
if ($OneFile) {
    Write-Host "Single-file output: dist/$exeName.exe"
}
