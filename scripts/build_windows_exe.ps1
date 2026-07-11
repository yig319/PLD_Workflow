param(
    [string]$PythonExe = "python",
    [ValidateSet(
        "parameter_form",
        "plume_manager",
        "xrd_visualizer",
        "afm_pfm_visualizer",
        "rheed_imm_visualizer",
        "parameter",
        "plume",
        "visualizer",
        "afm_visualizer"
    )]
    [string]$App = "parameter_form",
    [switch]$OneFile
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonScript = Join-Path $scriptDir "build_windows_exe.py"

$args = @($pythonScript, "--app", $App, "--python", $PythonExe)
if ($OneFile) {
    $args += "--onefile"
}

Write-Host "Running Python build helper..."
& $PythonExe @args
if ($LASTEXITCODE -ne 0) {
    throw "Build failed: $PythonExe $($args -join ' ')"
}
