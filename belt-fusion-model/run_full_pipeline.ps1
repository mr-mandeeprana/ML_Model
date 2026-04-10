param(
    [switch]$UseVenv = $true,
    [string]$PythonPath = "",
    [switch]$ContinueOnError = $false
)

$ErrorActionPreference = "Continue"
# Allow native tools (python) to write stderr logs without terminating the script.
if ($null -ne (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue)) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$resultsDir = Join-Path $root "results"
if (-not (Test-Path $resultsDir)) {
    New-Item -Path $resultsDir -ItemType Directory | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $resultsDir ("pipeline_run_" + $timestamp + ".log")

function Write-Banner {
    param([string]$Text)
    $line = "=" * 78
    Write-Host "`n$line" -ForegroundColor Cyan
    Write-Host $Text -ForegroundColor Cyan
    Write-Host $line -ForegroundColor Cyan
}

function Resolve-Python {
    param([switch]$UseVenv, [string]$PythonPath)

    if ($PythonPath -and (Test-Path $PythonPath)) {
        return $PythonPath
    }

    if ($UseVenv) {
        $venvPython = Join-Path $root ".venv\Scripts\python.exe"
        if (Test-Path $venvPython) {
            return $venvPython
        }
    }

    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    throw "Python executable not found. Pass -PythonPath or create .venv."
}

function Invoke-Step {
    param(
        [string]$Title,
        [string[]]$Arguments
    )

    Write-Banner $Title
    Write-Host ("Command: " + $pythonExe + " " + ($Arguments -join " ")) -ForegroundColor DarkGray

    $argLine = ($Arguments | ForEach-Object {
        if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
    }) -join ' '
    $cmdLine = '"' + $pythonExe + '" ' + $argLine + ' 2>&1'

    cmd /c $cmdLine | Tee-Object -FilePath $logFile -Append
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Host ("Step failed with exit code: " + $exitCode) -ForegroundColor Red
        if (-not $ContinueOnError) {
            throw "Stopping pipeline due to failed step: $Title"
        }
    }
    else {
        Write-Host "Step completed successfully." -ForegroundColor Green
    }
}

$pythonExe = Resolve-Python -UseVenv:$UseVenv -PythonPath $PythonPath

Write-Banner "BELT PIPELINE ORCHESTRATOR"
Write-Host ("Workspace: " + $root)
Write-Host ("Python: " + $pythonExe)
Write-Host ("Log file: " + $logFile)

Invoke-Step -Title "STEP 1/2 - ML MODEL TRAINING AND PREDICTION" -Arguments @("-m", "ml_model.run_standalone")
Invoke-Step -Title "STEP 2/2 - FINAL ML RESULT REPORT" -Arguments @("main.py")

Write-Banner "PIPELINE COMPLETED"
Write-Host "All requested stages executed."
Write-Host ("Full log saved to: " + $logFile)
