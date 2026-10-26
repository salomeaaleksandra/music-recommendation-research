$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$researchPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $researchPython)) {
    throw "Create .venv with Python 3.11 first: <python-3.11-path> -m venv .venv"
}
& $researchPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. Check package-index/network access." }
& $researchPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency compatibility check failed." }
& $researchPython -m ipykernel install --sys-prefix --name music-recommendation-research --display-name "Python (music recommendation research)"
if ($LASTEXITCODE -ne 0) { throw "Jupyter kernel registration failed." }
Write-Host "Ready. Launch Jupyter with: .\.venv\Scripts\python.exe -m jupyterlab"
