$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskTemp = Join-Path $taskRoot 'work\tmp'
New-Item -ItemType Directory -Force -Path $taskTemp | Out-Null
$env:TEMP = $taskTemp
$env:TMP = $taskTemp
$env:PIP_CACHE_DIR = Join-Path $taskRoot 'work\pip-cache'
$env:HF_HOME = Join-Path $taskRoot 'work\hf-cache'
$env:TORCH_HOME = Join-Path $taskRoot 'work\torch-cache'
$env:CUDA_CACHE_PATH = Join-Path $taskRoot 'work\cuda-cache'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
Push-Location $taskRoot
try {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create D-drive environment' }
    $taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
    $taskWheel = Join-Path $taskRoot 'work\torch-2.11.0+cu128-cp313-cp313-win_amd64.whl'
    if (Test-Path -LiteralPath $taskWheel) {
        & $taskPython -m pip install $taskWheel
    } else {
        & $taskPython -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128
    }
    if ($LASTEXITCODE -ne 0) { throw 'Cannot install PyTorch' }
    & $taskPython -m pip install 'transformers==4.57.1' numpy scikit-learn requests
    if ($LASTEXITCODE -ne 0) { throw 'Cannot install inference dependencies' }
} finally { Pop-Location }
