param([int]$BatchSize = 4, [string]$PythonPath = '')
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$env:TEMP = Join-Path $taskRoot 'work\tmp'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $taskRoot 'work\pip-cache'
$env:HF_HOME = Join-Path $taskRoot 'work\hf-cache'
$env:TORCH_HOME = Join-Path $taskRoot 'work\torch-cache'
$env:CUDA_CACHE_PATH = Join-Path $taskRoot 'work\cuda-cache'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if ($PythonPath) { $taskPython = $PythonPath }
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
Push-Location $taskRoot
try {
    & $taskPython -u run_all_v2.py --batch-size $BatchSize
    if ($LASTEXITCODE -ne 0) { throw 'V2 inference or verification failed; see Python output' }
} finally { Pop-Location }
