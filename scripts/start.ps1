param([int]$Port=8899)
$ErrorActionPreference='Stop'
$Root=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Py=Join-Path $Root '.venv\Scripts\python.exe'
if(-not (Test-Path $Py)){throw 'Run scripts/setup.ps1 first.'}
Set-Location $Root
& $Py -m streamlit run studio_app.py --server.address=127.0.0.1 --server.port=$Port --server.headless=true --server.fileWatcherType=none --browser.gatherUsageStats=false
