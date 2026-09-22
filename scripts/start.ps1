param([int]$Port=0)
$ErrorActionPreference='Stop'
$Root=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Py=Join-Path $Root '.venv\Scripts\python.exe'
if(-not (Test-Path $Py)){throw 'Run scripts/setup.ps1 first.'}
Set-Location $Root
$AppArgs=@('-m','studio_cli','app')
if($Port -gt 0){$AppArgs+=@('--port',"$Port")}
elseif($env:STUDIO_PORT){$AppArgs+=@('--port',$env:STUDIO_PORT)}
& $Py @AppArgs
exit $LASTEXITCODE
