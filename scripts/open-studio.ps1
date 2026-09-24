$ErrorActionPreference='Stop'
$Root=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Py=Join-Path $Root '.venv\Scripts\python.exe'
if(-not (Test-Path $Py)){throw 'Run scripts/setup.ps1 first.'}
Set-Location $Root
$Port=& $Py -c "from studio_config import load_config; print(load_config().get('runtime',{}).get('port',8899))"
if($LASTEXITCODE -ne 0){throw 'Could not read Studio configuration.'}
$Url="http://127.0.0.1:$Port"
function Healthy {
  try {return (Invoke-WebRequest "$Url/_stcore/health" -UseBasicParsing -TimeoutSec 2).Content.Trim() -eq 'ok'}
  catch {return $false}
}
if(-not (Healthy)){
  $Logs=Join-Path $Root 'logs'
  New-Item -ItemType Directory -Force $Logs|Out-Null
  $Process=Start-Process -FilePath $Py -ArgumentList @('-m','studio_cli','app') -WorkingDirectory $Root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $Logs 'ui-start.log') -RedirectStandardError (Join-Path $Logs 'ui-start.err')
  for($i=0;$i -lt 30;$i++){
    if(Healthy){break}
    if($Process.HasExited){throw "Studio exited. Review $Logs\ui-start.err"}
    Start-Sleep -Milliseconds 500
  }
  if(-not (Healthy)){throw "Studio has not answered yet. Review $Logs\ui-start.err"}
}
Start-Process $Url
