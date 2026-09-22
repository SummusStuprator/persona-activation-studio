param(
  [ValidateSet('core','scrape','persona','train','all')] [string]$Profile = 'core',
  [switch]$BuildNative,
  [switch]$Cuda,
  [string]$Python = ''
)
$ErrorActionPreference='Stop'
$Root=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Venv=Join-Path $Root '.venv'

if(-not $Python){
  $launcher=(Get-Command py.exe -ErrorAction SilentlyContinue).Source
  if($launcher){
    foreach($version in @('3.12','3.11','3.13')){
      try{
        $candidate=& $launcher "-$version" -c "import sys; print(sys.executable)" 2>$null
        if($LASTEXITCODE -eq 0 -and $candidate){$Python=$candidate.Trim(); break}
      }catch{}
    }
  }
}
if(-not $Python){
  $Python=(Get-Command python.exe -ErrorAction Stop).Source
}
Write-Host "Bootstrap Python: $Python"

if(-not (Test-Path $Venv)){ & $Python -m venv $Venv }
$Py=Join-Path $Venv 'Scripts\python.exe'
if(-not (Test-Path $Py)){throw "Virtualenv Python was not created: $Py"}

& $Py -m pip install --upgrade pip setuptools wheel
$Extra = switch($Profile){ 'core' {''}; 'scrape' {'[scrape]'}; 'persona' {'[persona]'}; 'train' {'[train]'}; 'all' {'[all]'} }
& $Py -m pip install -e ($Root+$Extra)
if($LASTEXITCODE -ne 0){ throw 'Python dependency install failed.' }
& $Py -m studio_cli init
if($BuildNative){
  & (Join-Path $PSScriptRoot 'build-native.ps1') -Cuda:$Cuda
}
Write-Host ''
Write-Host 'Core environment installed.'
Write-Host ('Start UI: ' + (Join-Path $PSScriptRoot 'start.ps1'))
Write-Host ('CLI:      ' + (Join-Path $Venv 'Scripts\studio.exe') + ' --help')
