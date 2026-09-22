param([switch]$Cuda,[int]$Jobs=2,[string]$CudaRoot='',[string]$Architectures='native')
$ErrorActionPreference='Stop'
$Root=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Py=$env:STUDIO_PYTHON
if(-not $Py){$Py=Join-Path $Root '.venv\Scripts\python.exe'}
if(-not (Test-Path $Py)){$Py=(Get-Command python -ErrorAction Stop).Source}
$Arguments=@('-m','studio_native','--jobs',"$Jobs",'--architectures',$Architectures)
if($Cuda){$Arguments+='--cuda'}
if($CudaRoot){$Arguments+=@('--cuda-root',$CudaRoot)}
Set-Location $Root
& $Py @Arguments
exit $LASTEXITCODE
