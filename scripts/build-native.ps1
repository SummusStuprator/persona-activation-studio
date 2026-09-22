param([switch]$Cuda,[int]$Jobs=2,[string]$CudaRoot='',[string]$Architectures='native')
$ErrorActionPreference='Stop'
$Root=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Vendor=Join-Path $Root '.vendor\llama.cpp'
$flavor=if($Cuda){'cuda'}else{'cpu'}
$Build=Join-Path $Root "native\build-$flavor"
$Py=Join-Path $Root '.venv\Scripts\python.exe'
$Commit='c0bc8591e8815c63cb01dd3f051a8b0df02501c9'
if(-not (Test-Path $Py)){throw 'Run scripts/setup.ps1 first.'}
if($Jobs -lt 1){throw 'Jobs must be positive.'}
if(-not (Test-Path (Join-Path $Vendor '.git'))){
  New-Item -ItemType Directory -Force -Path $Vendor|Out-Null
  git -C $Vendor init
  git -C $Vendor remote add origin https://github.com/ggml-org/llama.cpp.git
}
$present=$false
try {
  git -C $Vendor cat-file -e "$Commit^{commit}" 2>$null
  $present=($LASTEXITCODE -eq 0)
} catch { $present=$false }
if(-not $present){
  git -C $Vendor fetch --quiet --depth 1 origin $Commit
  if($LASTEXITCODE -ne 0){throw 'Pinned llama.cpp fetch failed.'}
}
if(git -C $Vendor status --porcelain){throw 'Preserve vendor edits before building a pinned revision.'}
git -C $Vendor checkout --quiet --detach $Commit
if($LASTEXITCODE -ne 0){throw 'Pinned checkout failed.'}
$options=@('-S',(Join-Path $Root 'native'),'-B',$Build,"-DLLAMA_CPP_DIR=$Vendor",'-DGGML_BACKEND_DL=ON','-DGGML_NATIVE=OFF','-DCMAKE_BUILD_TYPE=Release')
$targets=@('activation_bridge','ggml-cpu')
if($Cuda){
  if(-not $CudaRoot){
    $nvcc=(Get-Command nvcc -ErrorAction Stop).Source
    $CudaRoot=Split-Path (Split-Path $nvcc -Parent) -Parent
  }
  $options+=@('-T',"cuda=$CudaRoot",'-DGGML_CUDA=ON',"-DCMAKE_CUDA_ARCHITECTURES=$Architectures")
  $targets+='ggml-cuda'
}else{$options+='-DGGML_CUDA=OFF'}
& cmake @options
if($LASTEXITCODE -ne 0){throw 'Native configure failed; installed runtime is unchanged.'}
& cmake --build $Build --config Release --target @targets --parallel $Jobs
if($LASTEXITCODE -ne 0){throw 'Native build failed; installed runtime is unchanged.'}
$deploy=@((Join-Path $Root 'native_deploy.py'),$Build)
if($Cuda){$deploy+='--cuda'}
& $Py @deploy
if($LASTEXITCODE -ne 0){throw 'Runtime validation/deployment failed; review the error above.'}
