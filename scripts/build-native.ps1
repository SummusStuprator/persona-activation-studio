param([switch]$Cuda,[int]$Jobs=0)
$ErrorActionPreference='Stop'
$Root=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Vendor=Join-Path $Root '.vendor\llama.cpp'
$Build=Join-Path $Root 'native\build'
$Runtime=Join-Path $Root 'native\runtime'
$Commit='c0bc8591e8815c63cb01dd3f051a8b0df02501c9'
if($Jobs -le 0){$Jobs=[Math]::Max(1,[Math]::Min(4,[Math]::Ceiling([Environment]::ProcessorCount/2)))}
if(-not (Test-Path (Join-Path $Vendor '.git'))){
  New-Item -ItemType Directory -Force -Path $Vendor|Out-Null
  git -C $Vendor init
  git -C $Vendor remote add origin https://github.com/ggerganov/llama.cpp.git
}
git -C $Vendor fetch --depth 1 origin $Commit
if($LASTEXITCODE -ne 0){throw 'llama.cpp pinned-commit fetch failed.'}
git -C $Vendor checkout --detach FETCH_HEAD
if($LASTEXITCODE -ne 0){throw "Could not checkout llama.cpp $Commit"}
$cudaFlag=if($Cuda){'ON'}else{'OFF'}
& cmake --fresh -S (Join-Path $Root 'native') -B $Build "-DLLAMA_CPP_DIR=$Vendor" "-DGGML_CUDA=$cudaFlag" "-DCMAKE_BUILD_TYPE=Release"
if($LASTEXITCODE -ne 0){throw "CMake configure failed (GGML_CUDA=$cudaFlag)."}
& cmake --build $Build --config Release --target activation_bridge --parallel $Jobs
if($LASTEXITCODE -ne 0){throw 'Native activation_bridge build failed.'}
if(Test-Path $Runtime){Remove-Item -Recurse -Force $Runtime}
New-Item -ItemType Directory -Force -Path $Runtime|Out-Null
$patterns=@('*.dll','*.so','*.so.*','*.dylib')
foreach($pat in $patterns){
  Get-ChildItem $Build -Recurse -File -Filter $pat -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item -Force $_.FullName (Join-Path $Runtime $_.Name)
  }
}
if(-not (Get-ChildItem $Runtime -File | Where-Object {$_.Name -match 'activation_bridge'})){throw 'activation_bridge library was not found after build.'}
if(-not (Get-ChildItem $Runtime -File | Where-Object {$_.Name -match '(^|lib)llama'})){throw 'llama shared library was not found after build.'}
Write-Host "Native runtime copied to $Runtime (GGML_CUDA=$cudaFlag, parallel jobs=$Jobs)"
