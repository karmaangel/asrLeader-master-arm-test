param(
    [string]$Version = "20260610",
    [string]$BaseImage = "crpi-h5ujs5q57e0e72tz.cn-shanghai.personal.cr.aliyuncs.com/dream_acr/meeting-asr:npu",
    [string]$ImageTag = "",
    [string]$OutputDir = "",
    [string]$QwenSource = "",
    [switch]$SkipBaseProbe
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
if (-not $ImageTag) {
    $ImageTag = "funasr-leader-asr:npu-prod-complete-$Version"
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $Root "dist"
}
if (-not $QwenSource) {
    $QwenSource = Join-Path $Root "deploy\ascend-npu\models\Qwen2.5-1.5B-Instruct"
}

$qwenModelFile = Join-Path $QwenSource "model.safetensors"
if (-not (Test-Path -LiteralPath $qwenModelFile)) {
    throw "Qwen model.safetensors not found: $qwenModelFile"
}
$qwenSize = (Get-Item -LiteralPath $qwenModelFile).Length
if ($qwenSize -lt 1000000000) {
    throw "Qwen model.safetensors looks too small: $qwenSize bytes"
}

$entrypoint = Join-Path $Root "docker\entrypoint.npu.sh"
$entrypointText = [System.IO.File]::ReadAllText($entrypoint)
if ($entrypointText.Contains("`r`n")) {
    throw "NPU entrypoint has CRLF line endings. Convert to LF before building: $entrypoint"
}

$baseInspectRaw = docker image inspect $BaseImage 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Base image not found locally: $BaseImage. Pull or load the production NPU base image first."
}
$baseInspect = ($baseInspectRaw -join "`n") | ConvertFrom-Json
$baseArch = $baseInspect[0].Architecture
$baseOs = $baseInspect[0].Os
$baseId = $baseInspect[0].Id
if ($baseArch -ne "arm64" -or $baseOs -ne "linux") {
    throw "Base image must be linux/arm64, got ${baseOs}/${baseArch}: $BaseImage"
}
if ($BaseImage -eq "meeting-asr:npu-new") {
    throw "Refusing known incompatible base image meeting-asr:npu-new. Use the production NPU base image instead."
}

if (-not $SkipBaseProbe) {
    $probeCode = "import importlib.metadata as md, sys; print('python=%d.%d.%d' % sys.version_info[:3]); [print(d + '=' + md.version(d)) for d in ['funasr','modelscope','torch','torch-npu','fastapi','pydantic','transformers','uvicorn','soundfile','python-multipart','safetensors']]"
    $probeOutput = docker run --rm --platform linux/arm64 --entrypoint python $BaseImage -c $probeCode 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Base probe:"
        $probeOutput | ForEach-Object { Write-Host "  $_" }

        function Get-ProbeValue([string]$Name) {
            $line = $probeOutput | Where-Object { $_ -like "$Name=*" } | Select-Object -First 1
            if (-not $line) {
                return $null
            }
            return $line.ToString().Substring($Name.Length + 1)
        }

        $expectedExact = @{
            "funasr" = "1.3.1"
            "modelscope" = "1.36.0"
            "torch" = "2.1.0"
            "torch-npu" = "2.1.0.post10"
            "fastapi" = "0.136.0"
            "pydantic" = "2.9.2"
            "transformers" = "4.44.0"
            "uvicorn" = "0.44.0"
            "soundfile" = "0.13.1"
            "python-multipart" = "0.0.26"
            "safetensors" = "0.4.5"
        }

        $pythonVersion = Get-ProbeValue "python"
        if (-not $pythonVersion -or -not $pythonVersion.StartsWith("3.11.")) {
            throw "Base image Python version is not the working production 3.11.x: python=$pythonVersion"
        }
        foreach ($name in $expectedExact.Keys) {
            $actual = Get-ProbeValue $name
            if ($actual -ne $expectedExact[$name]) {
                throw "Base image package mismatch: $name expected $($expectedExact[$name]), got $actual"
            }
        }
    } else {
        Write-Warning "Could not run base-image version probe locally. This is OK if Docker Desktop cannot emulate linux/arm64. Probe output:"
        $probeOutput | ForEach-Object { Write-Warning "  $_" }
    }
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "Base image : $BaseImage"
Write-Host "Base id    : $baseId"
Write-Host "Target tag : $ImageTag"
Write-Host "Platform   : linux/arm64"
Write-Host "Qwen model : $qwenModelFile ($qwenSize bytes)"

$env:DOCKER_BUILDKIT = "1"

docker build `
    --platform linux/arm64 `
    --build-arg "BASE_IMAGE=$BaseImage" `
    -f docker/Dockerfile.npu-prod-complete `
    -t $ImageTag `
    .
if ($LASTEXITCODE -ne 0) {
    Write-Warning "docker build failed, trying docker buildx build --load"
    docker buildx build `
        --platform linux/arm64 `
        --load `
        --build-arg "BASE_IMAGE=$BaseImage" `
        -f docker/Dockerfile.npu-prod-complete `
        -t $ImageTag `
        .
    if ($LASTEXITCODE -ne 0) {
        throw "docker build/buildx failed"
    }
}

$inspect = docker image inspect $ImageTag --format '{{.Id}} {{.Architecture}} {{.Os}} {{json .Config.Entrypoint}} {{json .Config.Cmd}}'
if ($LASTEXITCODE -ne 0) {
    throw "docker image inspect failed for $ImageTag"
}
Write-Host $inspect
if ($inspect -notmatch "arm64") {
    throw "Built image is not arm64: $inspect"
}

$safeName = $ImageTag -replace "[:/]", "-"
$imageTar = Join-Path $OutputDir "$safeName.tar"
if (Test-Path -LiteralPath $imageTar) {
    Remove-Item -LiteralPath $imageTar -Force
}

docker save -o $imageTar $ImageTag
if ($LASTEXITCODE -ne 0) {
    throw "docker save failed"
}

$hash = Get-FileHash -LiteralPath $imageTar -Algorithm SHA256
$hashPath = "$imageTar.sha256"
"$($hash.Hash)  $(Split-Path -Leaf $imageTar)" | Set-Content -LiteralPath $hashPath -Encoding ASCII

Write-Host ""
Write-Host "Image tar : $imageTar"
Write-Host "SHA256    : $($hash.Hash)"
Write-Host ""
Write-Host "Upload the image tar to the production worker, then run:"
Write-Host "  nerdctl -n k8s.io load -i $(Split-Path -Leaf $imageTar)"
Write-Host "  kubectl -n test set image deployment/asr-leader-npu asr-leader-npu=$ImageTag"
Write-Host "  kubectl -n test rollout status deployment/asr-leader-npu --timeout=10m"
