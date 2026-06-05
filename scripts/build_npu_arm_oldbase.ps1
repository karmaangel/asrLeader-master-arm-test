param(
    [string]$ImageTag = "funasr-leader-asr:npu-ascend-oldbase-20260604",
    [string]$BaseImage = "ascendai/pytorch:2.1.0-ubuntu22.04",
    [string]$TorchNpuVersion = "2.1.0.post10",
    [string]$PackageDir = "deploy\ascend-npu-oldbase-20260604"
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

function Invoke-Checked {
    param(
        [Parameter(Mandatory=$true)]
        [scriptblock]$Command,
        [string]$Name = "command"
    )

    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Building NPU ARM image"
Write-Host "  image: $ImageTag"
Write-Host "  base : $BaseImage"
Write-Host "  torch-npu: $TorchNpuVersion"

Invoke-Checked -Name "docker buildx build" -Command {
    docker buildx build `
        --platform linux/arm64 `
        --provenance=false `
        --sbom=false `
        --build-arg "NPU_BASE_IMAGE=$BaseImage" `
        --build-arg "TORCH_NPU_VERSION=$TorchNpuVersion" `
        -f docker/Dockerfile.npu-builtin `
        -t $ImageTag `
        --load `
        .
}

$ImageSafeName = ($ImageTag -replace "[:/]", "-")
$OutDir = Join-Path $Root $PackageDir
$ImageDir = Join-Path $OutDir "image"
New-Item -ItemType Directory -Force -Path $ImageDir | Out-Null

$ImageTar = Join-Path $ImageDir "$ImageSafeName.tar"
Invoke-Checked -Name "docker save" -Command {
    docker save -o $ImageTar $ImageTag
}

$Hash = (Get-FileHash -Algorithm SHA256 $ImageTar).Hash
Set-Content -Encoding ASCII -Path (Join-Path $OutDir "SHA256SUMS.txt") -Value "$Hash  image/$ImageSafeName.tar"

$PackageName = "$(Split-Path -Leaf $OutDir).tar"
if (Test-Path $PackageName) {
    Remove-Item -Force $PackageName
}
Invoke-Checked -Name "tar package" -Command {
    tar -cf $PackageName -C (Split-Path -Parent $OutDir) (Split-Path -Leaf $OutDir)
}

Write-Host ""
Write-Host "Done"
Write-Host "  image tar : $ImageTar"
Write-Host "  package   : $(Join-Path $Root $PackageName)"
Write-Host "  deploy dir: $OutDir"
