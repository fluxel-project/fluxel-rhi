<#
.SYNOPSIS
Builds example shader artifacts from their single WGSL source of truth.

.DESCRIPTION
The generator compiles each
WGSL entry through Naga 27.0.3 to SPIR-V, runs a fixed SPIRV-Tools legalizing
pipeline, validates Vulkan SPIR-V, reflects descriptor ABI, translates the
validated module with SPIRV-Cross, and produces DXIL with DXC.

No shader text is rewritten. The only backend-specific spelling supplied by
this driver is DX12's LOCATION<n> input semantic, derived from the manifest's
public vertex locations. The WGSL source keeps the WebGPU coordinate contract;
no Y or depth adjustment flag is used.

Without -Write, artifacts are generated in a temporary staging directory and
checked against the checked-in filenames. -Write replaces a target only after
every generation, validation, and ABI check for the selected job succeeds.
#>
[CmdletBinding()]
param(
    [string]$Example = '02_texture',
    [string]$VulkanSdkBin = "$env:VULKAN_SDK\Bin",
    [switch]$Write,
    [switch]$KeepStaging
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-ShaderAbi {
    param(
        [Parameter(Mandatory)] [string]$SpirvCross,
        [Parameter(Mandatory)] [string]$Dxc,
        [Parameter(Mandatory)] [string]$Spv,
        [Parameter(Mandatory)] [string]$Dxil,
        [Parameter(Mandatory)] $Job,
        [Parameter(Mandatory)] $Stage
    )

    $reflection = (& $SpirvCross $Spv --reflect | Out-String | ConvertFrom-Json)
    $groups = @{}
    foreach ($name in @('ubos', 'ssbos', 'separate_images', 'separate_samplers', 'storage_images', 'images')) {
        $property = $reflection.PSObject.Properties[$name]
        if ($null -ne $property -and $null -ne $property.Value) {
            foreach ($resource in @($property.Value)) {
                $groups["${name}:$($resource.set):$($resource.binding)"] = $true
            }
        }
    }
    $bindings = if ($Stage.ContainsKey('Bindings')) { $Stage['Bindings'] } else { $Job.Bindings }
    $expected = @($Stage.VisibleBindings | ForEach-Object { $bindings | Select-Object -Index $_ })
    $expectedKeys = @()
    foreach ($binding in $expected) {
        if ($null -eq $binding -or !$binding.ContainsKey('Kind')) { throw "Invalid expected binding for $($Stage.Name): $($binding | ConvertTo-Json -Compress -Depth 4)" }
        $reflectionKind = switch ($binding.Kind) {
            'Uniform' { 'ubos' }
            'StorageBuffer' { 'ssbos' }
            'Texture' { 'separate_images' }
            'Sampler' { 'separate_samplers' }
            'StorageTexture' { if ($groups.ContainsKey("images:$($binding.Group):$($binding.Binding)")) { 'images' } else { 'storage_images' } }
            default { throw "Unknown binding kind $($binding.Kind)" }
        }
        $key = "${reflectionKind}:$($binding.Group):$($binding.Binding)"
        if (!$groups.ContainsKey($key)) {
            throw "SPIR-V is missing $key for $($Stage.Name)"
        }
        $expectedKeys += $key
    }
    if ($Stage.ContainsKey('NativePushConstant')) {
        $pushResources = $reflection.PSObject.Properties['push_constants']
        if ($null -eq $pushResources -or @($pushResources.Value).Count -ne 1) {
            throw "SPIR-V does not expose the expected single push-constant block for $($Stage.Name)"
        }
    }
    if ($Stage.ContainsKey('NativeDescriptorArray')) {
        $target = $Stage.NativeDescriptorArray
        $image = @($reflection.separate_images | Where-Object {
            $_.set -eq 0 -and $_.binding -eq $target.Binding
        })
        if ($image.Count -ne 1 -or @($image[0].array)[0] -ne $target.Count) {
            throw "SPIR-V descriptor array does not have $($target.Count) entries at binding $($target.Binding) for $($Stage.Name)"
        }
    }
    foreach ($key in $groups.Keys) {
        if ($key -notin $expectedKeys) {
            throw "SPIR-V exposes unexpected resource $key for $($Stage.Name)"
        }
    }

    $dump = (& $Dxc -dumpbin $Dxil | Out-String)
    $entryName = if ($Stage.ContainsKey('Entry')) { $Stage.Entry } else { $Stage.Name }
    if ($dump -notmatch "EntryFunctionName: $([regex]::Escape($entryName))") {
        throw "DXIL entry point mismatch for $entryName"
    }
    if ($Stage.Stage -eq 'vertex') {
        $vertexLocations = if ($Stage.ContainsKey('VertexLocations')) {
            $Stage['VertexLocations']
        } else {
            $Job.VertexLocations
        }
        $consumedLocations = if ($Stage.ContainsKey('ConsumedVertexLocations')) {
            $Stage['ConsumedVertexLocations']
        } else {
            $Job.VertexLocations
        }
        foreach ($location in $consumedLocations) {
            $semanticPattern = "(?m)^; LOCATION\s+$location\s"
            Write-Verbose "Checking DXIL input semantic $location with $semanticPattern"
            if ($dump -notmatch $semanticPattern) {
                throw "DXIL is missing vertex semantic LOCATION$location"
            }
        }
    }
    foreach ($binding in $expected) {
        $register = switch ($binding.Kind) {
            'Uniform' { "cb$($binding.Binding)" }
            'StorageBuffer' { "u$($binding.Binding)" }
            'Texture' { "t$($binding.Binding)" }
            'Sampler' { "s$($binding.Binding)" }
            'StorageTexture' { "u$($binding.Binding)" }
        }
        $space = if ($binding.Group -eq 0) { '(?:,\s*space0)?' } else { ",\s*space$($binding.Group)" }
        if ($dump -notmatch "\b$register$space\b") {
            throw "DXIL is missing register $register in space $($binding.Group) for $($Stage.Name)"
        }
    }
    if ($Stage.ContainsKey('NativeDescriptorArray')) {
        $target = $Stage.NativeDescriptorArray
        if ($dump -notmatch "(?m)\bt$($target.Binding)\s+$($target.Count)\b") {
            throw "DXIL descriptor array does not have $($target.Count) entries at t$($target.Binding) for $($Stage.Name)"
        }
        if ($dump -notmatch '(?s)@dx\.op\.createHandle.{0,256}i1 true') {
            throw "DXIL does not mark descriptor indexing non-uniform for $($Stage.Name)"
        }
    }
    if ($Stage.ContainsKey('NativePushConstant')) {
        $target = $Stage.NativePushConstant
        if ($dump -notmatch "(?m)^;\s*[^\r\n]+\s+cbuffer\s+NA\s+NA\s+CB\d+\s+cb$($target.Register)\s+\d+") {
            throw "DXIL is missing push-constant cbuffer b$($target.Register) for $($Stage.Name)"
        }
        if ($dump -notmatch "(?m)^;\s*}\s+[^;]+;\s*;\s*Offset:\s*0\s+Size:\s*$($target.Bytes)\b") {
            throw "DXIL push-constant block is not $($target.Bytes) bytes for $($Stage.Name)"
        }
    }
    if ($Stage.ContainsKey('NativeViewIndex') -and $dump -notmatch 'UsesViewID:\s*true') {
        throw "DXIL does not consume ViewIndex for $($Stage.Name)"
    }
}

$examplesRoot = $PSScriptRoot
$manifest = Import-PowerShellDataFile (Join-Path $examplesRoot 'wgsl-artifacts/manifest.psd1')
$toolchain = Import-PowerShellDataFile (Join-Path $examplesRoot 'wgsl-artifacts/toolchain.psd1')
$job = @($manifest.Jobs | Where-Object { $_.Name -eq $Example })
if ($job.Count -ne 1) { throw "Unknown or ambiguous WGSL artifact job: $Example" }
$job = $job[0]

$generatorManifest = Join-Path $examplesRoot 'wgsl-artifacts/Cargo.toml'
$spirvOpt = Join-Path $VulkanSdkBin 'spirv-opt.exe'
$spirvVal = Join-Path $VulkanSdkBin 'spirv-val.exe'
$spirvCross = Join-Path $VulkanSdkBin 'spirv-cross.exe'
$dxc = Join-Path $VulkanSdkBin 'dxc.exe'
foreach ($tool in @($spirvOpt, $spirvVal, $spirvCross, $dxc)) {
    if (!(Test-Path -LiteralPath $tool)) { throw "Missing Vulkan SDK tool: $tool" }
}
if (!(Test-Path -LiteralPath $generatorManifest)) { throw "Missing Naga generator: $generatorManifest" }
if ((Split-Path -Leaf (Split-Path -Parent $VulkanSdkBin)) -ne $toolchain.VulkanSdkDirectory) {
    throw "Expected Vulkan SDK $($toolchain.VulkanSdkDirectory), got $VulkanSdkBin."
}
$dxCompiler = Join-Path $VulkanSdkBin 'dxcompiler.dll'
if ((Get-Item -LiteralPath $dxCompiler).VersionInfo.FileVersion -ne $toolchain.DxCompilerFileVersion) {
    throw "Expected dxcompiler.dll $($toolchain.DxCompilerFileVersion)."
}
if (((& $spirvOpt --version 2>&1 | Out-String).Trim()) -notlike "*$($toolchain.SpirvToolsVersion)*") {
    throw "Expected SPIRV-Tools $($toolchain.SpirvToolsVersion)."
}
if (((& $spirvCross --revision 2>&1 | Out-String) -notlike "*$($toolchain.SpirvCrossRevision)*")) {
    throw "Expected SPIRV-Cross $($toolchain.SpirvCrossRevision)."
}

$generatorLock = Join-Path $examplesRoot 'wgsl-artifacts/Cargo.lock'
if (!(Test-Path -LiteralPath $generatorLock)) {
    throw 'Missing locked generator dependencies. Run cargo generate-lockfile in wgsl-artifacts and commit Cargo.lock.'
}
if (-not ((Get-Content -Raw $generatorLock) -match "name = `"naga`"\s+version = `"$([regex]::Escape($toolchain.Naga))`"")) {
    throw "The generator must use locked Naga $($toolchain.Naga)."
}

$stageRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("fluxel-wgsl-$Example-" + [guid]::NewGuid().ToString('N'))
$generatorTarget = Join-Path ([System.IO.Path]::GetTempPath()) 'fluxel-wgsl-artifacts-target'
$previousCargoTargetDir = $env:CARGO_TARGET_DIR
$env:CARGO_TARGET_DIR = $generatorTarget
New-Item -ItemType Directory -Force -Path $stageRoot | Out-Null
try {
    $source = Join-Path $examplesRoot $job.Source
    if (!(Test-Path -LiteralPath $source)) { throw "Missing WGSL source: $source" }
    $built = @()
    foreach ($stage in $job.Stages) {
        $rawSpv = Join-Path $stageRoot ($stage.Name + '.raw.spv')
        $legalSpv = Join-Path $stageRoot ($stage.Name + '.spv')
        $hlsl = Join-Path $stageRoot ($stage.Name + '.hlsl')
        $dxil = Join-Path $stageRoot ($stage.Name + '.dxil')

        $stageSource = if ($stage.ContainsKey('Source')) {
            Join-Path $examplesRoot $stage.Source
        } else {
            $source
        }
        $entryName = if ($stage.ContainsKey('Entry')) { $stage.Entry } else { $stage.Name }
        if (!(Test-Path -LiteralPath $stageSource)) { throw "Missing WGSL source: $stageSource" }
        $vertexLocations = if ($stage.ContainsKey('VertexLocations')) {
            $stage.VertexLocations
        } else {
            $job.VertexLocations
        }
        $generatorArgs = @($stageSource, $stage.Stage, $entryName, $rawSpv)
        if ($stage.Stage -eq 'vertex') {
            $generatorArgs += ($vertexLocations -join ',')
        }
        if ($stage.ContainsKey('NativeHlsl')) {
            if ($stage.Stage -ne 'vertex') { $generatorArgs += '-' }
            if ($stage.ContainsKey('NativePushConstant')) {
                $target = $stage.NativePushConstant
                $generatorArgs += "push=$($target.Group),$($target.Binding),$($target.Register),$($target.Space)"
            } else {
                $generatorArgs += 'hlsl'
            }
            $generatorArgs += $hlsl
        }
        cargo run --locked --manifest-path $generatorManifest --quiet -- @generatorArgs
        if ($LASTEXITCODE -ne 0) { throw "Naga WGSL compilation failed: $($stage.Name)" }

        # Naga 27 preserves WGSL I/O member offsets on a temporary local
        # aggregate. Vulkan validation rejects that raw form. These are stable
        # SPIRV-Tools IR passes, not source patches; the final form is validated
        # before it can become an example artifact.
        & $spirvOpt --eliminate-local-single-block --eliminate-local-single-store `
            --scalar-replacement=0 --ssa-rewrite --eliminate-dead-code-aggressive `
            --simplify-instructions --compact-ids $rawSpv -o $legalSpv
        if ($LASTEXITCODE -ne 0) { throw "SPIRV-Tools legalization failed: $($stage.Name)" }
        & $spirvVal --target-env vulkan1.3 $legalSpv
        if ($LASTEXITCODE -ne 0) { throw "Vulkan SPIR-V validation failed: $($stage.Name)" }

        if (!$stage.ContainsKey('NativeHlsl')) {
            $crossStage = switch ($stage.Stage) {
                'vertex' { 'vert' }
                'fragment' { 'frag' }
                'compute' { 'comp' }
                default { throw "Unsupported SPIRV-Cross stage: $($stage.Stage)" }
            }
            $shaderModel = if ($stage.ContainsKey('HlslShaderModel')) { $stage.HlslShaderModel } else { 60 }
            $crossArgs = @($legalSpv, '--hlsl', '--shader-model', $shaderModel, '--rename-entry-point', $entryName, $entryName, $crossStage)
            if ($stage.ContainsKey('HlslFlattenUbo') -and $stage['HlslFlattenUbo']) {
                $crossArgs += '--flatten-ubo'
            }
            if ($stage.Stage -eq 'vertex') {
                foreach ($location in $vertexLocations) {
                    $crossArgs += @('--set-hlsl-vertex-input-semantic', $location, "LOCATION$location")
                }
            }
            $crossArgs += @('--output', $hlsl)
            & $spirvCross @crossArgs
            if ($LASTEXITCODE -ne 0) { throw "SPIRV-Cross HLSL generation failed: $($stage.Name)" }
        }
        $dxcArgs = @('-T', $stage.Profile, '-E', $entryName)
        if ($stage.ContainsKey('NativeHlsl')) {
            $semanticLocations = if ($stage.ContainsKey('NativePushConstant')) {
                $stage.NativePushConstant.SemanticLocations
            } else {
                $stage.NativeHlsl.SemanticLocations
            }
            foreach ($location in $semanticLocations) {
                $dxcArgs += @('-D', "LOC$location=LOCATION$location")
            }
        }
        $dxcArgs += @('-Fo', $dxil, $hlsl)
        & $dxc @dxcArgs
        if ($LASTEXITCODE -ne 0) { throw "DXC compilation failed: $($stage.Name)" }

        Write-Verbose "Checking generated ABI for $($stage.Name)"
        Assert-ShaderAbi -SpirvCross $spirvCross -Dxc $dxc -Spv $legalSpv -Dxil $dxil -Job $job -Stage $stage
        $built += [pscustomobject]@{ Stage = $stage; SpirV = $legalSpv; Dxil = $dxil }
    }

    foreach ($artifact in $built) {
        foreach ($kind in @('SpirV', 'Dxil')) {
            $from = $artifact.$kind
            $relative = if ($kind -eq 'SpirV') { $artifact.Stage.SpirV } else { $artifact.Stage.Dxil }
            $to = Join-Path $examplesRoot $relative
            if ($Write) {
                $pending = "$to.pending-wgsl"
                Copy-Item -LiteralPath $from -Destination $pending -Force
                Copy-Item -LiteralPath $pending -Destination $to -Force
                Remove-Item -LiteralPath $pending -Force
                Write-Host "Wrote $relative"
            } elseif (Test-Path -LiteralPath $to) {
                $expected = (Get-FileHash -LiteralPath $to -Algorithm SHA256).Hash
                $actual = (Get-FileHash -LiteralPath $from -Algorithm SHA256).Hash
                if ($expected -eq $actual) {
                    Write-Host "Verified $relative $actual"
                } else {
                    Write-Host "Generated $relative $actual (checked-in artifact differs: $expected)"
                }
            } else {
                Write-Host "Generated $relative $(Get-FileHash -LiteralPath $from -Algorithm SHA256 | Select-Object -ExpandProperty Hash)"
            }
        }
    }
} finally {
    if ($KeepStaging) {
        Write-Host "WGSL staging retained at $stageRoot"
    } elseif (Test-Path -LiteralPath $stageRoot) {
        Remove-Item -LiteralPath $stageRoot -Recurse -Force
    }
    if ($null -eq $previousCargoTargetDir) {
        Remove-Item Env:CARGO_TARGET_DIR
    } else {
        $env:CARGO_TARGET_DIR = $previousCargoTargetDir
    }
}
