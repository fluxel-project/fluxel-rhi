<#
.SYNOPSIS
Fetches the exact SaschaWillems/Vulkan example assets used by this workspace.

.DESCRIPTION
Downloads exactly the 20 resources declared below from the pinned Vulkan-Assets
commit. Each download is SHA-256 verified before it is placed in Destination.
Existing files are verified and retained; a mismatched existing file causes the
script to stop instead of overwriting it.
#>
[CmdletBinding()]
param(
    [Parameter()]
    [string]$Destination = (Join-Path $PSScriptRoot 'assets')
)

$ErrorActionPreference = 'Stop'
$commit = 'a27c0e584434d59b7c7a714e9180eefca6f0ec4b'
$baseUri = "https://raw.githubusercontent.com/SaschaWillems/Vulkan-Assets/$commit"

$assets = [ordered]@{
    'models/rock01.gltf' = 'edadeca515daa8c27ba429b531265a7f2fd63fc48690cfde95d91a36d122b66e'
    'models/lavaplanet.gltf' = '23b5aa47d30d16e4f35be4fd8f4db69dad4989c1ef48462447c4276381dea501'
    'models/sphere.gltf' = 'b12dc1d1945e9a450beea6bea49c3a5162b7c9cfdc7f336a0ba3172e49da2903'
    'models/plane.gltf' = 'f77c1b7d558a26610cde672efc53eac8975ca68aab631efe8aac6d34ee98e10e'
    'models/chinesedragon.gltf' = '8e69e1a2337babf6a508bfa44d7aed41f92d85b04c9c3c2eeb5ba4059509a2e5'
    'models/voyager.gltf' = '4188ba7cdd417548a8c19d2f2107630be1ae4a11f453756fce4827c2d94cd2ee'
    'models/tunnel_cylinder.gltf' = '8e78a2166751eebb6aa8e2b3c97922c8c72e27bd2879f1afc74fa2b34562ff7c'
    'models/plants.gltf' = 'bc7157aab8b08d7e4b22c8cf1aa593fd4362459321d5d155b48207266cbdfe51'
    'models/plane_circle.gltf' = '1c907424d6c6b397357a436e1184e8284e2c1ed2311421664db536f3ab676781'
    'models/plane_z.gltf' = '0b4b92841fdc7a2995f904aeab2bf683ac2eb95a2bccd4c74d8e0b1fdbd1fae6'
    'models/teapot.gltf' = '0e0b31bd21d9830cae2dabdfdc1f456f0417788ad6042b390fa7b8b87e0f80ae'
    'models/retroufo_red_lowpoly.gltf' = '22fed876f9dd64b0a7b27755fa880ce751017d1b1057cd2980656e2459082bf8'
    'models/sampleroom.gltf' = '7378e750d5b62cd0131e72bbf75e0f173fcc1e98487fbbe4e381cb41b8ea2fb7'
    'textures/metalplate01_rgba.ktx' = '38495b089915d38e2ae6ce14cdf872ff7d2df6f79f3a1e5a7a7851cbe65882bd'
    'textures/lavaplanet_rgba.ktx' = '73a3b8a8a72a395a790a335f4ffc29951ed2590e491552031b16e2613b5c9fe2'
    'textures/texturearray_rocks_rgba.ktx' = 'f6f25a7ae5049eca84fc4611d4926231812087f9f3490891dbd65732a25608bd'
    'textures/vulkan_11_rgba.ktx' = '9b960ab010872681bf26a63bf269ffc638d182def9ad5dd6823d7d871e7d809f'
    'textures/metalplate_nomips_rgba.ktx' = 'b108b26bc436ddf760283380acbd22ce953eef458fc4e790dc089297d74aeb95'
    'textures/texturearray_plants_rgba.ktx' = '524dbb802d98d790d061c8c1445eecfb8e818f4d670016f40894351ecbef7a9e'
    'textures/ground_dry_rgba.ktx' = 'fd4d21f5c4d1ada025552bb2acec26cbf315ee696cb2821e9b00fedf971a2749'
}

foreach ($entry in $assets.GetEnumerator()) {
    $relativePath = $entry.Key
    $expectedHash = $entry.Value
    $destinationPath = Join-Path $Destination $relativePath

    if (Test-Path -LiteralPath $destinationPath) {
        $actualHash = (Get-FileHash -LiteralPath $destinationPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $expectedHash) {
            throw "Existing asset has an unexpected SHA-256: $destinationPath"
        }
        Write-Host "Verified $relativePath"
        continue
    }

    $parentDirectory = Split-Path -Parent $destinationPath
    New-Item -ItemType Directory -Force -Path $parentDirectory | Out-Null
    $partialPath = "$destinationPath.partial"
    if (Test-Path -LiteralPath $partialPath) {
        throw "Partial download already exists: $partialPath"
    }

    $uri = "$baseUri/$relativePath"
    Write-Host "Downloading $relativePath"
    Invoke-WebRequest -Uri $uri -OutFile $partialPath

    $actualHash = (Get-FileHash -LiteralPath $partialPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) {
        Remove-Item -LiteralPath $partialPath
        throw "Downloaded asset has an unexpected SHA-256: $relativePath"
    }

    Move-Item -LiteralPath $partialPath -Destination $destinationPath
    Write-Host "Verified $relativePath"
}

Write-Host "Verified $($assets.Count) pinned Vulkan-Assets resources."
