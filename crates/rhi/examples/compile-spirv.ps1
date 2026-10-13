<#
.SYNOPSIS
Rebuilds checked-in Vulkan SPIR-V example shaders from their project-owned HLSL.

.DESCRIPTION
Each job names one source file, entry point, shader profile, and checked-in
output. The script validates every newly written binary with `spirv-val`; it
does not generate DXIL, download assets, or alter example Rust code.
#>
param(
    [string]$VulkanSdkBin = "$env:VULKAN_SDK\Bin"
)

$ErrorActionPreference = 'Stop'
$dxc = Join-Path $VulkanSdkBin 'dxc.exe'
$validator = Join-Path $VulkanSdkBin 'spirv-val.exe'
if (!(Test-Path $dxc) -or !(Test-Path $validator)) {
    throw 'Set VULKAN_SDK or pass -VulkanSdkBin with dxc.exe and spirv-val.exe.'
}

$jobs = @(
    @('01_triangle', 'triangle.hlsl', 'vs_main', 'vs_6_0', 'triangle_vs.spv'),
    @('01_triangle', 'triangle.hlsl', 'ps_main', 'ps_6_0', 'triangle_ps.spv'),
    @('02_texture', 'texture.hlsl', 'vs_main', 'vs_6_0', 'texture.vs.spv'),
    @('02_texture', 'texture.hlsl', 'fs_main', 'ps_6_0', 'texture.ps.spv'),
    @('03_instancing', 'instancing.hlsl', 'rock_vs', 'vs_6_0', 'rock_vs.spv'),
    @('03_instancing', 'rock_ps.hlsl', 'rock_ps', 'ps_6_0', 'rock_ps.spv'),
    @('03_instancing', 'instancing.hlsl', 'planet_vs', 'vs_6_0', 'planet_vs.spv'),
    @('03_instancing', 'instancing.hlsl', 'planet_ps', 'ps_6_0', 'planet_ps.spv'),
    @('03_instancing', 'instancing.hlsl', 'star_vs', 'vs_6_0', 'star_vs.spv'),
    @('03_instancing', 'instancing.hlsl', 'star_ps', 'ps_6_0', 'star_ps.spv'),
    @('04_dynamic_uniform_buffer', 'dynamic_uniform.hlsl', 'vs_main', 'vs_6_0', 'dynamic_uniform.vs.spv'),
    @('04_dynamic_uniform_buffer', 'dynamic_uniform.hlsl', 'fs_main', 'ps_6_0', 'dynamic_uniform.ps.spv'),
    @('05_push_constants', 'push_constants.hlsl', 'vs_main', 'vs_6_0', 'push_constants.vs.spv'),
    @('05_push_constants', 'push_constants.hlsl', 'fs_main', 'ps_6_0', 'push_constants.ps.spv'),
    @('06_compute', 'compute_filters.hlsl', 'emboss', 'cs_6_0', 'emboss.spv'),
    @('06_compute', 'compute_filters.hlsl', 'edgedetect', 'cs_6_0', 'edgedetect.spv'),
    @('06_compute', 'compute_filters.hlsl', 'sharpen', 'cs_6_0', 'sharpen.spv'),
    @('06_compute', 'display.hlsl', 'vs_main', 'vs_6_0', 'display_vs.spv'),
    @('06_compute', 'display.hlsl', 'ps_main', 'ps_6_0', 'display_ps.spv'),
    @('07_offscreen', 'offscreen.hlsl', 'phong_vs', 'vs_6_0', 'phong.vs.spv'),
    @('07_offscreen', 'offscreen.hlsl', 'phong_ps', 'ps_6_0', 'phong.ps.spv'),
    @('07_offscreen', 'offscreen.hlsl', 'mirror_vs', 'vs_6_0', 'mirror.vs.spv'),
    @('07_offscreen', 'offscreen.hlsl', 'mirror_ps', 'ps_6_0', 'mirror.ps.spv'),
    @('08_msaa', 'msaa.hlsl', 'vs_main', 'vs_6_0', 'msaa.vs.spv'),
    @('08_msaa', 'msaa.hlsl', 'ps_main', 'ps_6_0', 'msaa.ps.spv'),
    @('09_texture_mipmap', 'tunnel.hlsl', 'vs_main', 'vs_6_0', 'tunnel.vs.spv'),
    @('09_texture_mipmap', 'tunnel.hlsl', 'ps_main', 'ps_6_0', 'tunnel.ps.spv'),
    @('10_indirect_draw', 'indirect.hlsl', 'plants_vs', 'vs_6_0', 'plants_vs.spv'),
    @('10_indirect_draw', 'indirect.hlsl', 'plants_ps', 'ps_6_0', 'plants_ps.spv'),
    @('10_indirect_draw', 'indirect.hlsl', 'ground_vs', 'vs_6_0', 'ground_vs.spv'),
    @('10_indirect_draw', 'indirect.hlsl', 'ground_ps', 'ps_6_0', 'ground_ps.spv'),
    @('10_indirect_draw', 'indirect.hlsl', 'sky_vs', 'vs_6_0', 'sky_vs.spv'),
    @('10_indirect_draw', 'indirect.hlsl', 'sky_ps', 'ps_6_0', 'sky_ps.spv'),
    @('11_occlusion_query', 'occlusion.hlsl', 'vs_main', 'vs_6_0', 'occlusion.vs.spv'),
    @('11_occlusion_query', 'occlusion.hlsl', 'simple_ps', 'ps_6_0', 'simple.ps.spv'),
    @('11_occlusion_query', 'occlusion.hlsl', 'mesh_ps', 'ps_6_0', 'mesh.ps.spv'),
    @('11_occlusion_query', 'occlusion.hlsl', 'occluder_ps', 'ps_6_0', 'occluder.ps.spv'),
    @('12_screenshot', 'screenshot.hlsl', 'vs_main', 'vs_6_0', 'screenshot.vs.spv'),
    @('12_screenshot', 'screenshot.hlsl', 'ps_main', 'ps_6_0', 'screenshot.ps.spv')
    ,@('13_multithreaded_recording', 'ufo.hlsl', 'vs_main', 'vs_6_0', 'ufo_vs.spv')
    ,@('13_multithreaded_recording', 'ufo.hlsl', 'ps_main', 'ps_6_0', 'ufo_ps.spv')
    ,@('13_multithreaded_recording', 'stars.hlsl', 'vs_main', 'vs_6_0', 'stars_vs.spv')
    ,@('13_multithreaded_recording', 'stars.hlsl', 'ps_main', 'ps_6_0', 'stars_ps.spv')
    ,@('14_descriptor_indexing', 'descriptor_indexing.hlsl', 'vs_main', 'vs_6_0', 'descriptor_indexing.vs.spv')
    ,@('14_descriptor_indexing', 'descriptor_indexing.hlsl', 'ps_main', 'ps_6_0', 'descriptor_indexing.ps.spv')
    ,@('15_multiview', 'scene.hlsl', 'vs_main', 'vs_6_1', 'scene_vs.spv')
    ,@('15_multiview', 'scene.hlsl', 'ps_main', 'ps_6_0', 'scene_ps.spv')
    ,@('15_multiview', 'display.hlsl', 'vs_main', 'vs_6_0', 'display_vs.spv')
    ,@('15_multiview', 'display.hlsl', 'ps_main', 'ps_6_0', 'display_ps.spv')
)

foreach ($job in $jobs) {
    $directory = Join-Path $PSScriptRoot ('assets/' + $job[0])
    $source = Join-Path $directory $job[1]
    $output = Join-Path $directory $job[4]
    & $dxc -spirv -D FLUXEL_VULKAN=1 -fvk-use-dx-layout -T $job[3] -E $job[2] -Fo $output $source
    if ($LASTEXITCODE -ne 0) { throw "DXC failed: $source $($job[2])" }
    & $validator --target-env vulkan1.3 $output
    if ($LASTEXITCODE -ne 0) { throw "SPIR-V validation failed: $output" }
}
