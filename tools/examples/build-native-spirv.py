"""Build and validate the native SPIR-V artifacts for native examples."""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'crates/rhi/examples/assets'
ARTIFACTS = (
    ('16_raytracing_hello/Raytracing_vulkan.hlsl', '', 'lib_6_3', '16_raytracing_hello/Raytracing.spv'),
    ('16_raytracing_hello/display.hlsl', 'vs_main', 'vs_6_0', '16_raytracing_hello/display_vs.spv'),
    ('16_raytracing_hello/display.hlsl', 'ps_main', 'ps_6_0', '16_raytracing_hello/display_ps.spv'),
    ('17_gpu_queries/shaders_rhi.hlsl', 'VSMain', 'vs_6_0', '17_gpu_queries/triangle_vs.spv'),
    ('17_gpu_queries/shaders_rhi.hlsl', 'PSMain', 'ps_6_0', '17_gpu_queries/triangle_ps.spv'),
    ('18_meshlet_instancing/MeshletMS.vulkan.hlsl', 'main', 'ms_6_5', '18_meshlet_instancing/MeshletMS.spv'),
    ('18_meshlet_instancing/MeshletPS.hlsl', 'main', 'ps_6_0', '18_meshlet_instancing/MeshletPS.spv'),
    ('19_ray_query_particles/controlled.vulkan.hlsl', 'main', 'cs_6_5', '19_ray_query_particles/controlled.spv'),
    ('20_gpu_draw_count/cull.rhi.hlsl', 'CSMain', 'cs_6_0', '20_gpu_draw_count/cull.spv'),
    ('20_gpu_draw_count/draw.rhi.hlsl', 'VSMain', 'vs_6_0', '20_gpu_draw_count/draw_vs.spv'),
    ('20_gpu_draw_count/draw.rhi.hlsl', 'PSMain', 'ps_6_0', '20_gpu_draw_count/draw_ps.spv'),
    ('21_sparse_texture/shaders.rhi.hlsl', 'VSMain', 'vs_6_0', '21_sparse_texture/vs.spv'),
    ('21_sparse_texture/shaders.rhi.hlsl', 'PSMain', 'ps_6_0', '21_sparse_texture/ps.spv'),
    ('22_clear_texture/shaders_rhi.hlsl', 'VSMain', 'vs_6_0', '22_clear_texture/triangle_vs.spv'),
    ('22_clear_texture/shaders_rhi.hlsl', 'PSMain', 'ps_6_0', '22_clear_texture/triangle_ps.spv'),
    ('23_multi_queue/compute.hlsl', 'CSMain', 'cs_6_0', '23_multi_queue/compute.spv'),
    ('23_multi_queue/draw.hlsl', 'VSMain', 'vs_6_0', '23_multi_queue/draw_vs.spv'),
    ('23_multi_queue/draw.hlsl', 'PSMain', 'ps_6_0', '23_multi_queue/draw_ps.spv'),
)

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    result = []
    with tempfile.TemporaryDirectory() as tmp:
        for source, entry, profile, output in ARTIFACTS:
            path = Path(tmp) / Path(output).name
            command = ['dxc', '-spirv', '-fspv-target-env=vulkan1.2', '-fvk-use-dx-layout']
            if profile.startswith('ms_'):
                command.append('-fspv-extension=SPV_EXT_mesh_shader')
            if source.startswith('19_ray_query_particles/'):
                command.append('-fspv-extension=SPV_KHR_ray_query')
            if profile.startswith('lib_'):
                command.append('-fspv-extension=SPV_KHR_ray_tracing')
            entry_args = ['-E', entry] if entry else []
            subprocess.run([*command, '-T', profile, *entry_args, '-Fo', str(path), str(ASSETS / source)], check=True)
            subprocess.run(['spirv-val', '--target-env', 'vulkan1.2', str(path)], check=True)
            data = path.read_bytes()
            target = ASSETS / output
            if args.verify:
                if target.read_bytes() != data:
                    raise SystemExit(f'artifact differs: {output}')
            else:
                target.write_bytes(data)
            result.append({'source': source, 'entry': entry, 'profile': profile, 'output': output, 'sha256': hashlib.sha256(data).hexdigest()})
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    run()
