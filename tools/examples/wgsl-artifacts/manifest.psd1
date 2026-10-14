@{
    # The manifest is intentionally data only. A new example adds one source,
    # its entry points, vertex input locations, and the public binding ABI.
    # The generator never discovers ABI from a backend-specific source file.
    Jobs = @(
        @{
            Name = '01_triangle'
            Source = 'assets/01_triangle/triangle.wgsl'
            VertexLocations = @(0, 1)
            Bindings = @()
            Stages = @(
                @{ Name = 'vs_main'; Stage = 'vertex'; Profile = 'vs_6_0'; VisibleBindings = @(); SpirV = 'assets/01_triangle/triangle_vs.spv'; Dxil = 'assets/01_triangle/triangle_vs.dxil' }
                @{ Name = 'ps_main'; Stage = 'fragment'; Profile = 'ps_6_0'; VisibleBindings = @(); SpirV = 'assets/01_triangle/triangle_ps.spv'; Dxil = 'assets/01_triangle/triangle_ps.dxil' }
            )
        }
        @{
            Name = '02_texture'
            Source = 'assets/02_texture/texture.wgsl'
            VertexLocations = @(0, 1, 2)
            Bindings = @(
                @{ Kind = 'Uniform'; Group = 0; Binding = 0 }
                @{ Kind = 'Texture'; Group = 0; Binding = 1 }
                @{ Kind = 'Sampler'; Group = 0; Binding = 2 }
            )
            Stages = @(
                @{ Name = 'vs_main'; Stage = 'vertex'; Profile = 'vs_6_0'; VisibleBindings = @(0); SpirV = 'assets/02_texture/texture.vs.spv'; Dxil = 'assets/02_texture/texture.vs.dxil' }
                @{ Name = 'fs_main'; Stage = 'fragment'; Profile = 'ps_6_0'; VisibleBindings = @(1, 2); SpirV = 'assets/02_texture/texture.ps.spv'; Dxil = 'assets/02_texture/texture.ps.dxil' }
            )
        }
        @{
            Name = '03_instancing'
            Source = 'assets/03_instancing/rock.wgsl'
            VertexLocations = @()
            Bindings = @()
            Stages = @(
                @{ Name = 'rock_vs'; Source = 'assets/03_instancing/rock.wgsl'; Stage = 'vertex'; Profile = 'vs_6_0'; VertexLocations = @(0, 1, 2, 3, 4, 5, 6, 7); Bindings = @(@{ Kind = 'Uniform'; Group = 0; Binding = 0 }); VisibleBindings = @(0); SpirV = 'assets/03_instancing/rock_vs.spv'; Dxil = 'assets/03_instancing/rock_vs.dxil' }
                @{ Name = 'rock_ps'; Source = 'assets/03_instancing/rock.wgsl'; Stage = 'fragment'; Profile = 'ps_6_0'; Bindings = @(@{ Kind = 'Uniform'; Group = 0; Binding = 0 }, @{ Kind = 'Texture'; Group = 0; Binding = 1 }, @{ Kind = 'Sampler'; Group = 0; Binding = 2 }); VisibleBindings = @(1, 2); SpirV = 'assets/03_instancing/rock_ps.spv'; Dxil = 'assets/03_instancing/rock_ps.dxil' }
                @{ Name = 'planet_vs'; Source = 'assets/03_instancing/planet.wgsl'; Stage = 'vertex'; Profile = 'vs_6_0'; VertexLocations = @(0, 1, 2, 3); Bindings = @(@{ Kind = 'Uniform'; Group = 0; Binding = 0 }); VisibleBindings = @(0); SpirV = 'assets/03_instancing/planet_vs.spv'; Dxil = 'assets/03_instancing/planet_vs.dxil' }
                @{ Name = 'planet_ps'; Source = 'assets/03_instancing/planet.wgsl'; Stage = 'fragment'; Profile = 'ps_6_0'; Bindings = @(@{ Kind = 'Uniform'; Group = 0; Binding = 0 }, @{ Kind = 'Texture'; Group = 0; Binding = 1 }, @{ Kind = 'Sampler'; Group = 0; Binding = 2 }); VisibleBindings = @(1, 2); SpirV = 'assets/03_instancing/planet_ps.spv'; Dxil = 'assets/03_instancing/planet_ps.dxil' }
                @{ Name = 'star_vs'; Source = 'assets/03_instancing/stars.wgsl'; Stage = 'vertex'; Profile = 'vs_6_0'; VertexLocations = @(); Bindings = @(); VisibleBindings = @(); SpirV = 'assets/03_instancing/star_vs.spv'; Dxil = 'assets/03_instancing/star_vs.dxil' }
                @{ Name = 'star_ps'; Source = 'assets/03_instancing/stars.wgsl'; Stage = 'fragment'; Profile = 'ps_6_0'; Bindings = @(); VisibleBindings = @(); SpirV = 'assets/03_instancing/star_ps.spv'; Dxil = 'assets/03_instancing/star_ps.dxil' }
            )
        }
        @{
            Name = '04_dynamic_uniform_buffer'
            Source = 'assets/04_dynamic_uniform_buffer/dynamic_uniform.wgsl'
            VertexLocations = @(0, 1)
            Bindings = @(
                @{ Kind = 'Uniform'; Group = 0; Binding = 0 }
                @{ Kind = 'Uniform'; Group = 0; Binding = 1 }
            )
            Stages = @(
                @{ Name = 'vs_main'; Stage = 'vertex'; Profile = 'vs_6_0'; VisibleBindings = @(0, 1); SpirV = 'assets/04_dynamic_uniform_buffer/dynamic_uniform.vs.spv'; Dxil = 'assets/04_dynamic_uniform_buffer/dynamic_uniform.vs.dxil' }
                @{ Name = 'fs_main'; Stage = 'fragment'; Profile = 'ps_6_0'; VisibleBindings = @(); SpirV = 'assets/04_dynamic_uniform_buffer/dynamic_uniform.ps.spv'; Dxil = 'assets/04_dynamic_uniform_buffer/dynamic_uniform.ps.dxil' }
            )
        }
        @{
            Name = '05_push_constants'
            Source = 'assets/05_push_constants/push_constants.wgsl'
            VertexLocations = @(0, 1, 2)
            Bindings = @(
                @{ Kind = 'Uniform'; Group = 0; Binding = 0 }
                @{ Kind = 'Uniform'; Group = 1; Binding = 0 }
            )
            Stages = @(
                @{ Name = 'vs_main'; Stage = 'vertex'; Profile = 'vs_6_0'; VisibleBindings = @(0); NativeHlsl = @{ SemanticLocations = @(0, 1, 2) }; NativePushConstant = @{ Group = 1; Binding = 0; Register = 1; Space = 0; Bytes = 32; SemanticLocations = @(0, 1, 2) }; SpirV = 'assets/05_push_constants/push_constants.vs.spv'; Dxil = 'assets/05_push_constants/push_constants.vs.dxil' }
                @{ Name = 'fs_main'; Stage = 'fragment'; Profile = 'ps_6_0'; VisibleBindings = @(); NativeHlsl = @{ SemanticLocations = @(0) }; SpirV = 'assets/05_push_constants/push_constants.ps.spv'; Dxil = 'assets/05_push_constants/push_constants.ps.dxil' }
            )
        }
        @{
            Name = '06_compute'
            Source = 'assets/06_compute/display.wgsl'
            VertexLocations = @()
            Bindings = @()
            Stages = @(
                @{ Name = 'emboss'; Source = 'assets/06_compute/filters.wgsl'; Stage = 'compute'; Profile = 'cs_6_0'; Bindings = @(@{ Kind = 'Texture'; Group = 0; Binding = 0 }, @{ Kind = 'StorageTexture'; Group = 0; Binding = 1 }); VisibleBindings = @(0, 1); SpirV = 'assets/06_compute/emboss.spv'; Dxil = 'assets/06_compute/emboss.dxil' }
                @{ Name = 'edgedetect'; Source = 'assets/06_compute/filters.wgsl'; Stage = 'compute'; Profile = 'cs_6_0'; Bindings = @(@{ Kind = 'Texture'; Group = 0; Binding = 0 }, @{ Kind = 'StorageTexture'; Group = 0; Binding = 1 }); VisibleBindings = @(0, 1); SpirV = 'assets/06_compute/edgedetect.spv'; Dxil = 'assets/06_compute/edgedetect.dxil' }
                @{ Name = 'sharpen'; Source = 'assets/06_compute/filters.wgsl'; Stage = 'compute'; Profile = 'cs_6_0'; Bindings = @(@{ Kind = 'Texture'; Group = 0; Binding = 0 }, @{ Kind = 'StorageTexture'; Group = 0; Binding = 1 }); VisibleBindings = @(0, 1); SpirV = 'assets/06_compute/sharpen.spv'; Dxil = 'assets/06_compute/sharpen.dxil' }
                @{ Name = 'vs_main'; Source = 'assets/06_compute/display.wgsl'; Stage = 'vertex'; Profile = 'vs_6_0'; VertexLocations = @(0, 1); Bindings = @(); VisibleBindings = @(); SpirV = 'assets/06_compute/display_vs.spv'; Dxil = 'assets/06_compute/display_vs.dxil' }
                @{ Name = 'ps_main'; Source = 'assets/06_compute/display.wgsl'; Stage = 'fragment'; Profile = 'ps_6_0'; Bindings = @(@{ Kind = 'Texture'; Group = 0; Binding = 1 }, @{ Kind = 'Sampler'; Group = 0; Binding = 2 }); VisibleBindings = @(0, 1); SpirV = 'assets/06_compute/display_ps.spv'; Dxil = 'assets/06_compute/display_ps.dxil' }
            )
        }
        @{
            Name = '07_offscreen'
            Source = 'assets/07_offscreen/phong.wgsl'
            VertexLocations = @()
            Bindings = @()
            Stages = @(
                @{ Name = 'phong_vs'; Source = 'assets/07_offscreen/phong.wgsl'; Stage = 'vertex'; Profile = 'vs_6_0'; VertexLocations = @(0, 1, 2, 3); Bindings = @(@{ Kind = 'Uniform'; Group = 0; Binding = 0 }); VisibleBindings = @(0); SpirV = 'assets/07_offscreen/phong.vs.spv'; Dxil = 'assets/07_offscreen/phong.vs.dxil' }
                @{ Name = 'phong_ps'; Source = 'assets/07_offscreen/phong.wgsl'; Stage = 'fragment'; Profile = 'ps_6_0'; Bindings = @(); VisibleBindings = @(); SpirV = 'assets/07_offscreen/phong.ps.spv'; Dxil = 'assets/07_offscreen/phong.ps.dxil' }
                @{ Name = 'mirror_vs'; Source = 'assets/07_offscreen/mirror.wgsl'; Stage = 'vertex'; Profile = 'vs_6_0'; VertexLocations = @(0); Bindings = @(@{ Kind = 'Uniform'; Group = 0; Binding = 0 }); VisibleBindings = @(0); SpirV = 'assets/07_offscreen/mirror.vs.spv'; Dxil = 'assets/07_offscreen/mirror.vs.dxil' }
                @{ Name = 'mirror_ps'; Source = 'assets/07_offscreen/mirror.wgsl'; Stage = 'fragment'; Profile = 'ps_6_0'; Bindings = @(@{ Kind = 'Uniform'; Group = 0; Binding = 0 }, @{ Kind = 'Texture'; Group = 0; Binding = 1 }, @{ Kind = 'Sampler'; Group = 0; Binding = 2 }); VisibleBindings = @(1, 2); SpirV = 'assets/07_offscreen/mirror.ps.spv'; Dxil = 'assets/07_offscreen/mirror.ps.dxil' }
            )
        }
        @{
            Name = '08_msaa'
            Source = 'assets/08_msaa/msaa.wgsl'
            VertexLocations = @(0, 1, 2, 3)
            Bindings = @(
                @{ Kind = 'Uniform'; Group = 0; Binding = 0 }
                @{ Kind = 'Texture'; Group = 0; Binding = 1 }
                @{ Kind = 'Sampler'; Group = 0; Binding = 2 }
            )
            Stages = @(
                @{ Name = 'vs_main'; Stage = 'vertex'; Profile = 'vs_6_0'; VisibleBindings = @(0); SpirV = 'assets/08_msaa/msaa.vs.spv'; Dxil = 'assets/08_msaa/msaa.vs.dxil' }
                @{ Name = 'ps_main'; Stage = 'fragment'; Profile = 'ps_6_0'; VisibleBindings = @(1, 2); SpirV = 'assets/08_msaa/msaa.ps.spv'; Dxil = 'assets/08_msaa/msaa.ps.dxil' }
            )
        }
        @{
            Name = '09_texture_mipmap'
            Source = 'assets/09_texture_mipmap/tunnel.wgsl'
            VertexLocations = @()
            Bindings = @()
            Stages = @(
                @{ Name = 'vs_main'; Source = 'assets/09_texture_mipmap/tunnel.wgsl'; Stage = 'vertex'; Profile = 'vs_6_0'; VertexLocations = @(0, 1, 2); Bindings = @(@{ Kind = 'Uniform'; Group = 0; Binding = 0 }); VisibleBindings = @(0); SpirV = 'assets/09_texture_mipmap/tunnel.vs.spv'; Dxil = 'assets/09_texture_mipmap/tunnel.vs.dxil' }
                @{ Name = 'ps_main'; Source = 'assets/09_texture_mipmap/tunnel.wgsl'; Stage = 'fragment'; Profile = 'ps_6_0'; Bindings = @(@{ Kind = 'Uniform'; Group = 0; Binding = 0 }, @{ Kind = 'Texture'; Group = 0; Binding = 1 }, @{ Kind = 'Sampler'; Group = 0; Binding = 2 }); VisibleBindings = @(1, 2); SpirV = 'assets/09_texture_mipmap/tunnel.ps.spv'; Dxil = 'assets/09_texture_mipmap/tunnel.ps.dxil' }
            )
        }
        @{
            Name = '10_indirect_draw'
            Source = 'assets/10_indirect_draw/plants.wgsl'
            VertexLocations = @()
            Bindings = @()
            Stages = @(
                @{ Name = 'plants_vs'; Source = 'assets/10_indirect_draw/plants.wgsl'; Stage = 'vertex'; Profile = 'vs_6_0'; VertexLocations = @(0, 1, 2, 3, 4, 5, 6, 7); Bindings = @(@{ Kind = 'Uniform'; Group = 0; Binding = 0 }); VisibleBindings = @(0); SpirV = 'assets/10_indirect_draw/plants_vs.spv'; Dxil = 'assets/10_indirect_draw/plants_vs.dxil' }
                @{ Name = 'plants_ps'; Source = 'assets/10_indirect_draw/plants.wgsl'; Stage = 'fragment'; Profile = 'ps_6_0'; Bindings = @(@{ Kind = 'Uniform'; Group = 0; Binding = 0 }, @{ Kind = 'Texture'; Group = 0; Binding = 1 }, @{ Kind = 'Sampler'; Group = 0; Binding = 2 }); VisibleBindings = @(1, 2); SpirV = 'assets/10_indirect_draw/plants_ps.spv'; Dxil = 'assets/10_indirect_draw/plants_ps.dxil' }
                @{ Name = 'ground_vs'; Source = 'assets/10_indirect_draw/ground.wgsl'; Stage = 'vertex'; Profile = 'vs_6_0'; VertexLocations = @(0, 1, 2); Bindings = @(@{ Kind = 'Uniform'; Group = 0; Binding = 0 }); VisibleBindings = @(0); SpirV = 'assets/10_indirect_draw/ground_vs.spv'; Dxil = 'assets/10_indirect_draw/ground_vs.dxil' }
                @{ Name = 'ground_ps'; Source = 'assets/10_indirect_draw/ground.wgsl'; Stage = 'fragment'; Profile = 'ps_6_0'; Bindings = @(@{ Kind = 'Uniform'; Group = 0; Binding = 0 }, @{ Kind = 'Texture'; Group = 0; Binding = 1 }, @{ Kind = 'Sampler'; Group = 0; Binding = 2 }); VisibleBindings = @(1, 2); SpirV = 'assets/10_indirect_draw/ground_ps.spv'; Dxil = 'assets/10_indirect_draw/ground_ps.dxil' }
                @{ Name = 'sky_vs'; Source = 'assets/10_indirect_draw/sky.wgsl'; Stage = 'vertex'; Profile = 'vs_6_0'; VertexLocations = @(0, 2); Bindings = @(@{ Kind = 'Uniform'; Group = 0; Binding = 0 }); VisibleBindings = @(0); SpirV = 'assets/10_indirect_draw/sky_vs.spv'; Dxil = 'assets/10_indirect_draw/sky_vs.dxil' }
                @{ Name = 'sky_ps'; Source = 'assets/10_indirect_draw/sky.wgsl'; Stage = 'fragment'; Profile = 'ps_6_0'; Bindings = @(); VisibleBindings = @(); SpirV = 'assets/10_indirect_draw/sky_ps.spv'; Dxil = 'assets/10_indirect_draw/sky_ps.dxil' }
            )
        }
        @{
            Name = '11_occlusion_query'
            Source = 'assets/11_occlusion_query/occlusion.wgsl'
            VertexLocations = @(0, 1, 2, 3)
            Bindings = @(
                @{ Kind = 'Uniform'; Group = 0; Binding = 0 }
            )
            Stages = @(
                # The source declares UV at location 2 to share the glTF vertex ABI;
                # WGSL does not consume it, so DXIL correctly omits it from the input
                # signature. Flattening preserves the 256-byte WebGPU uniform layout
                # that SPIRV-Cross cannot express as an HLSL struct/packoffset pair.
                @{ Name = 'vs_main'; Stage = 'vertex'; Profile = 'vs_6_0'; VisibleBindings = @(0); ConsumedVertexLocations = @(0, 1, 3); HlslFlattenUbo = $true; SpirV = 'assets/11_occlusion_query/occlusion.vs.spv'; Dxil = 'assets/11_occlusion_query/occlusion.vs.dxil' }
                @{ Name = 'simple_ps'; Stage = 'fragment'; Profile = 'ps_6_0'; VisibleBindings = @(); SpirV = 'assets/11_occlusion_query/simple.ps.spv'; Dxil = 'assets/11_occlusion_query/simple.ps.dxil' }
                @{ Name = 'mesh_ps'; Stage = 'fragment'; Profile = 'ps_6_0'; VisibleBindings = @(); SpirV = 'assets/11_occlusion_query/mesh.ps.spv'; Dxil = 'assets/11_occlusion_query/mesh.ps.dxil' }
                @{ Name = 'occluder_ps'; Stage = 'fragment'; Profile = 'ps_6_0'; VisibleBindings = @(); SpirV = 'assets/11_occlusion_query/occluder.ps.spv'; Dxil = 'assets/11_occlusion_query/occluder.ps.dxil' }
            )
        }
        @{
            Name = '12_screenshot'
            Source = 'assets/12_screenshot/screenshot.wgsl'
            VertexLocations = @(0, 1, 2)
            Bindings = @(@{ Kind = 'Uniform'; Group = 0; Binding = 0 })
            Stages = @(
                @{ Name = 'vs_main'; Stage = 'vertex'; Profile = 'vs_6_0'; VisibleBindings = @(0); SpirV = 'assets/12_screenshot/screenshot.vs.spv'; Dxil = 'assets/12_screenshot/screenshot_vs.dxil' }
                @{ Name = 'ps_main'; Stage = 'fragment'; Profile = 'ps_6_0'; VisibleBindings = @(); SpirV = 'assets/12_screenshot/screenshot.ps.spv'; Dxil = 'assets/12_screenshot/screenshot_ps.dxil' }
            )
        }
        @{
            Name = '13_multithreaded_recording'
            Source = 'assets/13_multithreaded_recording/ufo.wgsl'
            VertexLocations = @(0, 1, 2, 3)
            Bindings = @(@{ Kind = 'Uniform'; Group = 0; Binding = 0 })
            Stages = @(
                @{ Name = 'ufo_vs'; Entry = 'vs_main'; Stage = 'vertex'; Profile = 'vs_6_0'; VertexLocations = @(0, 1, 2, 3); ConsumedVertexLocations = @(0, 1, 3); VisibleBindings = @(0); SpirV = 'assets/13_multithreaded_recording/ufo_vs.spv'; Dxil = 'assets/13_multithreaded_recording/ufo_vs.dxil' }
                @{ Name = 'ufo_ps'; Entry = 'ps_main'; Stage = 'fragment'; Profile = 'ps_6_0'; VisibleBindings = @(); SpirV = 'assets/13_multithreaded_recording/ufo_ps.spv'; Dxil = 'assets/13_multithreaded_recording/ufo_ps.dxil' }
                # The WebGPU shader keeps a uniform binding. Native secondary recording uses
                # vertex push constants instead, so lower this WGSL binding only for native artifacts.
                @{ Name = 'stars_vs'; Source = 'assets/13_multithreaded_recording/stars.wgsl'; Entry = 'vs_main'; Stage = 'vertex'; Profile = 'vs_6_0'; VertexLocations = @(0); ConsumedVertexLocations = @(0); Bindings = @(@{ Kind = 'Uniform'; Group = 0; Binding = 0 }); VisibleBindings = @(); NativeHlsl = @{ SemanticLocations = @(0) }; NativePushConstant = @{ Group = 0; Binding = 0; Register = 1; Space = 0; Bytes = 64; SemanticLocations = @(0) }; SpirV = 'assets/13_multithreaded_recording/stars_vs.spv'; Dxil = 'assets/13_multithreaded_recording/stars_vs.dxil' }
                # Keep both native stages on the same emitter so their user varying semantics link.
                @{ Name = 'stars_ps'; Source = 'assets/13_multithreaded_recording/stars.wgsl'; Entry = 'ps_main'; Stage = 'fragment'; Profile = 'ps_6_0'; Bindings = @(@{ Kind = 'Uniform'; Group = 0; Binding = 0 }); VisibleBindings = @(); NativeHlsl = @{ SemanticLocations = @(0) }; SpirV = 'assets/13_multithreaded_recording/stars_ps.spv'; Dxil = 'assets/13_multithreaded_recording/stars_ps.dxil' }
            )
        }
        @{
            Name = '14_descriptor_indexing'
            Source = 'assets/14_descriptor_indexing/descriptor_indexing_native.wgsl'
            VertexLocations = @(0, 1, 2)
            Bindings = @(
                @{ Kind = 'Uniform'; Group = 0; Binding = 0 }
                @{ Kind = 'Sampler'; Group = 0; Binding = 1 }
                @{ Kind = 'Texture'; Group = 0; Binding = 2 }
            )
            Stages = @(
                @{ Name = 'vs_main'; Stage = 'vertex'; Profile = 'vs_6_0'; VisibleBindings = @(0); SpirV = 'assets/14_descriptor_indexing/descriptor_indexing.vs.spv'; Dxil = 'assets/14_descriptor_indexing/descriptor_indexing.vs.dxil' }
                @{ Name = 'ps_main'; Stage = 'fragment'; Profile = 'ps_6_0'; VisibleBindings = @(1, 2); NativeDescriptorArray = @{ Binding = 2; Count = 32 }; SpirV = 'assets/14_descriptor_indexing/descriptor_indexing.ps.spv'; Dxil = 'assets/14_descriptor_indexing/descriptor_indexing.ps.dxil' }
            )
        }
        @{
            Name = '15_multiview'
            Source = 'assets/15_multiview/scene_native.wgsl'
            VertexLocations = @(0, 1, 2, 3)
            Bindings = @(@{ Kind = 'Uniform'; Group = 0; Binding = 0 })
            Stages = @(
                @{ Name = 'scene_vs'; Entry = 'vs_main'; Stage = 'vertex'; Profile = 'vs_6_1'; HlslShaderModel = 61; VisibleBindings = @(0); NativeViewIndex = $true; SpirV = 'assets/15_multiview/scene_vs.spv'; Dxil = 'assets/15_multiview/scene_vs.dxil' }
                @{ Name = 'scene_ps'; Entry = 'ps_main'; Stage = 'fragment'; Profile = 'ps_6_0'; VisibleBindings = @(); SpirV = 'assets/15_multiview/scene_ps.spv'; Dxil = 'assets/15_multiview/scene_ps.dxil' }
            )
        }
        @{
            Name = '15_multiview_display'
            Source = 'assets/15_multiview/display_native.wgsl'
            VertexLocations = @()
            Bindings = @(
                @{ Kind = 'Texture'; Group = 0; Binding = 0 }
                @{ Kind = 'Sampler'; Group = 0; Binding = 1 }
            )
            Stages = @(
                @{ Name = 'display_vs'; Entry = 'vs_main'; Stage = 'vertex'; Profile = 'vs_6_0'; VertexLocations = @(); Bindings = @(); VisibleBindings = @(); SpirV = 'assets/15_multiview/display_vs.spv'; Dxil = 'assets/15_multiview/display_vs.dxil' }
                @{ Name = 'display_ps'; Entry = 'ps_main'; Stage = 'fragment'; Profile = 'ps_6_0'; VisibleBindings = @(0, 1); SpirV = 'assets/15_multiview/display_ps.spv'; Dxil = 'assets/15_multiview/display_ps.dxil' }
            )
        }
    )
}
