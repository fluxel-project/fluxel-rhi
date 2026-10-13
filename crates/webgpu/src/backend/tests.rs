//! Real browser bootstrap evidence for the historical reflective WebGPU path.

use wasm_bindgen_futures::JsFuture;
use wasm_bindgen_test::*;

use super::js;
use super::{WebGpu, WebGpuInstance};
use fluxel_rhi_core::api::{
    BindGroupLayoutDesc, BindGroupLayoutEntry, BindingCount, BindingType, BufferDesc, BufferUsage,
    ColorSpace, ColorTargetState, CompositeAlphaMode, CullMode, FrontFace, MemoryClass,
    MultisampleState, PipelineLayoutDesc, PipelineStage, PolygonMode, PresentMode, PrimitiveState,
    PrimitiveTopology, RasterPipelineDesc, ShaderDesc, ShaderEntryPoint, ShaderFormat, ShaderStage,
    StorageAccess, Support, SurfaceConfig, SurfaceUsage, TextureAspects, TextureDesc,
    TextureDimension, TextureFormat, TextureUsage, TextureViewDesc, TextureViewDimension,
    VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode,
};
use std::sync::Arc;
use wasm_bindgen::JsCast;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn browser_opens_a_webgpu_device() {
    let gpu = js::browser_gpu().expect("navigator.gpu");
    let adapter = JsFuture::from(js::request_adapter(&gpu, None).expect("requestAdapter call"))
        .await
        .expect("requestAdapter promise");
    assert!(!adapter.is_null(), "browser reported no WebGPU adapter");
    let device =
        JsFuture::from(js::request_device(&adapter, &[], &[]).expect("requestDevice call"))
            .await
            .expect("requestDevice promise");
    assert!(!device.is_null(), "browser reported no WebGPU device");
    js::queue(&device).expect("GPUQueue");
}

#[wasm_bindgen_test(async)]
async fn core_instance_opens_typed_webgpu_device() {
    let device = WebGpuInstance::new()
        .expect("navigator.gpu")
        .open_default()
        .await
        .expect("typed WebGPU device");
    assert_eq!(device.lanes().len(), 1);
    assert!(device.capabilities().limits.max_texture_dimension_2d >= 2048);
}

#[wasm_bindgen_test(async)]
async fn typed_webgpu_resources_are_created_in_browser() {
    let device = WebGpuInstance::new().unwrap().open_default().await.unwrap();
    let buffer = device
        .create_buffer(&BufferDesc {
            label: Some("test vertex buffer".into()),
            size: 72,
            usage: BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })
        .unwrap();
    let texture = device
        .create_texture(&TextureDesc {
            label: Some("test render texture".into()),
            dimension: TextureDimension::D2,
            extent: [64, 64, 1],
            array_layers: 1,
            format: TextureFormat::Bgra8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })
        .unwrap();
    let view = device
        .create_view(
            &texture,
            &TextureViewDesc {
                format: TextureFormat::Bgra8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    drop((buffer, view));
}

#[wasm_bindgen_test(async)]
async fn invalid_texture_and_view_descriptors_fail_before_browser_validation() {
    let device = WebGpuInstance::new().unwrap().open_default().await.unwrap();
    let invalid_mips = TextureDesc {
        label: Some("invalid mip count".into()),
        dimension: TextureDimension::D2,
        extent: [1, 1, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 2,
        samples: 1,
        usage: TextureUsage::TEXTURE_BINDING,
        view_formats: vec![],
    };
    assert!(matches!(
        device.texture_support(&invalid_mips),
        Support::Unsupported(_)
    ));
    assert!(device.create_texture(&invalid_mips).is_err());
    let invalid_msaa = TextureDesc {
        label: Some("MSAA without render attachment".into()),
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 4,
        usage: TextureUsage::TEXTURE_BINDING,
        view_formats: vec![],
    };
    assert!(matches!(
        device.texture_support(&invalid_msaa),
        Support::Unsupported(_)
    ));
    assert!(device.create_texture(&invalid_msaa).is_err());
    let invalid_view_format = TextureDesc {
        label: Some("incompatible view format".into()),
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::TEXTURE_BINDING,
        view_formats: vec![TextureFormat::Bgra8Unorm],
    };
    assert!(matches!(
        device.texture_support(&invalid_view_format),
        Support::Unsupported(_)
    ));

    let texture = device
        .create_texture(&TextureDesc {
            label: Some("single mip texture".into()),
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::TEXTURE_BINDING,
            view_formats: vec![],
        })
        .unwrap();
    assert!(
        device
            .create_view(
                &texture,
                &TextureViewDesc {
                    format: TextureFormat::Rgba8Unorm,
                    dimension: TextureViewDimension::D2,
                    aspects: TextureAspects::COLOR,
                    mips: 1..2,
                    layers: 0..1,
                },
            )
            .is_err()
    );
}

#[wasm_bindgen_test(async)]
async fn standard_webgpu_rejects_readable_storage_texture_layouts() {
    let device = WebGpuInstance::new().unwrap().open_default().await.unwrap();
    let result = device.create_bind_group_layout(&BindGroupLayoutDesc {
        label: Some("unsupported read storage texture".into()),
        entries: vec![BindGroupLayoutEntry {
            binding: 0,
            visibility: vec![ShaderStage::Compute],
            ty: BindingType::StorageTexture {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                access: StorageAccess::Read,
            },
            count: BindingCount::One,
            partially_bound: false,
            nonuniform_indexing: false,
        }],
    });
    assert!(result.is_err());
}

#[wasm_bindgen_test(async)]
async fn typed_wgsl_triangle_pipeline_compiles_in_browser() {
    let device = WebGpuInstance::new().unwrap().open_default().await.unwrap();
    let module = device
        .shader(ShaderDesc {
            label: Some("triangle WGSL".into()),
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(
                r#"
struct VertexOut { @builtin(position) position: vec4f, @location(0) color: vec3f };
@vertex fn vs_main(@location(0) position: vec3f, @location(1) color: vec3f) -> VertexOut {
    var output: VertexOut;
    output.position = vec4f(position, 1.0);
    output.color = color;
    return output;
}

@fragment fn ps_main(input: VertexOut) -> @location(0) vec4f {
    return vec4f(input.color, 1.0);
}
"#
                .as_bytes(),
            ),
            entries: vec![
                ShaderEntryPoint {
                    name: "vs_main".into(),
                    stage: ShaderStage::Vertex,
                    workgroup_size: None,
                },
                ShaderEntryPoint {
                    name: "ps_main".into(),
                    stage: ShaderStage::Fragment,
                    workgroup_size: None,
                },
            ],
            required_features: vec![],
        })
        .await
        .unwrap();
    let layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("triangle layout"),
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let _pipeline = device
        .raster_pipeline(RasterPipelineDesc {
            label: Some("triangle pipeline".into()),
            layout,
            vertex: PipelineStage {
                module: module.clone(),
                entry: "vs_main".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module,
                entry: "ps_main".into(),
                constants: vec![],
            }),
            vertex_buffers: vec![VertexBufferLayout {
                stride: 24,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![
                    VertexAttribute {
                        location: 0,
                        offset: 0,
                        format: VertexFormat::Float32(3),
                    },
                    VertexAttribute {
                        location: 1,
                        offset: 12,
                        format: VertexFormat::Float32(3),
                    },
                ],
            }],
            color_targets: vec![Some(ColorTargetState {
                format: TextureFormat::Bgra8Unorm,
                blend: None,
                write_mask: 0xf,
            })],
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: u32::MAX.into(),
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
        .await
        .unwrap();
}

#[wasm_bindgen_test(async)]
async fn canvas_lease_is_exclusive_and_generation_checked() {
    let device = WebGpuInstance::new().unwrap().open_default().await.unwrap();
    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(64);
    canvas.set_height(64);
    let mut surface = WebGpu::surface_from_canvas(&device, canvas).unwrap();
    let config = SurfaceConfig {
        extent: [64, 64],
        format: surface.capabilities().unwrap().formats[0],
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    surface.configure(&config).await.unwrap();
    let image = surface.acquire().await.unwrap();
    assert!(image.is_current());
    assert!(surface.acquire().await.is_err());
    assert!(surface.configure(&config).await.is_err());
    drop(image);
    assert!(surface.acquire().await.is_err());
    surface.configure(&config).await.unwrap();
    assert!(surface.acquire().await.unwrap().is_current());
}
