//! Core resource compatibility and sampled view validation checks.
use super::*;

async fn binding_validation(d: &Device<Gl>) {
    let entry = |ty| BindGroupLayoutEntry {
        binding: 0,
        visibility: vec![ShaderStage::Fragment],
        ty,
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    };
    let layout = |ty| {
        d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![entry(ty)],
        })
        .unwrap()
    };
    let texture_type = |sample, multisampled| BindingType::SampledTexture {
        sample,
        dimension: TextureViewDimension::D2,
        multisampled,
    };
    let texture = |format, mips, samples, usage| {
        d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 1,
            format,
            mip_levels: mips,
            samples,
            usage,
            view_formats: vec![],
        })
        .unwrap()
    };
    let view = |t, format, mips| {
        d.create_view(
            t,
            &TextureViewDesc {
                format,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips,
                layers: 0..1,
            },
        )
    };
    let rgba = texture(
        TextureFormat::Rgba8Unorm,
        2,
        1,
        TextureUsage::TEXTURE_BINDING,
    );
    let sint = texture(
        TextureFormat::Rgba8Sint,
        1,
        1,
        TextureUsage::TEXTURE_BINDING,
    );
    let full = view(&rgba, TextureFormat::Rgba8Unorm, 0..2).unwrap();
    let signed = view(&sint, TextureFormat::Rgba8Sint, 0..1).unwrap();
    let uint_layout = layout(texture_type(TextureSampleType::Uint, false));
    let float_layout = layout(texture_type(TextureSampleType::Float, false));
    let group = |l: &BindGroupLayout<Gl>, v: &TextureView<Gl>| {
        d.create_bind_group(&BindGroupDesc {
            label: None,
            layout: l,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(v),
            }],
        })
    };
    assert!(
        group(&uint_layout, &full).is_err(),
        "Uint layout must reject unorm"
    );
    assert!(
        group(&uint_layout, &signed).is_err(),
        "Uint layout must reject Sint"
    );
    assert!(
        group(&float_layout, &full).is_ok(),
        "full mip range must work"
    );
    // Partial sampled views are covered by the native view readback fixture:
    // contexts with real texture views can now bind them independently.
    let msaa_layout = layout(texture_type(TextureSampleType::Float, true));
    assert!(
        group(&msaa_layout, &full).is_err(),
        "sample count mismatch must be rejected"
    );
    let sampler = |filter, compare| {
        d.create_sampler(&SamplerDesc {
            label: None,
            address: [AddressMode::ClampToEdge; 3],
            mag_filter: filter,
            min_filter: filter,
            mip_filter: FilterMode::Nearest,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare,
            border: None,
        })
        .unwrap()
    };
    let nearest = sampler(FilterMode::Nearest, None);
    let linear = sampler(FilterMode::Linear, None);
    let compare = sampler(FilterMode::Nearest, Some(CompareFunction::Less));
    let ordinary = layout(BindingType::Sampler {
        filtering: false,
        comparison: false,
    });
    let comparison = layout(BindingType::Sampler {
        filtering: false,
        comparison: true,
    });
    let bind_sampler = |l: &BindGroupLayout<Gl>, s: &Sampler<Gl>| {
        d.create_bind_group(&BindGroupDesc {
            label: None,
            layout: l,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Sampler(s),
            }],
        })
    };
    assert!(bind_sampler(&ordinary, &nearest).is_ok());
    assert!(
        bind_sampler(&ordinary, &linear).is_err(),
        "nonfiltering must reject linear"
    );
    assert!(
        bind_sampler(&ordinary, &compare).is_err(),
        "ordinary sampler must reject compare"
    );
    assert!(
        bind_sampler(&comparison, &nearest).is_err(),
        "comparison sampler must reject ordinary"
    );
    assert!(bind_sampler(&comparison, &compare).is_ok());
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_binding_validation() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let d = block_on(open(native));
    block_on(binding_validation(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_binding_validation() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(binding_validation(&d));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_binding_validation() {
    let d = webgl2_device([8, 8]).await;
    binding_validation(&d).await;
}
