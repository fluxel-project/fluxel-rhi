//! Per-sample combined depth/stencil copy output assertions.
use super::*;

#[wasm_bindgen_test(async)]
async fn gpu_msaa_combined_depth_stencil_copy_preserves_every_sample() {
    use js_sys::{Array, Object, Promise, Uint32Array};
    use wasm_bindgen::JsValue;
    use wasm_bindgen_futures::JsFuture;
    let d = open().await;
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [2, 2, 1],
        array_layers: 1,
        format: TextureFormat::Depth32FloatStencil8,
        mip_levels: 1,
        samples: 4,
        usage: TextureUsage::COPY_SRC
            | TextureUsage::COPY_DST
            | TextureUsage::RENDER_ATTACHMENT
            | TextureUsage::TEXTURE_BINDING,
        view_formats: vec![],
    };
    if !matches!(d.texture_support(&desc), Support::Supported(_)) {
        console_log!("SKIP multisampled depth32float-stencil8");
        return;
    }
    let source = d.create_texture(&desc).unwrap();
    let depth_target = d.create_texture(&desc).unwrap();
    let stencil_target = d.create_texture(&desc).unwrap();
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    for (texture, depth, stencil) in [
        (&source, 0.25, 37),
        (&depth_target, 0.75, 91),
        (&stencil_target, 0.75, 91),
    ] {
        let view = d
            .create_view(
                texture,
                &TextureViewDesc {
                    format: desc.format,
                    dimension: TextureViewDimension::D2,
                    aspects: TextureAspects::DEPTH | TextureAspects::STENCIL,
                    mips: 0..1,
                    layers: 0..1,
                },
            )
            .unwrap();
        e.begin_raster(&RasterPassDesc {
            label: None,
            colors: &[],
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&view),
                depth: Some((LoadOp::Clear(depth), StoreOp::Store)),
                stencil: Some((LoadOp::Clear(stencil), StoreOp::Store)),
                depth_read_only: false,
                stencil_read_only: false,
            }),
            render_area: Rect {
                x: 0,
                y: 0,
                width: 2,
                height: 2,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap()
        .finish()
        .unwrap();
    }
    submit(&d, e).await;
    seed_distinct_msaa_samples(&d, &source);
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    for (target, aspect) in [
        (&depth_target, TextureAspects::DEPTH),
        (&stencil_target, TextureAspects::STENCIL),
    ] {
        e.copy_texture(
            TextureRegion {
                texture: &source,
                mip: 0,
                origin: [0, 0, 0],
                extent: [2, 2, 1],
                aspect,
            },
            TextureRegion {
                texture: target,
                mip: 0,
                origin: [0, 0, 0],
                extent: [2, 2, 1],
                aspect,
            },
        )
        .unwrap();
    }
    submit(&d, e).await;
    // SAFETY: the owning device and textures remain alive through native submission and map.
    let native = unsafe { d.native_ref() };
    let dev = &native.0.device;
    let set = |o: &JsValue, k: &str, v: JsValue| super::super::js::set(o, k, &v).unwrap();
    let obj = || -> JsValue { Object::new().into() };
    let call = |o: &JsValue, n: &str, args: &[JsValue]| super::super::js::call(o, n, args).unwrap();
    let num = |n: u32| JsValue::from_f64(n as f64);
    let shader = "@group(0) @binding(0) var d0:texture_depth_multisampled_2d; @group(0) @binding(1) var s0:texture_multisampled_2d<u32>; @group(0) @binding(2) var d1:texture_depth_multisampled_2d; @group(0) @binding(3) var s1:texture_multisampled_2d<u32>; @group(0) @binding(4) var<storage,read_write> out:array<u32>; @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3<u32>){let s=i32(id.x); out[id.x]=bitcast<u32>(textureLoad(d0,vec2<i32>(0,0),s)); out[4u+id.x]=textureLoad(s0,vec2<i32>(0,0),s).x; out[8u+id.x]=bitcast<u32>(textureLoad(d1,vec2<i32>(0,0),s)); out[12u+id.x]=textureLoad(s1,vec2<i32>(0,0),s).x;}";
    let md = obj();
    set(&md, "code", JsValue::from_str(shader));
    let module = call(dev, "createShaderModule", &[md]);
    let stage = obj();
    set(&stage, "module", module);
    set(&stage, "entryPoint", JsValue::from_str("main"));
    let pd = obj();
    set(&pd, "layout", JsValue::from_str("auto"));
    set(&pd, "compute", stage);
    let pipe = call(dev, "createComputePipeline", &[pd]);
    let layout = call(&pipe, "getBindGroupLayout", &[num(0)]);
    let bd = obj();
    set(&bd, "size", num(64));
    set(&bd, "usage", num(128 | 4));
    let output = call(dev, "createBuffer", &[bd]);
    let rd = obj();
    set(&rd, "size", num(64));
    set(&rd, "usage", num(1 | 8));
    let readback = call(dev, "createBuffer", &[rd]);
    let entries = Array::new();
    for (index, texture, aspect) in [
        (0, &depth_target, "depth-only"),
        (1, &depth_target, "stencil-only"),
        (2, &stencil_target, "depth-only"),
        (3, &stencil_target, "stencil-only"),
    ] {
        // SAFETY: texture belongs to this device and remains live through submission.
        let raw = &unsafe { texture.native_ref() }.raw;
        let vd = obj();
        set(&vd, "dimension", JsValue::from_str("2d"));
        set(&vd, "aspect", JsValue::from_str(aspect));
        let view = call(raw, "createView", &[vd]);
        let entry = obj();
        set(&entry, "binding", num(index));
        set(&entry, "resource", view);
        entries.push(&entry);
    }
    let entry = obj();
    set(&entry, "binding", num(4));
    let resource = obj();
    set(&resource, "buffer", output.clone());
    set(&entry, "resource", resource);
    entries.push(&entry);
    let gd = obj();
    set(&gd, "layout", layout);
    set(&gd, "entries", entries.into());
    let group = call(dev, "createBindGroup", &[gd]);
    let encoder = call(dev, "createCommandEncoder", &[]);
    let pass = call(&encoder, "beginComputePass", &[obj()]);
    call(&pass, "setPipeline", &[pipe]);
    call(&pass, "setBindGroup", &[num(0), group]);
    call(&pass, "dispatchWorkgroups", &[num(4)]);
    call(&pass, "end", &[]);
    call(
        &encoder,
        "copyBufferToBuffer",
        &[output, num(0), readback.clone(), num(0), num(64)],
    );
    let command = call(&encoder, "finish", &[]);
    let commands = Array::new();
    commands.push(&command);
    call(&native.0.queue, "submit", &[commands.into()]);
    JsFuture::from(Promise::from(call(&readback, "mapAsync", &[num(1)])))
        .await
        .unwrap();
    let data = Uint32Array::new(&call(&readback, "getMappedRange", &[])).to_vec();
    call(&readback, "unmap", &[]);
    for sample in 0..4 {
        assert_eq!(f32::from_bits(data[sample]), (sample + 1) as f32 / 8.0);
        assert_eq!(data[4 + sample], 91);
        assert_eq!(f32::from_bits(data[8 + sample]), 0.75);
        assert_eq!(data[12 + sample], 37 + sample as u32);
    }
}

fn seed_distinct_msaa_samples(d: &Device<WebGpu>, texture: &Texture<WebGpu>) {
    use js_sys::{Array, Object, Uint32Array};
    use wasm_bindgen::JsValue;
    // SAFETY: device and texture are owned by the fixture and remain alive.
    let device = unsafe { d.native_ref() };
    let raw = unsafe { texture.native_ref() }.raw.clone();
    let native = &device.0.device;
    let obj = || -> JsValue { Object::new().into() };
    let num = |v: u32| JsValue::from_f64(v as f64);
    let set = |o: &JsValue, k: &str, v: JsValue| super::super::js::set(o, k, &v).unwrap();
    let call = |o: &JsValue, n: &str, args: &[JsValue]| super::super::js::call(o, n, args).unwrap();
    let code = "@group(0) @binding(0) var<uniform> selected:u32; @vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4<f32>{let p=vec2<f32>(f32((i<<1u)&2u),f32(i&2u));return vec4<f32>(p*vec2<f32>(2.,-2.)+vec2<f32>(-1.,1.),0.,1.);} @fragment fn fs(@builtin(sample_index) sample:u32)->@builtin(frag_depth) f32 {if(sample!=selected){discard;}return f32(sample+1u)/8.;}";
    let md = obj();
    set(&md, "code", JsValue::from_str(code));
    let module = call(native, "createShaderModule", &[md]);
    let vs = obj();
    set(&vs, "module", module.clone());
    set(&vs, "entryPoint", JsValue::from_str("vs"));
    let fs = obj();
    set(&fs, "module", module);
    set(&fs, "entryPoint", JsValue::from_str("fs"));
    set(&fs, "targets", Array::new().into());
    let stencil = obj();
    set(&stencil, "compare", JsValue::from_str("always"));
    set(&stencil, "passOp", JsValue::from_str("replace"));
    let ds = obj();
    set(&ds, "format", JsValue::from_str("depth32float-stencil8"));
    set(&ds, "depthWriteEnabled", JsValue::TRUE);
    set(&ds, "depthCompare", JsValue::from_str("always"));
    set(&ds, "stencilFront", stencil.clone());
    set(&ds, "stencilBack", stencil);
    let ms = obj();
    set(&ms, "count", num(4));
    let pd = obj();
    set(&pd, "layout", JsValue::from_str("auto"));
    set(&pd, "vertex", vs);
    set(&pd, "fragment", fs);
    set(&pd, "depthStencil", ds);
    set(&pd, "multisample", ms);
    let pipeline = call(native, "createRenderPipeline", &[pd]);
    let layout = call(&pipeline, "getBindGroupLayout", &[num(0)]);
    let bd = obj();
    set(&bd, "size", num(1024));
    set(&bd, "usage", num(64));
    set(&bd, "mappedAtCreation", JsValue::TRUE);
    let uniform = call(native, "createBuffer", &[bd]);
    let range = call(&uniform, "getMappedRange", &[]);
    let words = Uint32Array::new(&range);
    for i in 0..4 {
        words.set_index(i * 64, i);
    }
    call(&uniform, "unmap", &[]);
    let view = call(&raw, "createView", &[]);
    let attachment = obj();
    set(&attachment, "view", view);
    set(&attachment, "depthLoadOp", JsValue::from_str("load"));
    set(&attachment, "depthStoreOp", JsValue::from_str("store"));
    set(&attachment, "stencilLoadOp", JsValue::from_str("load"));
    set(&attachment, "stencilStoreOp", JsValue::from_str("store"));
    let passdesc = obj();
    set(&passdesc, "colorAttachments", Array::new().into());
    set(&passdesc, "depthStencilAttachment", attachment);
    let encoder = call(native, "createCommandEncoder", &[]);
    let pass = call(&encoder, "beginRenderPass", &[passdesc]);
    call(&pass, "setPipeline", &[pipeline]);
    for i in 0..4u32 {
        let entry = obj();
        set(&entry, "binding", num(0));
        let resource = obj();
        set(&resource, "buffer", uniform.clone());
        set(&resource, "offset", num(i * 256));
        set(&resource, "size", num(4));
        set(&entry, "resource", resource);
        let entries = Array::new();
        entries.push(&entry);
        let gd = obj();
        set(&gd, "layout", layout.clone());
        set(&gd, "entries", entries.into());
        let group = call(native, "createBindGroup", &[gd]);
        call(&pass, "setBindGroup", &[num(0), group]);
        call(&pass, "setStencilReference", &[num(37 + i)]);
        call(&pass, "draw", &[num(3), num(1), num(0), num(0)]);
    }
    call(&pass, "end", &[]);
    let command = call(&encoder, "finish", &[]);
    let list = Array::new();
    list.push(&command);
    call(&device.0.queue, "submit", &[list.into()]);
}
