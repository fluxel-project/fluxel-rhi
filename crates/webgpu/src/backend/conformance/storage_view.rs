//! Real-GPU coverage for explicit storage-buffer view lowering.
use super::*;

#[wasm_bindgen_test(async)]
async fn gpu_structured_storage_view_binds_only_its_declared_range() {
    let d = open().await;
    let data = buffer(
        &d,
        512,
        BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    let group_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Compute],
                ty: BindingType::StorageBuffer {
                    dynamic_offset: false,
                    min_size: 16,
                    access: StorageAccess::ReadWrite,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::StorageBufferView {
                    buffer: &data,
                    range: 256..272,
                    view: StorageBufferView::Structured { stride: 4 },
                },
            }],
        })
        .unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&group_layout],
            push_constants: &[],
        })
        .unwrap();
    let shader = d
        .shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(
                b"@group(0) @binding(0) var<storage, read_write> values: array<u32>; @compute @workgroup_size(1) fn cs() { values[0] = 0x12345678u; values[3] = 0x87654321u; }"
                    .as_slice(),
            ),
            entries: vec![ShaderEntryPoint {
                name: "cs".into(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([1, 1, 1]),
            }],
            required_features: vec![],
        })
        .await
        .unwrap();
    let compute = d.compute().unwrap();
    let pipeline = compute
        .create_pipeline(ComputePipelineDesc {
            label: None,
            layout,
            compute: PipelineStage {
                module: shader,
                entry: "cs".into(),
                constants: vec![],
            },
            required_subgroup_size: None,
            cache: None,
        })
        .await
        .unwrap();
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    e.write_buffer(&data, 0, &[0xA5; 512]).unwrap();
    let mut pass = compute.begin(&mut e).unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_bind_group(0, &group, &[]).unwrap();
    pass.dispatch([1, 1, 1]).unwrap();
    pass.finish().unwrap();
    let ticket = e.readback_buffer(&data, 240..288).unwrap();
    let point = submit(&d, e).await;
    // SAFETY: the awaited point covers this dispatch and its readback copy.
    let actual = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    assert_eq!(&actual[..16], &[0xA5; 16]);
    assert_eq!(&actual[16..20], &0x12345678u32.to_le_bytes());
    assert_eq!(&actual[20..28], &[0xA5; 8]);
    assert_eq!(&actual[28..32], &0x87654321u32.to_le_bytes());
}
