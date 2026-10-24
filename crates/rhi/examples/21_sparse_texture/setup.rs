//! Create sparse mip regions and checkerboard pipelines.

use super::*;
use std::sync::Arc;

impl<
    B: ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + RasterBackend
        + ReadbackBackend
        + SparseBackend
        + DeferredBackend,
> Demo<B>
{
    pub(super) async fn new(
        d: &Device<B>,
        lane: LaneId,
        extent: [u32; 2],
        format: TextureFormat,
        shader_format: ShaderFormat,
    ) -> Result<Self> {
        if extent != [1280, 720] {
            return Err(Error::Unsupported("reference extent is 1280x720"));
        }
        let sparse = d.sparse().ok_or(Error::Unsupported("sparse textures"))?;
        let sparse_texture = sparse.texture(&TextureDesc {
            label: Some("reserved checkerboard".into()),
            dimension: TextureDimension::D2,
            extent: [256, 256, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 9,
            samples: 1,
            usage: TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
            view_formats: vec![],
        })?;
        let tiling = sparse.tiling(&sparse_texture)?;
        let mut allocations = Vec::new();
        let mut regions = Vec::new();
        let mut region_mips = Vec::new();
        for (mip, grid) in tiling.standard_mip_grids.iter().enumerate() {
            let tiles = u64::from(grid[0]) * u64::from(grid[1]) * u64::from(grid[2]);
            allocations.push(sparse.allocation_for(&sparse_texture, tiles * tiling.tile_bytes)?);
            regions.push(SparseTile {
                aspect: None,
                mip: mip as u32,
                layer: 0,
                origin: [0, 0, 0],
                count: *grid,
            });
            region_mips.push(mip as u32..mip as u32 + 1);
        }
        if let Some(tail) = tiling.packed_tail {
            allocations.push(sparse.allocation_for(
                &sparse_texture,
                u64::from(tail.tile_count) * tiling.tile_bytes,
            )?);
            regions.push(SparseTile::packed_tail(tail));
            region_mips.push(tail.first_mip..tail.first_mip + tail.mip_count);
        }
        if region_mips.last().map(|r| r.end) != Some(9) {
            return Err(Error::InvalidState);
        }
        let uploaded = vec![false; regions.len()];
        let vertex = d.create_buffer(&BufferDesc {
            label: Some("checkerboard quad".into()),
            size: 120,
            usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })?;
        let output = d.create_texture(&TextureDesc {
            label: Some("triangle output".into()),
            dimension: TextureDimension::D2,
            extent: [extent[0], extent[1], 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT
                | TextureUsage::TEXTURE_BINDING
                | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })?;
        let view = d.create_view(
            &output,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )?;
        let row_pitch = extent[0]
            .checked_mul(4)
            .and_then(|n| n.checked_add(255))
            .ok_or(Error::InvalidRange)?
            & !255;
        let readback = d.create_buffer(&BufferDesc {
            label: Some("triangle readback".into()),
            size: u64::from(row_pitch) * u64::from(extent[1]),
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })?;
        async fn shader<B: ShaderBackend>(
            d: &Device<B>,
            bytes: &[u8],
            entry: &str,
            stage: ShaderStage,
            shader_format: ShaderFormat,
        ) -> Result<Shader<B>> {
            d.shader(ShaderDesc {
                label: Some("Microsoft HelloTriangle shader".into()),
                format: shader_format,
                bytes: Arc::from(bytes),
                entries: vec![ShaderEntryPoint {
                    name: entry.into(),
                    stage,
                    workgroup_size: None,
                }],
                required_features: vec![],
            })
            .await
        }
        let mut sparse_views = Vec::new();
        for mip in 0..9 {
            sparse_views.push(d.create_view(
                &sparse_texture,
                &TextureViewDesc {
                    format: TextureFormat::Rgba8Unorm,
                    dimension: TextureViewDimension::D2,
                    aspects: TextureAspects::COLOR,
                    mips: mip..mip + 1,
                    layers: 0..1,
                },
            )?);
        }
        let sampler = d.create_sampler(&SamplerDesc {
            label: None,
            address: [AddressMode::ClampToBorder; 3],
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mip_filter: FilterMode::Linear,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare: None,
            border: Some(BorderColor::TransparentBlack),
        })?;
        let binding = |binding, ty| BindGroupLayoutEntry {
            binding,
            visibility: vec![ShaderStage::Fragment],
            ty,
            count: BindingCount::One,
            partially_bound: false,
            nonuniform_indexing: false,
        };
        let group_layout = d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                binding(
                    1,
                    BindingType::SampledTexture {
                        sample: TextureSampleType::Float,
                        dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                binding(
                    2,
                    BindingType::Sampler {
                        filtering: true,
                        comparison: false,
                    },
                ),
            ],
        })?;
        let mut groups = Vec::new();
        for sparse_view in &sparse_views {
            groups.push(d.create_bind_group(&BindGroupDesc {
                label: None,
                layout: &group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(sparse_view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Sampler(&sampler),
                    },
                ],
            })?);
        }
        let layout = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&group_layout],
            push_constants: &[],
        })?;
        let vs = shader(
            d,
            match shader_format {
                ShaderFormat::Dxil => {
                    include_bytes!("../assets/21_sparse_texture/vs.dxil").as_slice()
                }
                ShaderFormat::SpirV => {
                    include_bytes!("../assets/21_sparse_texture/vs.spv").as_slice()
                }
                _ => return Err(Error::Unsupported("shader format")),
            },
            "VSMain",
            ShaderStage::Vertex,
            shader_format,
        )
        .await?;
        let ps = shader(
            d,
            match shader_format {
                ShaderFormat::Dxil => {
                    include_bytes!("../assets/21_sparse_texture/ps.dxil").as_slice()
                }
                ShaderFormat::SpirV => {
                    include_bytes!("../assets/21_sparse_texture/ps.spv").as_slice()
                }
                _ => return Err(Error::Unsupported("shader format")),
            },
            "PSMain",
            ShaderStage::Fragment,
            shader_format,
        )
        .await?;
        let pipeline = d
            .raster_pipeline(RasterPipelineDesc {
                label: Some("reserved mip zero".into()),
                layout,
                vertex: PipelineStage {
                    module: vs,
                    entry: "VSMain".into(),
                    constants: vec![],
                },
                fragment: Some(PipelineStage {
                    module: ps,
                    entry: "PSMain".into(),
                    constants: vec![],
                }),
                vertex_buffers: vec![VertexBufferLayout {
                    stride: 20,
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
                            format: VertexFormat::Float32(2),
                        },
                    ],
                }],
                color_targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: 15,
                })],
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: FrontFace::Cw,
                    cull_mode: CullMode::Back,
                    unclipped_depth: false,
                    polygon_mode: PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: MultisampleState {
                    count: 1,
                    mask: u32::MAX as u64,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            })
            .await?;
        let display = display::Display::new_with_format(d, &view, format, shader_format).await?;
        Ok(Self {
            lane,
            vertex,
            sparse_texture,
            allocations,
            regions,
            region_mips,
            uploaded,
            mapped: None,
            selected: 0,
            frame: 0,
            output,
            view,
            readback,
            row_pitch,
            groups,
            pipeline,
            display,
            initialized: false,
        })
    }
    pub(super) fn select(&mut self, d: &Device<B>, mip: u32) -> Result<()> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(902100);
        let index = self
            .region_mips
            .iter()
            .position(|range| range.contains(&mip))
            .ok_or(Error::InvalidRange)?;
        if (self.frame == 0 || self.selected != mip)
            && std::env::var_os("FLUXEL_SPARSE_TRACE").is_some()
        {
            eprintln!("sparse frame={} mip={} region={}", self.frame, mip, index);
        }
        self.selected = mip;
        if self.mapped == Some(index) {
            return Ok(());
        }
        let tile = |i: usize| {
            let t = &self.regions[i];
            SparseTile {
                aspect: None,
                mip: t.mip,
                layer: t.layer,
                origin: t.origin,
                count: t.count,
            }
        };
        let mut binds = Vec::new();
        if let Some(old) = self.mapped {
            binds.push(SparseBind {
                resource: self.sparse_texture.clone(),
                tiles: vec![tile(old)],
                allocation: None,
                allocation_offset: 0,
            });
        }
        binds.push(SparseBind {
            resource: self.sparse_texture.clone(),
            tiles: vec![tile(index)],
            allocation: Some(self.allocations[index].clone()),
            allocation_offset: 0,
        });
        // SAFETY: the previous frame has completed; regions do not overlap, and each mapping covers a full standard mip or packed tail.
        let plan = unsafe {
            SubmissionPlan::from_compiled(
                PlanId {
                    device: d.id(),
                    ordinal: NEXT.fetch_add(1, Ordering::Relaxed),
                },
                Arc::from([StepDesc {
                    lane: self.lane,
                    kind: StepKind::SparseBind,
                    waits: vec![],
                }]),
            )
        };
        let step = plan.step(0);
        // SAFETY: the scene retains texture and allocations until completion, which is awaited before access.
        let outcome = unsafe {
            d.queue().submit(SubmitPacket {
                plan,
                steps: vec![BoundStep {
                    step,
                    payload: StepPayload::SparseBind(binds),
                }],
                completion: CompletionRequest::End,
            })
        };
        let receipt = match outcome {
            SubmitOutcome::Accepted(receipt) => receipt,
            SubmitOutcome::Rejected { error, .. } => return Err(error),
            outcome => {
                let error = match &outcome {
                    SubmitOutcome::PartiallyAccepted { error, .. }
                    | SubmitOutcome::Indeterminate { error, .. } => error.clone(),
                    _ => unreachable!("accepted and rejected outcomes handled above"),
                };
                // Keep the native outcome (and its retained sparse bindings)
                // alive through the drain. If that fails, do not drop it or
                // replace the original submission error with the drain error.
                if let Err(drain_error) = utils::block_on(d.wait_idle()) {
                    eprintln!(
                        "21_sparse_texture sparse bind failed: {error:?}; GPU drain failed: {drain_error:?}"
                    );
                    std::process::abort();
                }
                drop(outcome);
                return Err(error);
            }
        };
        let point = receipt
            .checkpoints
            .into_iter()
            .next()
            .ok_or(Error::InvalidState)?
            .point;
        utils::block_on(d.wait_completion(&point))?;
        self.mapped = Some(index);
        Ok(())
    }
}
