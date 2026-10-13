//! GPU resource ownership and two-pass mirror-scene recording for example 07.
//!
//! The returned work is submitted by the host. This workload therefore does
//! not own presentation, completion waiting, or cross-frame retirement.

use super::super::*;
use super::resources::*;
use crate::gltf;

pub struct Offscreen<B: OffscreenBackend> {
    lane: LaneId,
    dv: Buffer<B>,
    di: Buffer<B>,
    pv: Buffer<B>,
    pi: Buffer<B>,
    off: Texture<B>,
    off_view: TextureView<B>,
    off_depth: Texture<B>,
    off_depth_view: TextureView<B>,
    scene_depth: Texture<B>,
    scene_depth_view: TextureView<B>,
    scene_extent: [u32; 2],
    scene_u: Buffer<B>,
    off_u: Buffer<B>,
    mirror_u: Buffer<B>,
    scene_g: BindGroup<B>,
    off_g: BindGroup<B>,
    mirror_g: BindGroup<B>,
    scene_p: RasterPipeline<B>,
    off_p: RasterPipeline<B>,
    mirror_p: RasterPipeline<B>,
    dv_len: u64,
    di_len: u64,
    pv_len: u64,
    pi_len: u64,
    dn: u32,
    pn: u32,
    uploaded: bool,
}
impl<B: OffscreenBackend> Offscreen<B> {
    pub async fn new(
        d: &Device<B>,
        lane: LaneId,
        fmt: TextureFormat,
        extent: [u32; 2],
    ) -> Result<Self> {
        let td = TextureDesc {
            label: Some("07 mirrored dragon target".into()),
            dimension: TextureDimension::D2,
            extent: [512, 512, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT.union(TextureUsage::TEXTURE_BINDING),
            view_formats: vec![],
        };
        match d.texture_support(&td) {
            Support::Supported(s) if s.filterable && s.blendable => (),
            Support::Supported(_) => {
                return Err(Error::Unsupported(
                    "07_offscreen needs filterable RGBA8 render target",
                ));
            }
            Support::Unsupported(x) => return Err(Error::Unsupported(x)),
        };
        let dragon = gltf::load_embedded_model(DRAGON, gltf::LoadOptions::CPP_PORT)
            .map_err(|e| Error::Backend(e.to_string()))?;
        let plane = gltf::load_embedded_model(PLANE, gltf::LoadOptions::CPP_PORT)
            .map_err(|e| Error::Backend(e.to_string()))?;
        let dv = buf(
            d,
            "07 dragon vertices",
            dragon.vertices.len() as u64,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        )?;
        let di = buf(
            d,
            "07 dragon indices",
            dragon.indices.len() as u64,
            BufferUsage::INDEX.union(BufferUsage::COPY_DST),
        )?;
        let pv = buf(
            d,
            "07 plane vertices",
            plane.vertices.len() as u64,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        )?;
        let pi = buf(
            d,
            "07 plane indices",
            plane.indices.len() as u64,
            BufferUsage::INDEX.union(BufferUsage::COPY_DST),
        )?;
        let scene_u = buf(
            d,
            "07 scene transform",
            256,
            BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
        )?;
        let off_u = buf(
            d,
            "07 reflection transform",
            256,
            BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
        )?;
        let mirror_u = buf(
            d,
            "07 mirror transform",
            256,
            BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
        )?;
        let off = d.create_texture(&td)?;
        let off_view = d.create_view(
            &off,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )?;
        let (off_depth, off_depth_view) = depth(d, [512, 512])?;
        let (scene_depth, scene_depth_view) = depth(d, extent)?;
        let sampler = d.create_sampler(&SamplerDesc {
            label: Some("07 reflection sampler".into()),
            address: [AddressMode::ClampToEdge; 3],
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mip_filter: FilterMode::Linear,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare: None,
            border: None,
        })?;
        let sl = d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("07 scene layout".into()),
            entries: vec![uniform_entry(0, vec![ShaderStage::Vertex])],
        })?;
        let ml = d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("07 mirror layout".into()),
            entries: vec![
                uniform_entry(0, vec![ShaderStage::Vertex]),
                entry(
                    1,
                    vec![ShaderStage::Fragment],
                    BindingType::SampledTexture {
                        sample: TextureSampleType::Float,
                        dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                entry(
                    2,
                    vec![ShaderStage::Fragment],
                    BindingType::Sampler {
                        filtering: true,
                        comparison: false,
                    },
                ),
            ],
        })?;
        let scene_g = ug(d, &sl, &scene_u, "07 scene group")?;
        let off_g = ug(d, &sl, &off_u, "07 reflected group")?;
        let es = [
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &mirror_u,
                    range: 0..256,
                },
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&off_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Sampler(&sampler),
            },
        ];
        let mirror_g = d.create_bind_group(&BindGroupDesc {
            label: Some("07 mirror group".into()),
            layout: &ml,
            entries: &es,
        })?;
        let spl = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("07 scene pipeline layout".into()),
            groups: &[&sl],
            push_constants: &[],
        })?;
        let mpl = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("07 mirror pipeline layout".into()),
            groups: &[&ml],
            push_constants: &[],
        })?;
        let off_p = pipe(
            d,
            "07 reflected dragon",
            spl.clone(),
            TextureFormat::Rgba8Unorm,
            PVS,
            PVS_SPIRV,
            PPS,
            PPS_SPIRV,
            "phong_vs",
            "phong_ps",
            true,
            CullMode::Front,
        )
        .await?;
        let scene_p = pipe(
            d,
            "07 dragon",
            spl,
            fmt,
            PVS,
            PVS_SPIRV,
            PPS,
            PPS_SPIRV,
            "phong_vs",
            "phong_ps",
            true,
            CullMode::Back,
        )
        .await?;
        let mirror_p = pipe(
            d,
            "07 blurred mirror",
            mpl,
            fmt,
            MVS,
            MVS_SPIRV,
            MPS,
            MPS_SPIRV,
            "mirror_vs",
            "mirror_ps",
            false,
            CullMode::None,
        )
        .await?;
        Ok(Self {
            lane,
            dv,
            di,
            pv,
            pi,
            off,
            off_view,
            off_depth,
            off_depth_view,
            scene_depth,
            scene_depth_view,
            scene_extent: extent,
            scene_u,
            off_u,
            mirror_u,
            scene_g,
            off_g,
            mirror_g,
            scene_p,
            off_p,
            mirror_p,
            dv_len: dragon.vertices.len() as u64,
            di_len: dragon.indices.len() as u64,
            pv_len: plane.vertices.len() as u64,
            pi_len: plane.indices.len() as u64,
            dn: (dragon.indices.len() / 4) as u32,
            pn: (plane.indices.len() / 4) as u32,
            uploaded: false,
        })
    }
    pub fn record(
        &mut self,
        d: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
        time: f32,
    ) -> Result<RecordedWork<B>> {
        if extent != self.scene_extent {
            let (depth, view) = depth(d, extent)?;
            self.scene_depth = depth;
            self.scene_depth_view = view;
            self.scene_extent = extent;
        }
        let mut e = d.encoder(self.lane)?;
        let first = !self.uploaded;
        if first {
            let a = gltf::load_embedded_model(DRAGON, gltf::LoadOptions::CPP_PORT)
                .map_err(|x| Error::Backend(x.to_string()))?;
            let b = gltf::load_embedded_model(PLANE, gltf::LoadOptions::CPP_PORT)
                .map_err(|x| Error::Backend(x.to_string()))?;
            e.write_buffer(&self.dv, 0, &a.vertices)?;
            e.write_buffer(&self.di, 0, &a.indices)?;
            e.write_buffer(&self.pv, 0, &b.vertices)?;
            e.write_buffer(&self.pi, 0, &b.indices)?;
        }
        e.write_buffer(&self.off_u, 0, &scene_matrix(time, true, [512, 512]))?;
        e.write_buffer(&self.mirror_u, 0, &plane_matrix(extent))?;
        e.write_buffer(&self.scene_u, 0, &scene_matrix(time, false, extent))?;
        let range = Subresources {
            texture: &self.off,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..1,
        };
        e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: range,
            usage: StateTransition {
                from: if first {
                    TextureUses {
                        stages: StageMask::EMPTY,
                        accesses: AccessMask::EMPTY,
                    }
                } else {
                    TextureUses {
                        stages: StageMask::FRAGMENT,
                        accesses: AccessMask::SHADER_READ,
                    }
                },
                to: TextureUses {
                    stages: StageMask::COLOR_OUTPUT,
                    accesses: AccessMask::COLOR_WRITE,
                },
            },
            layout: Some((
                if first {
                    ImageLayout::Undefined
                } else {
                    ImageLayout::ShaderReadOnly
                },
                ImageLayout::ColorAttachment,
            )),
            ownership: None,
        })])?;
        let c = [Some(ColorAttachment {
            view: AttachmentView::Texture(&self.off_view),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 0.])),
            store: StoreOp::Store,
        })];
        let depth = DepthStencilAttachment {
            view: AttachmentView::Texture(&self.off_depth_view),
            depth: Some((LoadOp::Clear(1.0), StoreOp::Discard)),
            stencil: None,
            depth_read_only: false,
            stencil_read_only: false,
        };
        let mut p = e.begin_raster(&RasterPassDesc {
            label: Some("07 mirrored dragon"),
            colors: &c,
            depth_stencil: Some(depth),
            render_area: Rect {
                x: 0,
                y: 0,
                width: 512,
                height: 512,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })?;
        p.viewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: 512.0,
            height: 512.0,
            min_depth: 0.0,
            max_depth: 1.0,
        })?;
        p.scissor(Rect {
            x: 0,
            y: 0,
            width: 512,
            height: 512,
        })?;
        p.set_pipeline(&self.off_p)?;
        p.set_bind_group(0, &self.off_g, &[])?;
        p.set_vertex_buffer(0, &self.dv, 0..self.dv_len)?;
        p.set_index_buffer(&self.di, 0..self.di_len, IndexFormat::Uint32)?;
        p.draw_indexed(0..self.dn, 0, 0..1)?;
        p.finish()?;
        let range = Subresources {
            texture: &self.off,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..1,
        };
        e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: range,
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::COLOR_OUTPUT,
                    accesses: AccessMask::COLOR_WRITE,
                },
                to: TextureUses {
                    stages: StageMask::FRAGMENT,
                    accesses: AccessMask::SHADER_READ,
                },
            },
            layout: Some((ImageLayout::ColorAttachment, ImageLayout::ShaderReadOnly)),
            ownership: None,
        })])?;
        let c = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.025, 0.025, 0.025, 1.])),
            store: StoreOp::Store,
        })];
        let depth = DepthStencilAttachment {
            view: AttachmentView::Texture(&self.scene_depth_view),
            depth: Some((LoadOp::Clear(1.0), StoreOp::Discard)),
            stencil: None,
            depth_read_only: false,
            stencil_read_only: false,
        };
        let mut p = e.begin_raster(&RasterPassDesc {
            label: Some("07 mirror and dragon"),
            colors: &c,
            depth_stencil: Some(depth),
            render_area: Rect {
                x: 0,
                y: 0,
                width: extent[0],
                height: extent[1],
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })?;
        p.viewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: extent[0] as f32,
            height: extent[1] as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        })?;
        p.scissor(Rect {
            x: 0,
            y: 0,
            width: extent[0],
            height: extent[1],
        })?;
        p.set_pipeline(&self.mirror_p)?;
        p.set_bind_group(0, &self.mirror_g, &[])?;
        p.set_vertex_buffer(0, &self.pv, 0..self.pv_len)?;
        p.set_index_buffer(&self.pi, 0..self.pi_len, IndexFormat::Uint32)?;
        p.draw_indexed(0..self.pn, 0, 0..1)?;
        p.set_pipeline(&self.scene_p)?;
        p.set_bind_group(0, &self.scene_g, &[])?;
        p.set_vertex_buffer(0, &self.dv, 0..self.dv_len)?;
        p.set_index_buffer(&self.di, 0..self.di_len, IndexFormat::Uint32)?;
        p.draw_indexed(0..self.dn, 0, 0..1)?;
        p.finish()?;
        let work = e.finish()?;
        self.uploaded = true;
        Ok(work)
    }
}
