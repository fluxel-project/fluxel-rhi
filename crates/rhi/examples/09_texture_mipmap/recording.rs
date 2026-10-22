//! Per-frame upload, mip generation, and tunnel raster recording.
//!
//! The caller owns the acquired surface lease and submission result after this
//! module returns its one-shot recorded work.

use super::{MipmapBackend, MipmapWorkloadBackend, Tunnel};
use fluxel_rhi::api::*;

pub struct FrameCommit;

impl<B: MipmapWorkloadBackend> Tunnel<B> {
    pub fn record(
        &mut self,
        device: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<(RecordedWork<B>, FrameCommit)>
    where
        B: MipmapBackend,
    {
        let mut encoder = device.encoder(self.lane)?;
        let commit = self.encode(&mut encoder, device, image, extent)?;
        let work = encoder.finish()?;
        Ok((work, commit))
    }

    /// Records the same mip-generation and tunnel verbs into an immediate
    /// owner's encoder. State publication waits until that command step closes.
    pub fn encode(
        &mut self,
        encoder: &mut CommandEncoder<B>,
        device: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<FrameCommit> {
        self.resize_depth(device, extent)?;
        let now = frame_time_seconds();
        if let Some(last_frame) = self.last_frame.replace(now) {
            // `texturemipmapgen` advances its timer by elapsed seconds * 0.05
            // and then rotates by `timer * 360°`.  Retaining that relation
            // preserves the source animation without making the RHI own a
            // frame clock.
            self.rotation_turns =
                (self.rotation_turns + (now - last_frame).max(0.0) as f32 * 0.05).rem_euclid(1.0);
        }
        if !self.initialized {
            encoder.write_buffer(&self.vertices, 0, &self.vertex_data)?;
            encoder.write_buffer(&self.indices, 0, &self.index_data)?;
            self.chain.encode_generation(encoder, &self.pixels)?;
        }
        encoder.write_buffer(
            &self.uniform,
            0,
            &tunnel_camera(extent, self.rotation_turns),
        )?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.015, 0.015, 0.025, 1.0])),
            store: StoreOp::Store,
        })];
        encoder.transition_resources(&[super::super::utils::discarded_depth(&self.depth, 0..1)])?;
        let depth = DepthStencilAttachment {
            view: AttachmentView::Texture(&self.depth_view),
            depth: Some((LoadOp::Clear(1.0), StoreOp::Discard)),
            stencil: None,
            depth_read_only: false,
            stencil_read_only: false,
        };
        let mut pass = encoder.begin_raster(&RasterPassDesc {
            label: Some("09 textured tunnel"),
            colors: &colors,
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
        pass.set_pipeline(&self.pipeline)?;
        pass.set_bind_group(0, &self.group, &[])?;
        pass.set_vertex_buffer(0, &self.vertices, 0..self.vertex_data.len() as u64)?;
        pass.set_index_buffer(
            &self.indices,
            0..self.index_data.len() as u64,
            IndexFormat::Uint32,
        )?;
        pass.viewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: extent[0] as f32,
            height: extent[1] as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        })?;
        pass.scissor(Rect {
            x: 0,
            y: 0,
            width: extent[0],
            height: extent[1],
        })?;
        pass.draw_indexed(0..self.index_count, 0, 0..1)?;
        pass.finish()?;
        Ok(FrameCommit)
    }

    pub fn commit_frame(&mut self, _: FrameCommit) {
        self.initialized = true;
    }
}

#[cfg(target_arch = "wasm32")]
fn frame_time_seconds() -> f64 {
    web_sys::window()
        .and_then(|window| window.performance())
        .map(|performance| performance.now() / 1000.0)
        .unwrap_or(0.0)
}

#[cfg(not(target_arch = "wasm32"))]
fn frame_time_seconds() -> f64 {
    use std::sync::OnceLock;
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_secs_f64()
}
/// Matches the source sample's first-person camera and its three-matrix
/// texturemipmapgen uniform block.
fn tunnel_camera(extent: [u32; 2], rotation_turns: f32) -> [u8; 256] {
    let aspect = extent[0].max(1) as f32 / extent[1].max(1) as f32;
    let f = 1.0 / (60f32.to_radians() * 0.5).tan();
    let near = 0.1;
    let far = 1024.0;
    let projection = [
        f / aspect,
        0.,
        0.,
        0.,
        0.,
        f,
        0.,
        0.,
        0.,
        0.,
        far / (near - far),
        -1.,
        0.,
        0.,
        far * near / (near - far),
        0.,
    ];
    let (s, c) = 90f32.to_radians().sin_cos();
    // `Camera::updateViewMatrix`: rotY(90) * translate(40.75, 0, 0).
    let view = [
        c,
        0.,
        -s,
        0.,
        0.,
        1.,
        0.,
        0.,
        s,
        0.,
        c,
        0.,
        c * 40.75,
        0.,
        -s * 40.75,
        1.,
    ];
    // Reference animation: `rotate(identity, timer * 360°, +X)`.
    let (s, c) = (rotation_turns * 360.0).to_radians().sin_cos();
    let model = [1., 0., 0., 0., 0., c, s, 0., 0., -s, c, 0., 0., 0., 0., 1.];
    let mut bytes = [0; 256];
    for (index, value) in projection.into_iter().chain(view).chain(model).enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_ne_bytes());
    }
    for (index, value) in [-40.75_f32, 0.0, 0.0, 0.0, 0.0].into_iter().enumerate() {
        let offset = 192 + index * 4;
        bytes[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
    }
    bytes
}
