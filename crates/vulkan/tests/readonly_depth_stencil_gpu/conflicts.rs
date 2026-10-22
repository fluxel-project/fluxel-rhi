//! Reject primary and secondary pipelines that write a read-only plane.
use super::*;
use fluxel_rhi_vulkan::Vulkan;

pub(super) fn check(
    d: &Device<Vulkan>,
    lane: LaneId,
    depth: &TextureView<Vulkan>,
    pipeline: &RasterPipeline<Vulkan>,
    vertex: &Buffer<Vulkan>,
    readonly_depth: bool,
) {
    {
        let mut rejected_encoder = d.encoder(lane).unwrap();
        let bad_attachment = DepthStencilAttachment {
            view: AttachmentView::Texture(depth),
            depth: Some((LoadOp::Load, StoreOp::Store)),
            stencil: Some((LoadOp::Load, StoreOp::Store)),
            depth_read_only: readonly_depth,
            stencil_read_only: !readonly_depth,
        };
        let mut bad_pass = rejected_encoder
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &[],
                depth_stencil: Some(bad_attachment),
                render_area: Rect {
                    x: 0,
                    y: 0,
                    width: 16,
                    height: 16,
                },
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })
            .unwrap();
        assert!(
            bad_pass.set_pipeline(pipeline).is_err(),
            "pipeline must not write a read-only plane"
        );
    }
    {
        let signature = SecondaryRasterDesc {
            color_formats: vec![],
            depth_stencil_format: Some(TextureFormat::Depth24PlusStencil8),
            samples: 1,
            multiview_mask: None,
        };
        let domain = d.secondary().unwrap();
        let mut secondary = domain.begin(&signature).unwrap();
        secondary.set_pipeline(pipeline).unwrap();
        secondary.set_vertex_buffer(0, vertex, 0..72).unwrap();
        secondary
            .viewport(Viewport {
                x: 0.0,
                y: 0.0,
                width: 16.0,
                height: 16.0,
                min_depth: 0.0,
                max_depth: 1.0,
            })
            .unwrap();
        secondary
            .scissor(Rect {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
            })
            .unwrap();
        secondary.draw(0..3, 0..1).unwrap();
        let bundle = domain.finish(secondary).unwrap();
        let mut rejected_encoder = d.encoder(lane).unwrap();
        let bad_attachment = DepthStencilAttachment {
            view: AttachmentView::Texture(depth),
            depth: Some((LoadOp::Load, StoreOp::Store)),
            stencil: Some((LoadOp::Load, StoreOp::Store)),
            depth_read_only: readonly_depth,
            stencil_read_only: !readonly_depth,
        };
        let mut bad_pass = rejected_encoder
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &[],
                depth_stencil: Some(bad_attachment),
                render_area: Rect {
                    x: 0,
                    y: 0,
                    width: 16,
                    height: 16,
                },
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: Some(signature),
            })
            .unwrap();
        assert!(
            domain.execute(&mut bad_pass, &bundle).is_err(),
            "secondary pipeline must not write a read-only plane"
        );
    }
}
