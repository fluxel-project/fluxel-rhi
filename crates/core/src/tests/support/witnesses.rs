//! Command, raster, and completion SPI witnesses for Core contract tests.
use super::*;

impl SharedBackend for NativeShared {}
impl CommandBackend for NativeShared {
    fn begin_encoder(_: &Arc<()>, _: LaneId) -> Result<()> {
        Ok(())
    }
    fn debug_marker(_: &mut (), _: DebugMarkerScope, op: DebugMarkerOp<'_>) -> Result<()> {
        match op {
            DebugMarkerOp::Push(_) => DEBUG_PUSHES.with(|count| count.set(count.get() + 1)),
            DebugMarkerOp::Pop => DEBUG_POPS.with(|count| count.set(count.get() + 1)),
            DebugMarkerOp::Insert(_) => {}
        }
        Ok(())
    }
    fn transition(_: &mut (), _: &[ResourceTransition<'_, Self>]) -> Result<()> {
        Ok(())
    }
    fn copy_buffer(_: &mut (), _: &Arc<()>, _: u64, _: &Arc<()>, _: u64, _: u64) -> Result<()> {
        Ok(())
    }
    fn copy_texture(
        _: &mut (),
        _: TextureRegion<'_, Self>,
        _: TextureRegion<'_, Self>,
    ) -> Result<()> {
        Ok(())
    }
    fn buffer_to_texture(
        _: &mut (),
        _: &Arc<()>,
        _: TexelCopyBufferLayout,
        _: TextureRegion<'_, Self>,
    ) -> Result<()> {
        Ok(())
    }
    fn texture_to_buffer(
        _: &mut (),
        _: TextureRegion<'_, Self>,
        _: &Arc<()>,
        _: TexelCopyBufferLayout,
    ) -> Result<()> {
        Ok(())
    }
    fn write_buffer(_: &mut (), _: &Arc<()>, _: u64, _: &[u8]) -> Result<()> {
        Ok(())
    }
    fn write_texture(
        _: &mut (),
        _: TextureRegion<'_, Self>,
        _: TexelCopyBufferLayout,
        _: &[u8],
    ) -> Result<()> {
        Ok(())
    }
}
impl CommandBackend for GlLocal {
    fn begin_encoder(_: &Rc<()>, _: LaneId) -> Result<Rc<()>> {
        Ok(Rc::new(()))
    }
    fn transition(_: &mut Rc<()>, _: &[ResourceTransition<'_, Self>]) -> Result<()> {
        GL_TRANSITIONS.with(|calls| calls.set(calls.get() + 1));
        Ok(())
    }
    fn copy_buffer(_: &mut Rc<()>, _: &Rc<()>, _: u64, _: &Rc<()>, _: u64, _: u64) -> Result<()> {
        Ok(())
    }
    fn copy_texture(
        _: &mut Rc<()>,
        _: TextureRegion<'_, Self>,
        _: TextureRegion<'_, Self>,
    ) -> Result<()> {
        Ok(())
    }
    fn buffer_to_texture(
        _: &mut Rc<()>,
        _: &Rc<()>,
        _: TexelCopyBufferLayout,
        _: TextureRegion<'_, Self>,
    ) -> Result<()> {
        Ok(())
    }
    fn texture_to_buffer(
        _: &mut Rc<()>,
        _: TextureRegion<'_, Self>,
        _: &Rc<()>,
        _: TexelCopyBufferLayout,
    ) -> Result<()> {
        Ok(())
    }
    fn write_buffer(_: &mut Rc<()>, _: &Rc<()>, _: u64, _: &[u8]) -> Result<()> {
        Ok(())
    }
    fn write_texture(
        _: &mut Rc<()>,
        _: TextureRegion<'_, Self>,
        _: TexelCopyBufferLayout,
        _: &[u8],
    ) -> Result<()> {
        Ok(())
    }
}
impl RasterBackend for NativeShared {
    const DYNAMIC_DEPTH_BIAS_AVAILABLE: bool = true;
    const RECT_ATTACHMENT_CLEAR_AVAILABLE: bool = true;
    fn begin_raster(_: &mut (), _: &RasterPassDesc<'_, Self>) -> Result<()> {
        Ok(())
    }
    fn set_pipeline(_: &mut (), _: &Arc<()>) -> Result<()> {
        Ok(())
    }
    fn set_bind_group(_: &mut (), _: u32, _: &Arc<()>, _: &[u32]) -> Result<()> {
        Ok(())
    }
    fn set_vertex_buffer(_: &mut (), _: u32, _: &Arc<()>, _: u64, _: u64) -> Result<()> {
        Ok(())
    }
    fn set_index_buffer(_: &mut (), _: &Arc<()>, _: u64, _: u64, _: IndexFormat) -> Result<()> {
        Ok(())
    }
    fn viewport(_: &mut (), _: Viewport) -> Result<()> {
        Ok(())
    }
    fn scissor(_: &mut (), _: Rect) -> Result<()> {
        Ok(())
    }
    fn blend_constant(_: &mut (), _: [f32; 4]) -> Result<()> {
        Ok(())
    }
    fn stencil_reference(_: &mut (), _: u32) -> Result<()> {
        Ok(())
    }
    fn depth_bias(_: &mut (), value: DynamicDepthBias) -> Result<()> {
        DEPTH_BIAS_VALUES.with(|values| values.borrow_mut().push(value));
        Ok(())
    }
    fn clear_rect(
        _: &mut (),
        _: Rect,
        _: &[ColorAttachmentClear],
        _: Option<f32>,
        _: Option<u32>,
    ) -> Result<()> {
        CLEAR_RECT_CALLS.with(|calls| calls.set(calls.get() + 1));
        Ok(())
    }
    fn draw(_: &mut (), _: Range<u32>, _: Range<u32>) -> Result<()> {
        DRAW_CALLS.with(|calls| calls.set(calls.get() + 1));
        Ok(())
    }
    fn draw_indexed(_: &mut (), _: Range<u32>, _: i32, _: Range<u32>) -> Result<()> {
        DRAW_CALLS.with(|calls| calls.set(calls.get() + 1));
        Ok(())
    }
    fn end_raster(_: &mut ()) -> Result<()> {
        Ok(())
    }
    fn abort_raster(_: &mut ()) {}
}
impl CompletionBackend for NativeShared {
    type WaitFuture<'a> = Ready<Result<()>>;
    fn completion_status(_: &Arc<()>, _: &Arc<()>) -> CompletionState {
        CompletionState::Complete
    }
    fn wait_completion<'a>(_: &'a Arc<()>, _: &'a Arc<()>) -> Self::WaitFuture<'a> {
        ready(Ok(()))
    }
    fn wait_idle<'a>(_: &'a Arc<()>) -> Self::WaitFuture<'a> {
        ready(Ok(()))
    }
}
impl CompletionBackend for GlLocal {
    type WaitFuture<'a> = Ready<Result<()>>;
    fn completion_status(_: &Rc<()>, _: &Rc<()>) -> CompletionState {
        CompletionState::Complete
    }
    fn wait_completion<'a>(_: &'a Rc<()>, _: &'a Rc<()>) -> Self::WaitFuture<'a> {
        ready(Ok(()))
    }
    fn wait_idle<'a>(_: &'a Rc<()>) -> Self::WaitFuture<'a> {
        ready(Ok(()))
    }
}
