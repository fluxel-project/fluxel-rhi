//! Pipeline layout admission and immutable metadata ownership.

use super::*;

impl<B: ResourceBackend> Device<B> {
    fn validate_pipeline_layout(&self, desc: &PipelineLayoutDesc<'_, B>) -> Result<()> {
        if desc.groups.len() > self.caps.limits.max_bind_groups as usize {
            return Err(Error::InvalidRange);
        }
        if desc.groups.iter().any(|v| v.device != self.id) {
            return Err(Error::WrongDevice);
        }
        let uniform_limit = self
            .caps
            .limits
            .max_dynamic_uniform_buffers_per_pipeline_layout;
        let storage_limit = self
            .caps
            .limits
            .max_dynamic_storage_buffers_per_pipeline_layout;
        if uniform_limit != 0 || storage_limit != 0 {
            let mut uniforms = 0usize;
            let mut storage = 0usize;
            for entry in desc.groups.iter().flat_map(|group| group.entries.iter()) {
                match entry.ty {
                    BindingType::UniformBuffer {
                        dynamic_offset: true,
                        ..
                    } => uniforms += 1,
                    BindingType::StorageBuffer {
                        dynamic_offset: true,
                        ..
                    } => storage += 1,
                    _ => {}
                }
            }
            if (uniform_limit != 0 && uniforms > uniform_limit as usize)
                || (storage_limit != 0 && storage > storage_limit as usize)
            {
                return Err(Error::InvalidRange);
            }
        }
        if !desc.push_constants.is_empty() && !self.caps.push_constants {
            return Err(Error::Unsupported("push constants"));
        }
        let push_constant_limit = self.caps.limits.max_push_constant_bytes;
        if push_constant_limit != 0
            && desc
                .push_constants
                .iter()
                .any(|range| range.range.end > push_constant_limit)
        {
            return Err(Error::InvalidRange);
        }
        Ok(())
    }
    /// Checks the pipeline layout's combined binding and constant budgets
    /// without creating a native pipeline layout.
    pub fn pipeline_layout_support(&self, desc: &PipelineLayoutDesc<'_, B>) -> Result<()> {
        self.validate_pipeline_layout(desc)?;
        B::pipeline_layout_support(&self.native, desc)
    }
    pub fn create_pipeline_layout(
        &self,
        desc: &PipelineLayoutDesc<'_, B>,
    ) -> Result<PipelineLayout<B>> {
        self.validate_pipeline_layout(desc)?;
        let native = B::create_pipeline_layout(&self.native, desc)?;
        Ok(self.wrap_pipeline_layout(desc, native, &[]))
    }

    /// Checks a complete native register assignment and the layout budgets.
    pub fn pipeline_layout_native_register_support(
        &self,
        desc: &PipelineLayoutDesc<'_, B>,
        registers: &[NativeBindingRegister],
    ) -> Result<()> {
        if !self.caps.native_binding_registers {
            return Err(Error::Unsupported("native binding registers"));
        }
        self.validate_pipeline_layout(desc)?;
        B::pipeline_layout_native_register_support(&self.native, desc, registers)
    }

    /// Creates a native pipeline layout with one exact register assignment
    /// per binding. Bind groups continue to use their logical binding numbers.
    pub fn create_pipeline_layout_with_native_registers(
        &self,
        desc: &PipelineLayoutDesc<'_, B>,
        registers: &[NativeBindingRegister],
    ) -> Result<PipelineLayout<B>> {
        if !self.caps.native_binding_registers {
            return Err(Error::Unsupported("native binding registers"));
        }
        self.validate_pipeline_layout(desc)?;
        let native =
            B::create_pipeline_layout_with_native_registers(&self.native, desc, registers)?;
        Ok(self.wrap_pipeline_layout(desc, native, registers))
    }

    fn wrap_pipeline_layout(
        &self,
        desc: &PipelineLayoutDesc<'_, B>,
        native: B::NativePipelineLayout,
        registers: &[NativeBindingRegister],
    ) -> PipelineLayout<B> {
        // SAFETY: every referenced layout was checked to belong to this device.
        let mut layout = unsafe { PipelineLayout::from_native(self.id, native) };
        layout.description = Some(Arc::new(PipelineLayoutDescription {
            groups: desc.groups.iter().map(|&group| group.clone()).collect(),
            push_constants: desc.push_constants.to_vec(),
            native_binding_registers: registers.to_vec(),
        }));
        layout
    }
}
