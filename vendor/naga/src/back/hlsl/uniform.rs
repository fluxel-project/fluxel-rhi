//! Lower WGSL half-precision uniform reads through their actual byte offsets.

use alloc::string::String;
use core::{fmt, mem};

use super::{super::FunctionCtx, storage::StorageAccess, BackendResult, Error, Writer};
use crate::{
    proc::NameKey, ArraySize, Expression, GlobalVariable, Handle, Module, Scalar, Type, TypeInner,
    UniqueArena,
};

pub(super) struct RawUniform {
    word: String,
    half: String,
}

pub(super) fn contains_f16(types: &UniqueArena<Type>, ty: Handle<Type>) -> bool {
    match &types[ty].inner {
        TypeInner::Scalar(scalar)
        | TypeInner::Vector { scalar, .. }
        | TypeInner::Matrix { scalar, .. } => *scalar == Scalar::F16,
        TypeInner::Array { base, .. } | TypeInner::BindingArray { base, .. } => {
            contains_f16(types, *base)
        }
        TypeInner::Struct { members, .. } => {
            members.iter().any(|member| contains_f16(types, member.ty))
        }
        _ => false,
    }
}

impl<W: fmt::Write> Writer<'_, W> {
    pub(super) fn write_raw_uniform(
        &mut self,
        module: &Module,
        handle: Handle<GlobalVariable>,
    ) -> BackendResult {
        let global = &module.global_variables[handle];
        let target = self
            .options
            .resolve_resource_binding(global.binding.as_ref().unwrap())
            .unwrap();
        let (ty, binding_count) = match module.types[global.ty].inner {
            TypeInner::BindingArray {
                base,
                size: ArraySize::Constant(size),
            } => (base, Some(target.binding_array_size.unwrap_or(size.get()))),
            TypeInner::BindingArray { .. } => {
                return Err(Error::Unimplemented(
                    "runtime-sized uniform binding arrays".into(),
                ));
            }
            _ => (global.ty, None),
        };
        let bytes = module.types[ty]
            .inner
            .try_size(module.to_ctx())
            .ok_or_else(|| Error::Unimplemented("WGSL uniform size overflow".into()))?;
        let rows = bytes.div_ceil(16);
        if !(1..=4096).contains(&rows) {
            return Err(Error::Unimplemented(
                "WGSL uniform exceeds the DX12 constant buffer size".into(),
            ));
        }

        // A uint4 exactly matches one HLSL cbuffer row. The only layout-sensitive
        // operations are then the WGSL byte offsets in the access chain.
        let raw_type = self.namer.call("naga_uniform_rows");
        let word = self.namer.call("naga_uniform_word");
        let half = self.namer.call("naga_uniform_half");
        let name = &self.names[&NameKey::GlobalVariable(handle)];
        writeln!(self.out, "struct {raw_type} {{ uint4 rows[{rows}]; }};")?;
        write!(self.out, "ConstantBuffer<{raw_type}> {name}")?;
        if let Some(count) = binding_count {
            write!(self.out, "[{count}]")?;
        }
        writeln!(
            self.out,
            " : register(b{}, space{});",
            target.register, target.space
        )?;
        writeln!(
            self.out,
            "uint {word}(ConstantBuffer<{raw_type}> buffer, uint offset) {{"
        )?;
        writeln!(
            self.out,
            "    return buffer.rows[offset >> 4u][(offset >> 2u) & 3u];"
        )?;
        writeln!(self.out, "}}")?;
        writeln!(
            self.out,
            "float16_t {half}(ConstantBuffer<{raw_type}> buffer, uint offset) {{"
        )?;
        writeln!(
            self.out,
            "    return asfloat16(uint16_t({word}(buffer, offset) >> ((offset & 2u) * 8u)));"
        )?;
        writeln!(self.out, "}}")?;
        self.raw_uniforms.insert(handle, RawUniform { word, half });
        Ok(())
    }

    pub(super) fn is_raw_uniform(
        &self,
        mut expr: Handle<Expression>,
        func_ctx: &FunctionCtx,
    ) -> bool {
        if self.raw_uniforms.is_empty() {
            return false;
        }
        loop {
            match func_ctx.expressions[expr] {
                Expression::GlobalVariable(global) => {
                    return self.raw_uniforms.contains_key(&global)
                }
                Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
                    expr = base
                }
                _ => return false,
            }
        }
    }

    pub(super) fn write_raw_uniform_scalar(
        &mut self,
        module: &Module,
        access: StorageAccess,
        scalar: Scalar,
        func_ctx: &FunctionCtx,
    ) -> BackendResult {
        let uniform = &self.raw_uniforms[&access.global];
        let cast = match scalar {
            Scalar::F16 | Scalar::U32 => None,
            Scalar::I32 => Some("asint"),
            Scalar::F32 => Some("asfloat"),
            _ => return Err(Error::Unimplemented("WGSL raw uniform scalar type".into())),
        };
        let loader = if scalar == Scalar::F16 {
            &uniform.half
        } else {
            &uniform.word
        };
        if let Some(cast) = cast {
            write!(self.out, "{cast}(")?;
        }
        write!(self.out, "{loader}(")?;
        let chain = mem::take(&mut self.temp_access_chain);
        self.write_buffer_resource(module, access, func_ctx)?;
        write!(self.out, ", ")?;
        self.write_storage_address(module, &chain, func_ctx)?;
        self.temp_access_chain = chain;
        write!(self.out, ")")?;
        if cast.is_some() {
            write!(self.out, ")")?;
        }
        Ok(())
    }
}
