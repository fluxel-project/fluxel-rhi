//! Lowers immediate data using WGSL byte offsets and HLSL `packoffset`.

use alloc::{format, string::String, vec::Vec};
use core::fmt;

use crate::{proc::NameKey, GlobalVariable, Handle, Module, Scalar, Type, TypeInner, VectorSize};

use super::{BackendResult, Error, Writer};

struct Word {
    destination: String,
    offset: u32,
    scalar: Scalar,
}

impl<W: fmt::Write> Writer<'_, W> {
    pub(super) fn write_immediate_global(
        &mut self,
        module: &Module,
        handle: Handle<GlobalVariable>,
    ) -> BackendResult {
        let target = self.options.push_constants_target.as_ref().ok_or_else(|| {
            Error::Unimplemented("WGSL immediate data requires a bind target".into())
        })?;
        let global = &module.global_variables[handle];
        let name = self.names[&NameKey::GlobalVariable(handle)].clone();
        let block = self.namer.call("naga_immediate_words");
        let loader = self.namer.call("naga_load_immediate");
        let mut words = Vec::new();
        self.immediate_words(module, global.ty, &name, 0, &mut words)?;

        write!(self.out, "cbuffer {block} : register(b{}", target.register)?;
        if target.space != 0 {
            write!(self.out, ", space{}", target.space)?;
        }
        writeln!(self.out, ") {{")?;
        let mut names = Vec::with_capacity(words.len());
        for word in &words {
            let name = self.namer.call("naga_immediate_word");
            let register = word.offset / 16;
            let component = ["x", "y", "z", "w"][(word.offset % 16 / 4) as usize];
            writeln!(
                self.out,
                "    uint {name} : packoffset(c{register}.{component});"
            )?;
            names.push(name);
        }
        writeln!(self.out, "}};")?;
        write!(self.out, "static ")?;
        self.write_type(module, global.ty)?;
        writeln!(self.out, " {name};")?;
        writeln!(self.out, "void {loader}() {{")?;
        for (word, name) in words.iter().zip(names) {
            let value = match word.scalar {
                Scalar::F32 => format!("asfloat({name})"),
                Scalar::I32 => format!("asint({name})"),
                Scalar::U32 => name,
                _ => return Err(Error::Unimplemented("WGSL immediate scalar".into())),
            };
            writeln!(self.out, "    {} = {value};", word.destination)?;
        }
        writeln!(self.out, "}}")?;
        self.immediate_loaders.push((handle, loader));
        Ok(())
    }

    fn immediate_words(
        &self,
        module: &Module,
        ty: Handle<Type>,
        destination: &str,
        offset: u32,
        words: &mut Vec<Word>,
    ) -> BackendResult {
        match module.types[ty].inner {
            TypeInner::Scalar(scalar) => words.push(Word {
                destination: destination.into(),
                offset,
                scalar,
            }),
            TypeInner::Vector { size, scalar } => {
                append_vector(words, destination, offset, size, scalar)
            }
            TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => {
                for column in 0..columns as u32 {
                    append_vector(
                        words,
                        &format!("{destination}[{column}]"),
                        offset + column * column_stride(rows),
                        rows,
                        scalar,
                    );
                }
            }
            TypeInner::Struct { ref members, .. } => {
                for (index, member) in members.iter().enumerate() {
                    let name = &self.names[&NameKey::StructMember(ty, index as u32)];
                    let destination = format!("{destination}.{name}");
                    if let TypeInner::Matrix {
                        columns,
                        rows: VectorSize::Bi,
                        scalar,
                    } = module.types[member.ty].inner
                    {
                        // HLSL writer represents these members as separate vec2 fields.
                        for column in 0..columns as u32 {
                            append_vector(
                                words,
                                &format!("{destination}_{column}"),
                                offset + member.offset + column * 8,
                                VectorSize::Bi,
                                scalar,
                            );
                        }
                    } else {
                        self.immediate_words(
                            module,
                            member.ty,
                            &destination,
                            offset + member.offset,
                            words,
                        )?;
                    }
                }
            }
            _ => return Err(Error::Unimplemented("WGSL immediate store type".into())),
        }
        Ok(())
    }
}

fn column_stride(rows: VectorSize) -> u32 {
    if rows == VectorSize::Bi {
        8
    } else {
        16
    }
}

fn append_vector(
    words: &mut Vec<Word>,
    destination: &str,
    offset: u32,
    size: VectorSize,
    scalar: Scalar,
) {
    for component in 0..size as u32 {
        words.push(Word {
            destination: format!("{destination}[{component}]"),
            offset: offset + component * 4,
            scalar,
        });
    }
}
