//! Immediate native GL compute commands.
//!
//! The public encoder resolves every object handle before a command reaches
//! this module. `State` therefore contains only owner-local GL carriers and
//! scalar state; it is deliberately not a second command stream.

use crate::api::command::record::ImmediateWrite;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::ObjectId;
use crate::backend::gl::platform::GlObjectName;

use super::driver::{NativeBoundGroup, NativeOwnedProvider, NativePlatformContext};

/// A bind group selected for the current compute scope.
#[derive(Clone)]
pub(super) struct BoundGroup {
    pub(super) index: u32,
    pub(super) group: ObjectId,
    pub(super) name: GlObjectName,
    pub(super) dynamic_offsets: Vec<u32>,
}

/// Mutable state for one open native compute encoder.
///
/// This state stays on the context-owning thread and is consumed as soon as a
/// dispatch is encoded. It must never be copied into a submission token.
#[derive(Default)]
pub(super) struct State {
    pub(super) pipeline: Option<GlObjectName>,
    pub(super) groups: Vec<BoundGroup>,
    pub(super) immediates: Vec<ImmediateWrite>,
    pub(super) active: bool,
}

impl State {
    pub(super) fn reset(&mut self) {
        *self = Self::default();
        self.active = true;
    }

    pub(super) fn end(&mut self) {
        *self = Self::default();
    }

    pub(super) fn set_group(&mut self, group: BoundGroup) {
        self.groups.retain(|current| current.index != group.index);
        self.groups.push(group);
    }

    pub(super) fn set_immediate(&mut self, write: ImmediateWrite) {
        self.immediates
            .retain(|current| current.offset != write.offset);
        self.immediates.push(write);
    }
}

/// Fully owned operands for immediate compute lowering.
///
/// Object names have already been checked against the originating GL device.
/// The owner resolves them against its live tables immediately before calling
/// the GL provider, so a destroyed carrier cannot be used later.
pub(super) enum Command {
    Begin,
    SetPipeline(GlObjectName),
    SetBindGroup {
        index: u32,
        group: ObjectId,
        name: GlObjectName,
        dynamic_offsets: Vec<u32>,
    },
    SetImmediates(ImmediateWrite),
    Dispatch {
        x: u32,
        y: u32,
        z: u32,
    },
    DispatchIndirect {
        arguments: GlObjectName,
        offset: u64,
    },
    BeginQuery {
        set: GlObjectName,
        index: u32,
    },
    EndQuery {
        set: GlObjectName,
        index: u32,
    },
    WriteTimestamp {
        set: GlObjectName,
        index: u32,
    },
    PushDebugGroup(String),
    PopDebugGroup,
    InsertDebugMarker(String),
    End,
}

/// Executes one compute command on the context owner.
///
/// There is no completed-command object here. A dispatch has already reached
/// GL when this returns; submit only publishes the completion fence for the
/// accumulated native work.
pub(super) fn execute<C: NativePlatformContext>(
    owner: &mut NativeOwnedProvider<C>,
    command: Command,
) -> RhiResult<()> {
    const OP: &str = "NativeProviderOwner::typed compute";
    match command {
        Command::Begin => {
            owner.ready(OP)?;
            owner.typed_compute.reset();
            Ok(())
        }
        Command::End => {
            owner.ready(OP)?;
            require_active(owner, OP)?;
            owner.typed_compute.end();
            Ok(())
        }
        Command::SetPipeline(pipeline) => {
            require_active(owner, OP)?;
            owner.typed_compute.pipeline = Some(pipeline);
            Ok(())
        }
        Command::SetBindGroup {
            index,
            group,
            name,
            dynamic_offsets,
        } => {
            require_active(owner, OP)?;
            owner.typed_compute.set_group(BoundGroup {
                index,
                group,
                name,
                dynamic_offsets,
            });
            Ok(())
        }
        Command::SetImmediates(write) => {
            require_active(owner, OP)?;
            owner.typed_compute.set_immediate(write);
            Ok(())
        }
        Command::Dispatch { x, y, z } => {
            owner.ready(OP)?;
            let (program, _groups, bound) = dispatch_state(owner, OP)?;
            owner.execute_compute_dispatch(
                program,
                crate::backend::gl::state::CanonicalBlockId::object(program),
                crate::backend::gl::api::GlDispatchGroups([x, y, z]),
                bound,
            )
        }
        Command::DispatchIndirect { arguments, offset } => {
            owner.ready(OP)?;
            let (program, _groups, bound) = dispatch_state(owner, OP)?;
            owner.typed_compute_prepare(
                program,
                crate::backend::gl::state::CanonicalBlockId::object(program),
                bound,
            )?;
            use crate::backend::gl::api::GlDispatchIndirectApi as _;
            let buffer = owner.buffer_id(arguments, OP)?;
            let (_, descriptor) = owner.provider.buffer(OP, buffer).map_err(|error| {
                RhiError::new(RhiErrorKind::BackendFailure, format!("{OP}: {error:?}"))
            })?;
            owner
                .provider
                .dispatch_indirect(crate::backend::gl::api::GlDispatchIndirectCommand {
                    range: crate::backend::gl::api::GlBufferRange {
                        buffer,
                        offset: 0,
                        size: descriptor.size,
                    },
                    command_offset: offset,
                })
                .map_err(|error| {
                    RhiError::new(RhiErrorKind::BackendFailure, format!("{OP}: {error:?}"))
                })
        }
        Command::BeginQuery { set, index } => {
            owner.ready(OP)?;
            require_active(owner, OP)?;
            let (query, ty) = owner.query_id(set, index, None, OP)?;
            if !matches!(ty, crate::api::query::QueryType::Occlusion) {
                return Err(unsupported(
                    OP,
                    "native GL only lowers occlusion begin/end query sets",
                ));
            }
            let state_key = owner.canonical(OP)?;
            owner.execute_query_begin(query, ty, state_key)
        }
        Command::EndQuery { set, index } => {
            owner.ready(OP)?;
            require_active(owner, OP)?;
            let (_, ty) = owner.query_id(set, index, None, OP)?;
            if !matches!(ty, crate::api::query::QueryType::Occlusion) {
                return Err(unsupported(
                    OP,
                    "native GL only lowers occlusion begin/end query sets",
                ));
            }
            owner.execute_query_end(ty)
        }
        Command::WriteTimestamp { set, index } => {
            owner.ready(OP)?;
            require_active(owner, OP)?;
            let (query, _) = owner.query_id(
                set,
                index,
                Some(crate::api::query::QueryType::Timestamp),
                OP,
            )?;
            owner.execute_timestamp(query)
        }
        Command::PushDebugGroup(_) | Command::PopDebugGroup | Command::InsertDebugMarker(_) => {
            Err(unsupported(
                OP,
                "native GL debug-marker commands require a verified KHR_debug entry-point executor",
            ))
        }
    }
}

fn require_active<C: NativePlatformContext>(
    owner: &NativeOwnedProvider<C>,
    operation: &'static str,
) -> RhiResult<()> {
    if owner.typed_compute.active {
        Ok(())
    } else {
        Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "native GL compute command is outside an active compute scope",
        )
        .at(operation))
    }
}

fn dispatch_state<C: NativePlatformContext>(
    owner: &NativeOwnedProvider<C>,
    operation: &'static str,
) -> RhiResult<(
    crate::backend::gl::api::ProgramId,
    Vec<BoundGroup>,
    Vec<NativeBoundGroup>,
)> {
    require_active(owner, operation)?;
    if !owner.typed_compute.immediates.is_empty() {
        return Err(unsupported(
            operation,
            "native GL has no verified immediate-data lowering",
        ));
    }
    let pipeline = owner.typed_compute.pipeline.ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::InvalidUsage,
            "native GL typed dispatch requires a compute pipeline",
        )
        .at(operation)
    })?;
    let program = owner
        .compute_pipelines
        .get(&pipeline.raw())
        .copied()
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "native GL compute pipeline backing is not live",
            )
            .at(operation)
        })?;
    let groups = owner.typed_compute.groups.clone();
    let mut bound = Vec::with_capacity(groups.len());
    for group in &groups {
        if !owner.bind_groups.contains_key(&group.name.raw()) {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "native GL bind group backing is not live",
            )
            .at(operation));
        }
        bound.push(NativeBoundGroup {
            packet: crate::backend::gl::state::BoundGroupPacket {
                group: group.group,
                name: group.name,
                index: group.index,
                dynamic_offsets: group.dynamic_offsets.clone(),
                program_identity: crate::backend::gl::state::CanonicalBlockId::object(program),
                dependencies: std::collections::BTreeSet::new(),
            },
        });
    }
    Ok((program, groups, bound))
}

fn unsupported(operation: &'static str, message: &'static str) -> RhiError {
    RhiError::new(RhiErrorKind::Unsupported, message).at(operation)
}
