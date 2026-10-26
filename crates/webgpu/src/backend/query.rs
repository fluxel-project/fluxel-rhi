//! Native WebGPU occlusion and pass-boundary timestamp queries.
//!
//! WebGPU ties occlusion queries to the pass descriptor. The encoder records
//! that identity at pass begin and rejects attempts to begin a query from a
//! different set while the pass is live.

use std::ops::Range;

use fluxel_rhi_core::api::{self, Error, Result};
use js_sys::Object;
use wasm_bindgen::JsValue;

use super::{
    command::WebGpuEncoder,
    instance::{WebGpu, WebGpuDevice},
    js,
    resource::WebGpuBuffer,
};

/// The browser query-set object together with the count validated by Core.
#[derive(Clone)]
pub struct WebGpuQuerySet {
    pub(super) raw: JsValue,
    pub(super) count: u32,
    pub(super) kind: api::QueryKind,
    pub(super) results: Option<JsValue>,
}

fn error(value: JsValue) -> Error {
    Error::Backend(js::message(&value))
}

fn descriptor() -> JsValue {
    Object::new().into()
}

fn field(object: &JsValue, name: &str, value: JsValue) -> Result<()> {
    js::set(object, name, &value).map_err(error)
}

fn valid_slots(set: &WebGpuQuerySet, slots: Range<u32>) -> Result<()> {
    if slots.start >= slots.end || slots.end > set.count {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

impl fluxel_rhi_core::backend::QueryBackend for WebGpu {
    fn query_supported(device: &WebGpuDevice, kind: api::QueryKind) -> bool {
        match kind {
            api::QueryKind::Occlusion => true,
            api::QueryKind::Timestamp => js::has_feature(&device.0.device, "timestamp-query"),
            api::QueryKind::PipelineStatistics => false,
        }
    }

    fn timestamp_frequency(device: &WebGpuDevice, lane: api::LaneId) -> Option<u64> {
        (lane.device == device.0.id
            && lane.ordinal == 0
            && Self::query_supported(device, api::QueryKind::Timestamp))
        .then_some(1_000_000_000)
    }

    fn create_query_set(device: &WebGpuDevice, desc: &api::QueryDesc) -> Result<WebGpuQuerySet> {
        if !Self::query_supported(device, desc.kind) {
            return Err(Error::Unsupported("WebGPU query type/count"));
        }
        if desc.count == 0 || desc.count > 4096 {
            return Err(Error::InvalidRange);
        }
        let native = descriptor();
        field(
            &native,
            "type",
            JsValue::from_str(if desc.kind == api::QueryKind::Timestamp {
                "timestamp"
            } else {
                "occlusion"
            }),
        )?;
        field(
            &native,
            "count",
            JsValue::from_f64(if desc.kind == api::QueryKind::Occlusion {
                4096.0
            } else {
                desc.count as f64
            }),
        )?;
        let raw = js::call1(&device.0.device, "createQuerySet", &native).map_err(error)?;
        let results = if desc.kind == api::QueryKind::Occlusion {
            let buffer = descriptor();
            field(
                &buffer,
                "size",
                JsValue::from_f64(f64::from(desc.count) * 8.0),
            )?;
            field(&buffer, "usage", JsValue::from_f64(140.0))?;
            Some(js::call1(&device.0.device, "createBuffer", &buffer).map_err(error)?)
        } else {
            None
        };
        Ok(WebGpuQuerySet {
            raw,
            count: desc.count,
            kind: desc.kind,
            results,
        })
    }

    fn write_timestamp(encoder: &mut WebGpuEncoder, set: &WebGpuQuerySet, slot: u32) -> Result<()> {
        if encoder.pass.is_some() || set.kind != api::QueryKind::Timestamp {
            return Err(Error::InvalidState);
        }
        if slot >= set.count {
            return Err(Error::InvalidRange);
        }
        // The standard exposes timestampWrites at pass boundaries, rather than
        // an arbitrary command-encoder writeTimestamp method. An empty native
        // compute pass supplies precisely this ordered boundary without CPU waits.
        let writes = descriptor();
        field(&writes, "querySet", set.raw.clone())?;
        field(
            &writes,
            "beginningOfPassWriteIndex",
            JsValue::from_f64(slot as f64),
        )?;
        let desc = descriptor();
        field(&desc, "timestampWrites", writes)?;
        let pass = js::call1(&encoder.raw, "beginComputePass", &desc).map_err(error)?;
        js::call0(&pass, "end").map_err(error)?;
        encoder.keep_alive.push(set.raw.clone());
        Ok(())
    }

    fn begin_occlusion(encoder: &mut WebGpuEncoder, set: &WebGpuQuerySet, slot: u32) -> Result<()> {
        if slot >= set.count || set.kind != api::QueryKind::Occlusion {
            return Err(Error::InvalidRange);
        }
        if encoder.occlusion_active {
            return Err(Error::InvalidState);
        }
        if !encoder
            .occlusion_logical_set
            .as_ref()
            .is_some_and(|active| Object::is(active, &set.raw))
        {
            return Err(Error::InvalidState);
        }
        if encoder.occlusion_next_index >= 4096 {
            // The native set has exhausted its per-pass physical slots. End
            // the pass and flush completed queries before restoring its state;
            // no logical query is active at this point.
            let suspended = super::pass_split::suspend(encoder)?;
            super::pass_split::resume(encoder, suspended, None, true)?;
        }
        let pass = encoder.pass.as_ref().ok_or(Error::InvalidState)?;
        let physical = encoder.occlusion_next_index;
        js::call1(
            pass,
            "beginOcclusionQuery",
            &JsValue::from_f64(physical as f64),
        )
        .map_err(error)?;
        encoder.occlusion_next_index += 1;
        let native = encoder
            .occlusion_query_set
            .as_ref()
            .ok_or(Error::InvalidState)?
            .clone();
        encoder.occlusion_active_query = Some(super::query_segments::QuerySegments {
            set: set.clone(),
            slot,
            native: vec![(native, physical)],
        });
        encoder.occlusion_active = true;
        encoder.keep_alive.push(set.raw.clone());
        Ok(())
    }

    fn reset_queries(
        encoder: &mut WebGpuEncoder,
        set: &WebGpuQuerySet,
        slots: Range<u32>,
    ) -> Result<()> {
        valid_slots(set, slots.clone())?;
        if encoder.pass.is_some() {
            return Err(Error::InvalidState);
        }
        if let Some(results) = &set.results {
            js::call(
                &encoder.raw,
                "clearBuffer",
                &[
                    results.clone(),
                    JsValue::from_f64(f64::from(slots.start) * 8.0),
                    JsValue::from_f64(f64::from(slots.end - slots.start) * 8.0),
                ],
            )
            .map_err(error)?;
            encoder.keep_alive.push(results.clone());
        }
        Ok(())
    }

    fn end_occlusion(encoder: &mut WebGpuEncoder) -> Result<()> {
        if !encoder.occlusion_active {
            return Err(Error::InvalidState);
        }
        let pass = encoder.pass.as_ref().ok_or(Error::InvalidState)?;
        js::call0(pass, "endOcclusionQuery").map_err(error)?;
        encoder.occlusion_active = false;
        encoder.occlusion_pending.push(
            encoder
                .occlusion_active_query
                .take()
                .ok_or(Error::InvalidState)?,
        );
        Ok(())
    }

    fn begin_statistics(_: &mut WebGpuEncoder, _: &WebGpuQuerySet, _: u32) -> Result<()> {
        Err(Error::Unsupported("WebGPU pipeline statistics"))
    }

    fn end_statistics(_: &mut WebGpuEncoder) -> Result<()> {
        Err(Error::Unsupported("WebGPU pipeline statistics"))
    }

    fn resolve_queries(
        encoder: &mut WebGpuEncoder,
        set: &WebGpuQuerySet,
        slots: Range<u32>,
        destination: &WebGpuBuffer,
        offset: u64,
    ) -> Result<()> {
        valid_slots(set, slots.clone())?;
        if encoder.pass.is_some()
            || !destination.usage.contains(api::BufferUsage::QUERY_RESOLVE)
            || offset % 256 != 0
        {
            return Err(Error::InvalidRange);
        }
        let bytes = u64::from(slots.end - slots.start)
            .checked_mul(8)
            .ok_or(Error::InvalidRange)?;
        if offset
            .checked_add(bytes)
            .is_none_or(|end| end > destination.size)
        {
            return Err(Error::InvalidRange);
        }
        if let Some(results) = &set.results {
            js::call(
                &encoder.raw,
                "copyBufferToBuffer",
                &[
                    results.clone(),
                    JsValue::from_f64(f64::from(slots.start) * 8.0),
                    destination.raw.clone(),
                    JsValue::from_f64(offset as f64),
                    JsValue::from_f64(bytes as f64),
                ],
            )
            .map_err(error)?;
            encoder
                .keep_alive
                .extend([results.clone(), destination.raw.clone()]);
            return Ok(());
        }
        js::call(
            &encoder.raw,
            "resolveQuerySet",
            &[
                set.raw.clone(),
                JsValue::from_f64(slots.start as f64),
                JsValue::from_f64((slots.end - slots.start) as f64),
                destination.raw.clone(),
                JsValue::from_f64(offset as f64),
            ],
        )
        .map_err(error)?;
        encoder
            .keep_alive
            .extend([set.raw.clone(), destination.raw.clone()]);
        Ok(())
    }
}
