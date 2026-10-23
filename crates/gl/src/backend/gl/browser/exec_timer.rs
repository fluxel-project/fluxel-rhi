//! Browser timer-query execution and its disjoint-interval semantics.
//!
//! WebGL2 core has no timer target. Its extension supplies queryCounterEXT;
//! begin, end, and result operations use the rendering context's core methods.
//! The extension command is called only when the acquired object exposes it.
//! This module owns which commands such an
//! object must carry, how they are reflected, how a counter width is read, and
//! how one measurement is classified. A result is never handed out from an
//! interval the driver marked as disrupted: the flag is read immediately before
//! the value, and a flag that cannot be read at all yields an unknown result
//! rather than a number that might be wrong.
//!
//! The capability row is discovery's decision, and object lifetime is
//! `exec_sync.rs`'s; this module only decides what a timer command answers.

use js_sys::Function;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{WebGl2RenderingContext as Gl, WebGlQuery};

use super::super::api::{
    GlCapability, GlElapsedQueryApi, GlError, GlFamilyApi as _, GlQueryResult, GlTimestampQueryApi,
    QueryId,
};
use super::discovery::WebGl2BrowserDiscovery;

/// `TIME_ELAPSED_EXT`. The web-sys bindings name no timer target, so these are
/// spelled from the registry like the other unnamed constants in this tree.
const TIME_ELAPSED_EXT: u32 = 0x88BF;
/// `TIMESTAMP_EXT`.
const TIMESTAMP_EXT: u32 = 0x8E28;
/// `GPU_DISJOINT_EXT`, the target and parameter of the disruption flag.
const GPU_DISJOINT_EXT: u32 = 0x8FBB;
/// `QUERY_COUNTER_BITS_EXT`.
const QUERY_COUNTER_BITS_EXT: u32 = 0x8864;
/// `QUERY_RESULT_EXT`.
const QUERY_RESULT_EXT: u32 = 0x8866;
/// `QUERY_RESULT_AVAILABLE_EXT`.
const QUERY_RESULT_AVAILABLE_EXT: u32 = 0x8867;

/// Whether a recorded query target is a timer measurement.
///
/// The target word is what tells a timer query from an occlusion query both
/// when a result is classified and when an active begin/end pair is checked.
pub(super) const fn is_timer_target(target: u32) -> bool {
    target == TIME_ELAPSED_EXT || target == TIMESTAMP_EXT
}

/// The timer commands one acquired extension object must expose.
///
/// The object owns its commands, so it is retained as the receiver of every
/// call: a command invoked on the wrong receiver is not the interface the
/// browser published.
#[derive(Clone)]
pub(super) struct BrowserTimerQuery {
    object: JsValue,
    raw: Gl,
    query_counter: Function,
}

impl BrowserTimerQuery {
    /// WebGL2 places query operations on the context; only queryCounterEXT
    /// belongs to the extension object.
    pub(super) fn acquire(raw: &Gl, object: &JsValue) -> Option<Self> {
        Some(Self {
            object: object.clone(),
            raw: raw.clone(),
            query_counter: command(object, "queryCounterEXT")?,
        })
    }
    fn counter_bits(&self, target: u32) -> Option<u32> {
        counter_width(&self.raw.get_query(target, QUERY_COUNTER_BITS_EXT))
    }
    pub(super) fn elapsed_counter_bits(&self) -> Option<u32> {
        self.counter_bits(TIME_ELAPSED_EXT)
    }
    pub(super) fn timestamp_counter_bits(&self) -> Option<u32> {
        self.counter_bits(TIMESTAMP_EXT)
    }
    pub(super) fn begin_elapsed(&self, query: &WebGlQuery) -> Result<(), JsValue> {
        self.raw.begin_query(TIME_ELAPSED_EXT, query);
        Ok(())
    }
    pub(super) fn end_elapsed(&self) -> Result<(), JsValue> {
        self.raw.end_query(TIME_ELAPSED_EXT);
        Ok(())
    }
    pub(super) fn write_timestamp(&self, query: &WebGlQuery) -> Result<(), JsValue> {
        self.query_counter
            .call2(
                &self.object,
                &JsValue::from(query.clone()),
                &JsValue::from_f64(f64::from(TIMESTAMP_EXT)),
            )
            .map(|_| ())
    }
    fn result_available(&self, query: &WebGlQuery) -> Option<bool> {
        self.raw
            .get_query_parameter(query, QUERY_RESULT_AVAILABLE_EXT)
            .as_bool()
    }
    fn result_value(&self, query: &WebGlQuery) -> Option<u64> {
        self.raw
            .get_query_parameter(query, QUERY_RESULT_EXT)
            .as_f64()
            .and_then(super::exec_sync::measured_value)
    }
    fn disjoint(&self) -> Option<bool> {
        self.raw.get_parameter(GPU_DISJOINT_EXT).ok()?.as_bool()
    }
}

fn command(object: &JsValue, name: &str) -> Option<Function> {
    js_sys::Reflect::get(object, &JsValue::from_str(name))
        .ok()?
        .dyn_into::<Function>()
        .ok()
}

/// One answered counter width, if it is a usable one.
///
/// Zero is not a usable width: an interval measured in zero bits has no value
/// to report, so it is reported as no answer at all.
fn counter_width(value: &JsValue) -> Option<u32> {
    let value = value.as_f64()?;
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 || value > f64::from(u32::MAX) {
        return None;
    }
    Some(value as u32)
}

/// The counter width this context records for timer queries.
///
/// Both timer targets must answer, and the recorded number is the smaller of
/// the two: the capability covers elapsed intervals and timestamps alike, so a
/// width that only one of them answers would enable a domain whose other half
/// can never produce a result. Zero therefore means exactly "at least one
/// target has no usable width", which is what keeps the capability row honest.
pub(super) fn recorded_counter_width(raw: &Gl, timer: Option<&BrowserTimerQuery>) -> u32 {
    let width = timer
        .and_then(|timer| {
            Some(
                timer
                    .elapsed_counter_bits()?
                    .min(timer.timestamp_counter_bits()?),
            )
        })
        .unwrap_or(0);
    // Asking for a width a context cannot answer may be answered with an error.
    // That error belongs to this question, and leaving it pending would make the
    // next unrelated provider call report it as its own failure.
    let _ = raw.get_error();
    width
}

impl WebGl2BrowserDiscovery {
    fn retire_timer_disjoint(&mut self, operation: &'static str) -> Result<(), GlError> {
        let commands = self.require_timer_commands(operation)?;
        let disjoint = commands
            .disjoint()
            .ok_or_else(|| Self::validation(operation, "timer disjoint state unavailable"))?;
        if disjoint {
            for entry in self.queries.values_mut() {
                if entry.target.is_some_and(is_timer_target) {
                    entry.disjoint = true;
                }
            }
        }
        Ok(())
    }
    /// The retained timer commands, if this context proved the whole path.
    ///
    /// The capability row and the retained commands come from the same
    /// acquisition and the same counter-width read, so both must hold before a
    /// timer command may run; anything less stays rejected.
    fn require_timer_commands(
        &self,
        operation: &'static str,
    ) -> Result<BrowserTimerQuery, GlError> {
        let proved = self
            .discovery()
            .capabilities()
            .supports(GlCapability::TimerQuery);
        self.timer
            .clone()
            .filter(|_| proved)
            .ok_or(GlError::Unsupported {
                operation,
                reason: "this context did not prove a nonzero timer-query counter width",
            })
    }

    /// Classifies one timer query result, discarding disrupted intervals.
    pub(super) fn timer_result(
        &self,
        operation: &'static str,
        raw: &WebGlQuery,
    ) -> Result<GlQueryResult, GlError> {
        let commands = self.require_timer_commands(operation)?;
        // Availability is read first because the disruption flag is cleared by
        // reading it: the flag qualifies the interval that just became ready, so
        // it must be read after readiness is known and before the number.
        let availability = commands.result_available(raw);
        let (disjoint, value) = if availability == Some(true) {
            let disjoint = commands.disjoint();
            let value = match disjoint {
                Some(false) => commands.result_value(raw),
                _ => None,
            };
            (disjoint, value)
        } else {
            (None, None)
        };
        Ok(classify_measurement(availability, disjoint, value))
    }
}

/// Classifies one measurement from its three independent readings.
///
/// The reads happen on a live extension object, so this is the only place the
/// whole matrix can be decided and asserted: a driver that reports a disrupted
/// interval is a rare, unprovokable event, and a classification that only
/// exists inside a method taking a live query could not be tested at all.
///
/// An unreadable availability or disruption answer yields `Unknown` rather than
/// either extreme, because both extremes claim knowledge this context did not
/// give, and a value must never be handed out from a possibly disrupted
/// interval.
pub(super) fn classify_measurement(
    availability: Option<bool>,
    disjoint: Option<bool>,
    value: Option<u64>,
) -> GlQueryResult {
    match availability {
        None => GlQueryResult::Unknown,
        Some(false) => GlQueryResult::Pending,
        Some(true) => match disjoint {
            None => GlQueryResult::Unknown,
            Some(true) => GlQueryResult::Disjoint,
            Some(false) => match value {
                Some(value) => GlQueryResult::Available(value),
                None => GlQueryResult::Unknown,
            },
        },
    }
}

impl GlElapsedQueryApi for WebGl2BrowserDiscovery {
    fn begin_elapsed_query(&mut self, query: QueryId) -> Result<(), GlError> {
        const OP: &str = "begin-elapsed-query";
        // Fail closed before any browser side effect, then validate the object
        // and the single active-query slot before the interval starts.
        self.assert_provider_ready(OP)?;
        let commands = self.require_timer_commands(OP)?;
        self.retire_timer_disjoint(OP)?;
        let raw = self.query(OP, query)?.raw.clone();
        if self.active_query.is_some() {
            return Err(Self::validation(OP, "another query is already active"));
        }
        commands
            .begin_elapsed(&raw)
            .map_err(|value| Self::js_failure(OP, value))?;
        self.driver_error(OP)?;
        if let Some(entry) = self.queries.get_mut(&query.slot) {
            entry.target = Some(TIME_ELAPSED_EXT);
            entry.disjoint = false;
        }
        self.active_query = Some(query.slot);
        Ok(())
    }

    fn end_elapsed_query(&mut self) -> Result<(), GlError> {
        const OP: &str = "end-elapsed-query";
        self.assert_provider_ready(OP)?;
        let commands = self.require_timer_commands(OP)?;
        let Some(active) = self.active_query.take() else {
            return Err(Self::validation(OP, "no elapsed query is active"));
        };
        let target = self.queries.get(&active).and_then(|entry| entry.target);
        if !target.is_some_and(is_timer_target) {
            // The slot is restored because the active pair is still recording:
            // ending the wrong domain's query would leave the context with an
            // interval nothing can close.
            self.active_query = Some(active);
            return Err(Self::validation(
                OP,
                "the active query is not an elapsed query",
            ));
        }
        if let Err(value) = commands.end_elapsed() {
            self.active_query = Some(active);
            return Err(Self::js_failure(OP, value));
        }
        // As with core occlusion queries, retain the active slot if the
        // browser rejects the end command.  Clearing it early would make a
        // still-open interval impossible to close and would incorrectly admit
        // another query on the same context.
        if let Err(error) = self.driver_error(OP) {
            self.active_query = Some(active);
            return Err(error);
        }
        Ok(())
    }
}

impl GlTimestampQueryApi for WebGl2BrowserDiscovery {
    fn query_timestamp(&mut self, query: QueryId) -> Result<(), GlError> {
        const OP: &str = "query-timestamp";
        self.assert_provider_ready(OP)?;
        let commands = self.require_timer_commands(OP)?;
        self.retire_timer_disjoint(OP)?;
        let raw = self.query(OP, query)?.raw.clone();
        if self.active_query.is_some() {
            return Err(Self::validation(OP, "another query is already active"));
        }
        commands
            .write_timestamp(&raw)
            .map_err(|value| Self::js_failure(OP, value))?;
        self.driver_error(OP)?;
        if let Some(entry) = self.queries.get_mut(&query.slot) {
            entry.target = Some(TIMESTAMP_EXT);
            entry.disjoint = false;
        }
        Ok(())
    }
}
