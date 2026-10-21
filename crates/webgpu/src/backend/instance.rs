//! Browser adapter and device requests through Core's static backend contract.
//!
//! `WebGpuDevice` is deliberately `Rc`-backed and therefore owner-thread local.
//! Device loss makes every completion fail, while submitted JS resources remain
//! retained until the browser queue reports completion or loss.

use std::{
    cell::{Cell, RefCell},
    future::Future,
    pin::Pin,
    rc::Rc,
    sync::Arc,
};

use fluxel_rhi_core::api::{
    self, AdapterId, AdapterInfo, Backend, DeviceCapabilities, DeviceId, DeviceLimits,
    DeviceRequest, Error, InstanceBackend, InstanceId, LaneClass, LaneId, LaneInfo, Result,
};
use wasm_bindgen::JsValue;
use wasm_bindgen::closure::Closure;
use wasm_bindgen_futures::{JsFuture, spawn_local};

use super::{
    blit::BlitCache,
    command::WebGpuEncoder,
    compute::WebGpuCompute,
    indirect::WebGpuExtras,
    js,
    query::WebGpuQuerySet,
    resource::{WebGpuBuffer, WebGpuTexture, WebGpuTextureView},
    submit::{Completion, InFlight},
    surface::{WebGpuImageLease, WebGpuSurface},
};

/// Browser WebGPU selected once by the application.
pub struct WebGpu;

/// A browser WebGPU service. The browser owns adapter selection.
#[derive(Clone)]
pub struct WebGpuInstance {
    gpu: JsValue,
}

impl WebGpuInstance {
    /// Checks that this browser exposes `navigator.gpu`.
    pub fn new() -> Result<Self> {
        let gpu = js::browser_gpu().map_err(|error| Error::Backend(js::message(&error)))?;
        Ok(Self { gpu })
    }

    /// Creates a typed Core instance on the browser owner thread.
    pub fn instance(&self) -> Result<api::Instance<WebGpu>> {
        // SAFETY: `self` is this backend's only native instance value and the
        // freshly allocated id is associated with it exactly once.
        Ok(unsafe { api::Instance::from_native(InstanceId::allocate()?, self.clone()) })
    }

    /// Requests the browser's default adapter and opens its device.
    pub async fn open_default(&self) -> Result<api::Device<WebGpu>> {
        let instance = self.instance()?;
        let adapter = instance
            .enumerate()?
            .into_iter()
            .next()
            .ok_or(Error::Unsupported("browser WebGPU adapter"))?;
        instance
            .open(DeviceRequest {
                adapter: adapter.id,
                required_features: Vec::new(),
            })
            .await
    }
}

/// A browser device and queue, both confined to the JavaScript owner thread.
#[derive(Clone)]
pub struct WebGpuDevice(pub(super) Rc<DeviceState>);

pub(super) struct DeviceState {
    pub(super) id: DeviceId,
    pub(super) device: JsValue,
    pub(super) queue: JsValue,
    pub(super) next_submission: Cell<u64>,
    pub(super) in_flight: RefCell<Vec<InFlight>>,
    pub(super) uncertain_resources: RefCell<Vec<JsValue>>,
    pub(super) lost: Cell<bool>,
    pub(super) validation_error: RefCell<Option<String>>,
    pub(super) blit_cache: RefCell<Option<BlitCache>>,
    validation_listener: RefCell<Option<Closure<dyn FnMut(JsValue)>>>,
}

impl Backend for WebGpu {
    type NativeDevice = WebGpuDevice;
    type NativeBuffer = WebGpuBuffer;
    type NativeTexture = WebGpuTexture;
    type NativeTextureView = WebGpuTextureView;
    type NativeSampler = JsValue;
    type NativeBindGroupLayout = JsValue;
    type NativeBindGroup = JsValue;
    type NativePipelineLayout = JsValue;
    type NativeShader = JsValue;
    type NativeRasterPipeline = JsValue;
    type NativeComputePipeline = JsValue;
    type NativeMeshPipeline = JsValue;
    type NativeRayPipeline = JsValue;
    type NativePipelineCache = JsValue;
    type NativeAcceleration = JsValue;
    type NativeSparseAllocation = JsValue;
    type NativeQuerySet = WebGpuQuerySet;
    type NativeEncoder = WebGpuEncoder;
    type NativeCompletion = Completion;
    type NativeSurface = WebGpuSurface;
    type NativeSurfaceImage = WebGpuImageLease;
    type NativeUncertainty = ();
    type Compute = WebGpuCompute;
    type Mesh = api::Unavailable;
    type Ray = api::Unavailable;
    type Extras = WebGpuExtras;
}

impl InstanceBackend for WebGpu {
    type NativeInstance = WebGpuInstance;
    type OpenFuture<'a> =
        Pin<Box<dyn Future<Output = Result<(WebGpuDevice, DeviceCapabilities)>> + 'a>>;

    fn enumerate(_: &WebGpuInstance, scope: InstanceId) -> Result<Vec<AdapterInfo>> {
        // WebGPU selects its adapter asynchronously. The sole ordinal denotes
        // that browser-controlled selection, not a physical adapter list.
        Ok(vec![AdapterInfo {
            id: AdapterId {
                instance: scope,
                ordinal: 0,
            },
            name: "Browser WebGPU default adapter".into(),
        }])
    }

    fn open<'a>(
        instance: &'a WebGpuInstance,
        device_id: DeviceId,
        request: &'a DeviceRequest,
    ) -> Self::OpenFuture<'a> {
        Box::pin(async move {
            if request.adapter.ordinal != 0 {
                return Err(Error::Unsupported("explicit browser adapter"));
            }
            let adapter_promise = js::request_adapter(&instance.gpu, None)
                .map_err(|error| Error::Backend(js::message(&error)))?;
            let adapter = JsFuture::from(adapter_promise)
                .await
                .map_err(|error| Error::Backend(js::message(&error)))?;
            if adapter.is_null() || adapter.is_undefined() {
                return Err(Error::Unsupported("browser WebGPU adapter"));
            }
            let indirect_first_instance = js::has_feature(&adapter, "indirect-first-instance");
            let mut requested = request
                .required_features
                .iter()
                .map(|name| (*name).to_owned())
                .collect::<Vec<_>>();
            if indirect_first_instance
                && !requested
                    .iter()
                    .any(|name| name == "indirect-first-instance")
            {
                requested.push("indirect-first-instance".into());
            }
            let device_promise = js::request_device(&adapter, &requested, &[])
                .map_err(|error| Error::Backend(js::message(&error)))?;
            let device = JsFuture::from(device_promise)
                .await
                .map_err(|error| Error::Backend(js::message(&error)))?;
            let queue = js::queue(&device).map_err(|error| Error::Backend(js::message(&error)))?;
            let lost = js::lost(&device).map_err(|error| Error::Backend(js::message(&error)))?;
            let lane = LaneId {
                device: device_id,
                ordinal: 0,
            };
            let limits = js::property(&device, "limits")
                .map_err(|error| Error::Backend(js::message(&error)))?;
            let caps = DeviceCapabilities {
                compute: true,
                storage_buffers: true,
                storage_textures: true,
                indirect_draw: indirect_first_instance,
                occlusion: true,
                lanes: Arc::from([LaneInfo {
                    id: lane,
                    class: LaneClass::Graphics,
                    routes: vec![(lane, api::WaitRoute::SameQueue)],
                }]),
                limits: DeviceLimits {
                    max_bind_groups: limit_u32(&limits, "maxBindGroups"),
                    max_bindings_per_group: limit_u32(&limits, "maxBindingsPerBindGroup"),
                    max_color_attachments: limit_u32(&limits, "maxColorAttachments"),
                    max_texture_dimension_2d: limit_u32(&limits, "maxTextureDimension2D"),
                    max_texture_array_layers: limit_u32(&limits, "maxTextureArrayLayers"),
                    max_buffer_bytes: limit_u64(&limits, "maxBufferSize"),
                    max_vertex_buffers: limit_u32(&limits, "maxVertexBuffers"),
                    max_vertex_attributes: limit_u32(&limits, "maxVertexAttributes"),
                    uniform_offset_alignment: limit_u64(&limits, "minUniformBufferOffsetAlignment"),
                    storage_offset_alignment: limit_u64(&limits, "minStorageBufferOffsetAlignment"),
                    ..Default::default()
                },
                ..Default::default()
            };
            let native = WebGpuDevice(Rc::new(DeviceState {
                id: device_id,
                device,
                queue,
                next_submission: Cell::new(1),
                in_flight: RefCell::new(Vec::new()),
                uncertain_resources: RefCell::new(Vec::new()),
                lost: Cell::new(false),
                validation_error: RefCell::new(None),
                blit_cache: RefCell::new(None),
                validation_listener: RefCell::new(None),
            }));
            let weak_for_errors = Rc::downgrade(&native.0);
            let listener =
                Closure::<dyn FnMut(JsValue)>::new(move |event: JsValue| {
                    if let Some(state) = weak_for_errors.upgrade() {
                        let error = js::property(&event, "error").unwrap_or(event);
                        let message = js::message(&error);
                        *state.validation_error.borrow_mut() = Some(message.clone());
                        for flight in state.in_flight.borrow().iter() {
                            flight.completion.settled(api::CompletionState::Failed(
                                Error::Backend(message.clone()),
                            ));
                        }
                    }
                });
            js::call(
                &native.0.device,
                "addEventListener",
                &[
                    JsValue::from_str("uncapturederror"),
                    listener.as_ref().clone(),
                ],
            )
            .map_err(|error| Error::Backend(js::message(&error)))?;
            *native.0.validation_listener.borrow_mut() = Some(listener);
            let weak = Rc::downgrade(&native.0);
            spawn_local(async move {
                let _ = JsFuture::from(lost).await;
                if let Some(state) = weak.upgrade() {
                    state.lost.set(true);
                    for flight in state.in_flight.borrow().iter() {
                        flight
                            .completion
                            .settled(api::CompletionState::Failed(Error::DeviceLost));
                    }
                }
            });
            Ok((native, caps))
        })
    }
}

fn limit_u32(limits: &JsValue, name: &str) -> u32 {
    limit_u64(limits, name).try_into().unwrap_or(0)
}

fn limit_u64(limits: &JsValue, name: &str) -> u64 {
    js::property(limits, name)
        .ok()
        .and_then(|value| value.as_f64())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .map_or(0, |value| value as u64)
}

impl fluxel_rhi_core::api::ExecutionBackend for WebGpu {
    type Mode = fluxel_rhi_core::api::DeferredExecution;
}
