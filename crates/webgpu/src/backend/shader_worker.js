// Single-slot, CPU-only SPIR-V worker. The public Rust configuration method
// supplies the cdylib wasm-bindgen ES-module URL; no WebGPU value is moved.
let configuredUrl = null, configurationEpoch = 0, nextId = 1;
let worker = null, workerUrl = null, active = null, holdNext = false;
const queued = [];
const source = `let translate,loadedUrl;const holds=new Map;self.onmessage=async({data})=>{if(data.kind==='release'){const release=holds.get(data.id);if(release){holds.delete(data.id);release();}return;}if(data.kind!=='translate')return;try{if(data.hold)await new Promise(resolve=>holds.set(data.id,resolve));if(!translate||loadedUrl!==data.glueUrl){const glue=await import(data.glueUrl);await glue.default();translate=glue.fluxel_webgpu_translate_spirv;loadedUrl=data.glueUrl;if(typeof translate!=='function')throw new Error('worker glue lacks fluxel_webgpu_translate_spirv');}self.postMessage({id:data.id,ok:true,wgsl:translate(new Uint8Array(data.bytes))});}catch(error){self.postMessage({id:data.id,ok:false,error:String(error?.message||error)});}};`;

export function holdNextTranslation() { holdNext = true; }
export function releaseActiveHold() { if (active?.hold && worker) worker.postMessage({kind:'release', id:active.id}); }

export function configureSpirvWorker(url) {
  if (typeof url !== 'string' || !url) throw new Error('WebGPU SPIR-V worker URL must be a non-empty string');
  const normalized = new URL(url, globalThis.location?.href).href;
  if (configuredUrl !== normalized) { configuredUrl = normalized; configurationEpoch += 1; }
}
function settle(request, value) { if (!request.settled) { request.settled = true; request.resolve(value); } }
function fail(request, error) { if (!request.settled) { request.settled = true; request.reject(error instanceof Error ? error : new Error(String(error))); } }
function disposeWorker() { if (worker) worker.terminate(); worker = null; workerUrl = null; }
function createWorker(url) {
  let blobUrl;
  try {
    blobUrl = URL.createObjectURL(new Blob([source], {type:'text/javascript'}));
    const next = new Worker(blobUrl, {type:'module', name:'fluxel-webgpu-shader'});
    URL.revokeObjectURL(blobUrl);
    next.onmessage = ({data}) => {
      if (worker !== next) return;
      if (!active || data?.id !== active.id) return;
      const request = active; active = null;
      if (data?.ok) settle(request, data.wgsl); else fail(request, new Error(data?.error || 'WebGPU shader worker failed'));
      pump();
    };
    next.onerror = event => {
      if (worker !== next) return;
      if (active) { const request = active; active = null; fail(request, new Error(event.message || 'WebGPU shader worker failed')); }
      disposeWorker(); pump();
    };
    worker = next; workerUrl = url; return true;
  } catch (error) { if (blobUrl) URL.revokeObjectURL(blobUrl); return error; }
}
function pump() {
  if (active) return;
  while (queued.length) {
    const request = queued.shift();
    if (request.settled) continue;
    if (!worker || workerUrl !== request.url) {
      disposeWorker(); const created = createWorker(request.url);
      if (created !== true) { fail(request, created); continue; }
    }
    active = request;
    try { worker.postMessage({kind:'translate', id:request.id, glueUrl:request.url, bytes:request.bytes, hold:request.hold}, [request.bytes]); }
    catch (error) { active = null; disposeWorker(); fail(request, error); continue; }
    return;
  }
}
export function startSpirvTranslation(bytes) {
  let resolve, reject;
  const promise = new Promise((ok, error) => { resolve = ok; reject = error; });
  const request = {id:nextId++, url:configuredUrl, epoch:configurationEpoch, hold:holdNext, bytes:null, resolve, reject, settled:false};
  holdNext = false;
  try {
    if (typeof request.url !== 'string' || !request.url) fail(request, new Error('WebGPU SPIR-V worker is not configured; call WebGpuInstance::configure_spirv_worker first'));
    else { const copy = bytes.slice(); request.bytes = copy.buffer; queued.push(request); pump(); }
  } catch (error) { fail(request, error); }
  return { promise, cancel() {
    if (request.settled) return;
    if (active === request) { active = null; disposeWorker(); }
    else { const index = queued.indexOf(request); if (index >= 0) queued.splice(index, 1); }
    // A running Naga call is not instruction-preemptible; worker termination
    // is the cancellation boundary and the following FIFO item rebuilds it.
    settle(request, null); pump();
  }};
}
