use crate::gltf;
use fluxel_rhi::api::*;

pub const INDEXED_INDIRECT_STRIDE: u32 = 20;
pub const PLANT_INSTANCES: u32 = 2048;
pub(crate) const PLANTS: &[u8] = include_bytes!("../assets/models/plants.gltf");
pub(crate) const GROUND: &[u8] = include_bytes!("../assets/models/plane_circle.gltf");
pub(crate) const SKY: &[u8] = include_bytes!("../assets/models/sphere.gltf");
pub(crate) const PLANT_TEX: &[u8] =
    include_bytes!("../assets/textures/texturearray_plants_rgba.ktx");
pub(crate) const GROUND_TEX: &[u8] = include_bytes!("../assets/textures/ground_dry_rgba.ktx");
pub(crate) const PLANTS_VS: &[u8] = include_bytes!("../assets/10_indirect_draw/plants_vs.dxil");
pub(crate) const PLANTS_PS: &[u8] = include_bytes!("../assets/10_indirect_draw/plants_ps.dxil");
pub(crate) const GROUND_VS: &[u8] = include_bytes!("../assets/10_indirect_draw/ground_vs.dxil");
pub(crate) const GROUND_PS: &[u8] = include_bytes!("../assets/10_indirect_draw/ground_ps.dxil");
pub(crate) const SKY_VS: &[u8] = include_bytes!("../assets/10_indirect_draw/sky_vs.dxil");
pub(crate) const SKY_PS: &[u8] = include_bytes!("../assets/10_indirect_draw/sky_ps.dxil");

/// The original plant/ground/sky geometry and the native
/// `DrawIndexedIndirect` ABI.  Creation and upload are typed API operations;
/// executing `arguments` remains gated by `RasterExtrasOps`.
pub struct IndirectScene<B: ResourceBackend> {
    pub plant_vertices: Buffer<B>,
    pub plant_indices: Buffer<B>,
    pub ground_vertices: Buffer<B>,
    pub ground_indices: Buffer<B>,
    pub sky_vertices: Buffer<B>,
    pub sky_indices: Buffer<B>,
    pub instances: Buffer<B>,
    pub arguments: Buffer<B>,
    pub(crate) plant_data: Vec<u8>,
    pub(crate) ground_data: Vec<u8>,
    pub(crate) sky_data: Vec<u8>,
    pub(crate) plant_indices_data: Vec<u8>,
    pub(crate) ground_indices_data: Vec<u8>,
    pub(crate) sky_indices_data: Vec<u8>,
    pub(crate) instance_data: Vec<u8>,
    pub(crate) argument_data: Vec<u8>,
}
impl<B: ResourceBackend> IndirectScene<B> {
    pub fn new(device: &Device<B>) -> Result<Self> {
        let plant = gltf::load_embedded_model(PLANTS, gltf::LoadOptions::CPP_PORT)
            .map_err(|x| Error::Backend(x.to_string()))?;
        let ground = gltf::load_embedded_model(GROUND, gltf::LoadOptions::CPP_PORT)
            .map_err(|x| Error::Backend(x.to_string()))?;
        let sky = gltf::load_embedded_model(SKY, gltf::LoadOptions::CPP_PORT)
            .map_err(|x| Error::Backend(x.to_string()))?;
        let instances = instance_bytes(plant.primitives.len() as u32, 5);
        let arguments = indirect_bytes(&plant.primitives);
        let buffer = |label: &str, size: usize, usage: BufferUsage| {
            device.create_buffer(&BufferDesc {
                label: Some(label.into()),
                size: size as u64,
                usage,
                memory: MemoryClass::Device,
            })
        };
        Ok(Self {
            plant_vertices: buffer(
                "10 plant vertices",
                plant.vertices.len(),
                BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            )?,
            plant_indices: buffer(
                "10 plant indices",
                plant.indices.len(),
                BufferUsage::INDEX.union(BufferUsage::COPY_DST),
            )?,
            ground_vertices: buffer(
                "10 ground vertices",
                ground.vertices.len(),
                BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            )?,
            ground_indices: buffer(
                "10 ground indices",
                ground.indices.len(),
                BufferUsage::INDEX.union(BufferUsage::COPY_DST),
            )?,
            sky_vertices: buffer(
                "10 sky vertices",
                sky.vertices.len(),
                BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            )?,
            sky_indices: buffer(
                "10 sky indices",
                sky.indices.len(),
                BufferUsage::INDEX.union(BufferUsage::COPY_DST),
            )?,
            instances: buffer(
                "10 plant instances",
                instances.len(),
                BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            )?,
            arguments: buffer(
                "10 indexed indirect commands",
                arguments.len(),
                BufferUsage::INDIRECT.union(BufferUsage::COPY_DST),
            )?,
            plant_data: plant.vertices,
            ground_data: ground.vertices,
            sky_data: sky.vertices,
            plant_indices_data: plant.indices,
            ground_indices_data: ground.indices,
            sky_indices_data: sky.indices,
            instance_data: instances,
            argument_data: arguments,
        })
    }
    pub fn upload(&self, encoder: &mut CommandEncoder<B>) -> Result<()>
    where
        B: CommandBackend,
    {
        for (buffer, bytes) in [
            (&self.plant_vertices, &self.plant_data),
            (&self.plant_indices, &self.plant_indices_data),
            (&self.ground_vertices, &self.ground_data),
            (&self.ground_indices, &self.ground_indices_data),
            (&self.sky_vertices, &self.sky_data),
            (&self.sky_indices, &self.sky_indices_data),
            (&self.instances, &self.instance_data),
            (&self.arguments, &self.argument_data),
        ] {
            encoder.write_buffer(buffer, 0, bytes)?;
        }
        Ok(())
    }
}
pub(crate) fn instance_bytes(draws: u32, layers: u32) -> Vec<u8> {
    let count = draws.saturating_mul(PLANT_INSTANCES);
    let mut out = Vec::with_capacity(count as usize * 32);
    for n in 0..count {
        let a = n as f32 * 2.399963;
        let r = 25.0 * (((n % PLANT_INSTANCES) as f32 / PLANT_INSTANCES as f32).sqrt());
        for x in [r * a.cos(), 0.0, r * a.sin(), 0.0, a, 0.0, 0.7] {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out.extend_from_slice(&(n % layers.max(1)).to_le_bytes());
    }
    out
}
fn indirect_bytes(primitives: &[gltf::DrawPrimitive]) -> Vec<u8> {
    let mut out = Vec::with_capacity(primitives.len() * INDEXED_INDIRECT_STRIDE as usize);
    for (primitive_index, primitive) in primitives.iter().enumerate() {
        // D3D12_DRAW_INDEXED_ARGUMENTS: IndexCountPerInstance,
        // InstanceCount, StartIndexLocation, BaseVertexLocation, StartInstanceLocation.
        for x in [
            primitive.index_count,
            PLANT_INSTANCES,
            primitive.first_index,
            primitive_index as u32 * PLANT_INSTANCES,
            0,
        ] {
            out.extend_from_slice(&x.to_le_bytes())
        }
    }
    out
}

/// Encodes the plant portion of the original indirect-draw workload.
///
/// `draw_count` is the number of `DrawIndexedIndirect` records in `arguments`.
/// The caller must have transitioned all supplied resources to the uses implied
/// by the pass; the RHI deliberately does not infer those graph decisions.
pub fn encode_plants<B>(
    device: &Device<B>,
    encoder: &mut CommandEncoder<B>,
    pass: &RasterPassDesc<'_, B>,
    pipeline: &RasterPipeline<B>,
    group: &BindGroup<B>,
    vertices: &Buffer<B>,
    vertex_bytes: u64,
    indices: &Buffer<B>,
    index_bytes: u64,
    instances: &Buffer<B>,
    instance_bytes: u64,
    arguments: &Buffer<B>,
    draw_count: u32,
) -> Result<()>
where
    B: RasterBackend + DeferredBackend,
{
    if !device.capabilities().indirect_draw {
        return Err(Error::Unsupported("indexed indirect draw"));
    }
    let extras = device
        .raster_extras()
        .ok_or(Error::Unsupported("indexed indirect draw"))?;
    let mut raster = encoder.begin_raster(pass)?;
    raster.set_pipeline(pipeline)?;
    raster.set_bind_group(0, group, &[])?;
    raster.set_vertex_buffer(0, vertices, 0..vertex_bytes)?;
    raster.set_vertex_buffer(1, instances, 0..instance_bytes)?;
    raster.set_index_buffer(indices, 0..index_bytes, IndexFormat::Uint32)?;

    if device.capabilities().multi_draw {
        extras.multi_draw(
            &mut raster,
            arguments,
            0,
            draw_count,
            INDEXED_INDIRECT_STRIDE,
            true,
        )?;
    } else {
        for draw in 0..draw_count {
            extras.draw_indirect(
                &mut raster,
                arguments,
                u64::from(draw) * u64::from(INDEXED_INDIRECT_STRIDE),
                true,
            )?;
        }
    }
    raster.finish()
}
