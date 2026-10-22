//! Procedural particle bounds and per-frame TLAS instances.
use super::particles;
use fluxel_rhi::api::*;

pub fn instances<B: Backend>(
    scene: &AccelerationStructure<B>,
    particle: &AccelerationStructure<B>,
    sim: &particles::Simulation,
) -> Vec<AccelerationInstance<B>> {
    let mut instances = vec![AccelerationInstance {
        blas: scene.clone(),
        transform: [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]],
        mask: 1,
        custom_index: 0,
        shader_table_offset: 0,
        flags: AccelerationInstanceFlags::NONE,
    }];
    for (index, p) in sim.particles.iter().filter(|p| p.active).enumerate() {
        let r = p.radius;
        instances.push(AccelerationInstance {
            blas: particle.clone(),
            transform: [
                [r, 0., 0., p.position[0]],
                [0., r, 0., p.position[1]],
                [0., 0., r, p.position[2]],
            ],
            mask: 4,
            custom_index: index as u32,
            shader_table_offset: 0,
            flags: AccelerationInstanceFlags::NONE,
        });
    }
    while instances.len() <= particles::MAX_PARTICLES {
        instances.push(AccelerationInstance {
            blas: particle.clone(),
            transform: [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]],
            mask: 0,
            custom_index: 0,
            shader_table_offset: 0,
            flags: AccelerationInstanceFlags::NONE,
        });
    }
    instances
}
