//! Procedural particle bounds and per-frame TLAS instances.
use super::particles;
use fluxel_rhi::api::*;

pub fn instances<B: Backend>(
    scene: &AccelerationStructure<B>,
    particle: &AccelerationStructure<B>,
    quads: &AccelerationStructure<B>,
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
    instances.push(AccelerationInstance {
        blas: quads.clone(),
        transform: [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]],
        mask: 2,
        custom_index: 0,
        shader_table_offset: 0,
        flags: AccelerationInstanceFlags::NONE,
    });
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
    while instances.len() < particles::MAX_PARTICLES + 2 {
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

/// Expanded non-indexed triangles; unused slots are degenerate. Their order
/// matches the compact particle information buffer and primitiveIndex / 2.
pub fn quad_vertices(sim: &particles::Simulation, beam: bool) -> Vec<f32> {
    let mut out = Vec::with_capacity(particles::MAX_PARTICLES * 18);
    let camera_right = [0.70710677, 0.0, -0.70710677];
    let camera_up = if beam {
        [0., 1., 0.]
    } else {
        [-0.24184476, 0.9396926, -0.24184476]
    };
    for p in sim.particles.iter().filter(|p| p.active) {
        let (sin, cos) = if beam { (0., 1.) } else { p.rotation.sin_cos() };
        let right: [f32; 3] =
            std::array::from_fn(|i| (cos * camera_right[i] + sin * camera_up[i]) * p.radius);
        let up: [f32; 3] =
            std::array::from_fn(|i| (-sin * camera_right[i] + cos * camera_up[i]) * p.radius);
        for (x, y) in [
            (-1., 1.),
            (1., 1.),
            (1., -1.),
            (-1., 1.),
            (1., -1.),
            (-1., -1.),
        ] {
            out.extend((0..3).map(|i| p.position[i] + x * right[i] + y * up[i]));
        }
    }
    out.resize(particles::MAX_PARTICLES * 18, 0.);
    out
}
