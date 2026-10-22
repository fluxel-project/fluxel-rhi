//! Particle emission and integration matching NVIDIA's rt_particles sample.
//!
//! The source sample advances with actual frame deltas, emits at most one
//! particle per update, and uses the Windows C runtime's default rand sequence.
// Copyright (c) 2014-2022 NVIDIA CORPORATION. Licensed under the MIT license;
// see ../assets/19_ray_query_particles/LICENSE-NVIDIA.txt.

pub const MAX_PARTICLES: usize = 1024;
#[derive(Clone, Copy, Default)]
pub struct Particle {
    pub active: bool,
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub color: [f32; 3],
    pub radius: f32,
    pub age: f32,
    pub opacity: f32,
    pub rotation: f32,
}

pub struct Simulation {
    pub particles: Vec<Particle>,
    elapsed: f32,
    last_emit: f32,
    random: u32,
    pub emitter: [f32; 3],
}
impl Simulation {
    pub fn new() -> Self {
        Self {
            particles: vec![Particle::default(); MAX_PARTICLES],
            elapsed: 0.,
            last_emit: 0.,
            random: 1,
            emitter: [-2., 0., -2.],
        }
    }
    fn random_float(&mut self) -> f32 {
        self.random = self.random.wrapping_mul(214013).wrapping_add(2531011);
        ((self.random >> 16) & 0x7fff) as f32 / 32767.
    }
    pub fn advance(&mut self, delta: f32) {
        self.elapsed += delta;
        let mut first_empty = None;
        for (index, particle) in self.particles.iter_mut().enumerate() {
            if particle.active {
                for i in 0..3 {
                    particle.position[i] += particle.velocity[i] * delta;
                }
                particle.velocity[0] += delta;
                particle.velocity[1] += delta;
                particle.age += delta;
                particle.radius += 0.5 * delta;
                particle.opacity = ((2. - particle.age) * 0.5).clamp(0., 1.);
                if particle.age > 2. {
                    particle.active = false;
                }
            }
            if !particle.active && first_empty.is_none() {
                first_empty = Some(index);
            }
        }
        if self.elapsed - self.last_emit > 1. / 20. {
            if let Some(index) = first_empty {
                // MSVC evaluates the source float3 constructor arguments from
                // right to left; preserve that order for the reference build.
                let vz = self.random_float() - 0.5;
                let vy = self.random_float() + 0.5;
                let vx = self.random_float() - 0.5;
                let velocity = [vx, vy, vz];
                let radius = self.random_float() * 0.05 + 0.1;
                let b = self.random_float() * 0.5 + 0.1;
                let g = self.random_float() * 0.5 + 0.1;
                let r = self.random_float() * 0.5 + 0.1;
                let color = [r, g, b];
                let rotation = self.random_float() * 6.28;
                self.particles[index] = Particle {
                    active: true,
                    position: self.emitter,
                    velocity,
                    color,
                    radius,
                    age: 0.,
                    opacity: 1.,
                    rotation,
                };
                self.last_emit = self.elapsed;
            }
        }
    }
}
