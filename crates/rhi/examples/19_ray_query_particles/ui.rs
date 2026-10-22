//! CPU-rendered settings panel for the interactive particle example.
//!
//! The panel owns interaction state only; the caller applies `settings` to the
//! simulation and GPU resources, and composites `pixels` over the rendered image.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    AccumulatedVector,
    Quaternion,
    Beam,
    Basis,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticleTexture {
    Smoke,
    Logo,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    pub animate: bool,
    pub update_orientation_when_paused: bool,
    pub orientation: Orientation,
    pub reorient_primary: bool,
    pub reorient_secondary: bool,
    pub mlab_fragments: u32,
    pub emitter: [f32; 3],
    pub texture: ParticleTexture,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            animate: true,
            update_orientation_when_paused: true,
            orientation: Orientation::Quaternion,
            reorient_primary: false,
            reorient_secondary: true,
            mlab_fragments: 4,
            emitter: [-2.0, 0.0, -2.0],
            texture: ParticleTexture::Smoke,
        }
    }
}

pub const WIDTH: u32 = 320;
pub const HEIGHT: u32 = 310;
pub const X: u32 = 15;
pub const Y: u32 = 15;
pub const SCREEN_WIDTH: u32 = 375;
pub const SCREEN_HEIGHT: u32 = 450;

/// Pointer coordinates are window pixels. Returns true when the panel consumes
/// the event, so callers can suppress camera controls while interacting.
pub struct Panel {
    pub settings: Settings,
    drag: Option<(usize, f32, f32)>,
    visible: bool,
}
impl Default for Panel {
    fn default() -> Self {
        Self {
            settings: Settings::default(),
            drag: None,
            visible: false,
        }
    }
}
impl Panel {
    pub fn pointer_down(&mut self, x: f32, y: f32) -> bool {
        if !self.visible {
            return false;
        }
        let x = (x - X as f32) * WIDTH as f32 / SCREEN_WIDTH as f32;
        let y = (y - Y as f32) * HEIGHT as f32 / SCREEN_HEIGHT as f32;
        if x < 0.0 || y < 0.0 || x >= WIDTH as f32 || y >= HEIGHT as f32 {
            return false;
        }
        match y as u32 {
            28..=45 => self.settings.animate = !self.settings.animate,
            46..=63 => {
                self.settings.update_orientation_when_paused =
                    !self.settings.update_orientation_when_paused
            }
            88..=106 => {
                self.settings.orientation = match self.settings.orientation {
                    Orientation::AccumulatedVector => Orientation::Quaternion,
                    Orientation::Quaternion => Orientation::Beam,
                    Orientation::Beam => Orientation::Basis,
                    Orientation::Basis => Orientation::AccumulatedVector,
                }
            }
            134..=151 => self.settings.reorient_primary = !self.settings.reorient_primary,
            152..=169 => self.settings.reorient_secondary = !self.settings.reorient_secondary,
            180..=199 => {
                self.settings.mlab_fragments = match self.settings.mlab_fragments {
                    1 => 2,
                    2 => 4,
                    4 => 8,
                    _ => 1,
                }
            }
            222..=241 if (10.0..310.0).contains(&x) => {
                let axis = ((x - 10.0) / 100.0) as usize;
                self.drag = Some((axis, x, self.settings.emitter[axis]));
            }
            264..=283 => {
                self.settings.texture = match self.settings.texture {
                    ParticleTexture::Smoke => ParticleTexture::Logo,
                    ParticleTexture::Logo => ParticleTexture::Smoke,
                }
            }
            _ => {}
        }
        true
    }
    pub fn pointer_move(&mut self, x: f32, _y: f32) -> bool {
        if let Some((axis, start, value)) = self.drag {
            self.settings.emitter[axis] =
                value + ((x - X as f32) * WIDTH as f32 / SCREEN_WIDTH as f32 - start) * 0.01;
            true
        } else {
            false
        }
    }
    pub fn pointer_up(&mut self) {
        self.drag = None;
    }
    pub fn toggle_animation(&mut self) {
        self.settings.animate = !self.settings.animate;
    }
    pub fn toggle_visible(&mut self) {
        self.visible = !self.visible;
        self.drag = None;
    }
    pub fn visible(&self) -> bool {
        self.visible
    }

    pub fn flags(&self) -> u32 {
        let mut f = if self.visible { 64 } else { 0 };
        if self.settings.reorient_primary {
            f |= 1;
        }
        if !self.settings.reorient_secondary {
            f |= 2;
        }
        f |= match self.settings.orientation {
            Orientation::Quaternion => 0,
            Orientation::Basis => 1,
            Orientation::Beam => 2,
            Orientation::AccumulatedVector => 3,
        } << 2;
        f |= match self.settings.mlab_fragments {
            1 => 1,
            2 => 2,
            8 => 3,
            _ => 0,
        } << 4;
        if self.settings.texture == ParticleTexture::Logo {
            f |= 128;
        }
        f
    }

    /// Straight-alpha RGBA8 pixels, tightly packed, sized WIDTH by HEIGHT.
    pub fn pixels(&self) -> Vec<u8> {
        let mut canvas = Canvas(vec![0; (WIDTH * HEIGHT * 4) as usize]);
        canvas.rect(0, 0, WIDTH, HEIGHT, [15, 15, 15, 235]);
        canvas.rect(0, 0, WIDTH, 20, [41, 74, 122, 255]);
        canvas.text(8, 6, "Settings", [255, 255, 255, 255]);
        canvas.check(10, 30, self.settings.animate, "Animate particles (Space)");
        canvas.check(
            10,
            48,
            self.settings.update_orientation_when_paused,
            "Update orientation when paused",
        );
        canvas.line(67);
        canvas.text(10, 75, "Orientation mode:", [240, 240, 240, 255]);
        let orientation = match self.settings.orientation {
            Orientation::AccumulatedVector => "Accumulated Vector Transform",
            Orientation::Quaternion => "Quaternion Rotation",
            Orientation::Beam => "Beam or Vertical Sprite",
            Orientation::Basis => "Basis (RTG2)",
        };
        canvas.choice(20, 88, 286, orientation);
        canvas.line(111);
        canvas.text(10, 119, "Reorient particles:", [240, 240, 240, 255]);
        canvas.check(20, 136, self.settings.reorient_primary, "In primary rays");
        canvas.check(
            20,
            154,
            self.settings.reorient_secondary,
            "In secondary rays",
        );
        canvas.line(175);
        canvas.choice(10, 180, 42, &self.settings.mlab_fragments.to_string());
        canvas.text(60, 186, "Blending fragments", [240, 240, 240, 255]);
        canvas.line(203);
        canvas.text(10, 210, "Emitter position:", [240, 240, 240, 255]);
        for i in 0..3 {
            canvas.choice(
                10 + i as u32 * 100,
                222,
                95,
                &format!("{} {:.2}", ["X", "Y", "Z"][i], self.settings.emitter[i]),
            );
        }
        canvas.text(10, 249, "Particle texture:", [240, 240, 240, 255]);
        canvas.choice(
            20,
            264,
            140,
            match self.settings.texture {
                ParticleTexture::Smoke => "Smoke",
                ParticleTexture::Logo => "Logo",
            },
        );
        canvas.0
    }
}
struct Canvas(Vec<u8>);
impl Canvas {
    fn rect(&mut self, x: u32, y: u32, w: u32, h: u32, c: [u8; 4]) {
        for yy in y..(y + h).min(HEIGHT) {
            for xx in x..(x + w).min(WIDTH) {
                let p = ((yy * WIDTH + xx) * 4) as usize;
                self.0[p..p + 4].copy_from_slice(&c);
            }
        }
    }
    fn line(&mut self, y: u32) {
        self.rect(8, y, WIDTH - 16, 1, [90, 90, 90, 255]);
    }
    fn check(&mut self, x: u32, y: u32, checked: bool, label: &str) {
        self.rect(x, y, 13, 13, [64, 64, 64, 255]);
        if checked {
            self.rect(x + 3, y + 3, 7, 7, [90, 160, 235, 255]);
        }
        self.text(x + 19, y + 3, label, [240, 240, 240, 255]);
    }
    fn choice(&mut self, x: u32, y: u32, w: u32, label: &str) {
        self.rect(x, y, w, 18, [64, 64, 64, 255]);
        self.text(x + 5, y + 5, label, [240, 240, 240, 255]);
        self.text(x + w - 10, y + 5, ">", [200, 200, 200, 255]);
    }
    fn text(&mut self, mut x: u32, y: u32, s: &str, color: [u8; 4]) {
        for ch in s.chars() {
            let rows = glyph(ch);
            for yy in 0..10u32 {
                let bits = rows[(yy * 7 / 10) as usize];
                for xx in 0..7u32 {
                    let col = xx * 5 / 7;
                    if bits & (1 << (4 - col)) != 0 {
                        self.rect(x + xx, y + yy, 1, 1, color);
                    }
                }
            }
            x += 8;
        }
    }
}
fn glyph(c: char) -> [u8; 7] {
    match c.to_ascii_uppercase() {
        'A' => [14, 17, 17, 31, 17, 17, 17],
        'B' => [30, 17, 17, 30, 17, 17, 30],
        'C' => [14, 17, 16, 16, 16, 17, 14],
        'D' => [30, 17, 17, 17, 17, 17, 30],
        'E' => [31, 16, 16, 30, 16, 16, 31],
        'F' => [31, 16, 16, 30, 16, 16, 16],
        'G' => [14, 17, 16, 23, 17, 17, 15],
        'H' => [17, 17, 17, 31, 17, 17, 17],
        'I' => [14, 4, 4, 4, 4, 4, 14],
        'J' => [7, 2, 2, 2, 18, 18, 12],
        'K' => [17, 18, 20, 24, 20, 18, 17],
        'L' => [16, 16, 16, 16, 16, 16, 31],
        'M' => [17, 27, 21, 21, 17, 17, 17],
        'N' => [17, 25, 21, 19, 17, 17, 17],
        'O' => [14, 17, 17, 17, 17, 17, 14],
        'P' => [30, 17, 17, 30, 16, 16, 16],
        'Q' => [14, 17, 17, 17, 21, 18, 13],
        'R' => [30, 17, 17, 30, 20, 18, 17],
        'S' => [15, 16, 16, 14, 1, 1, 30],
        'T' => [31, 4, 4, 4, 4, 4, 4],
        'U' => [17, 17, 17, 17, 17, 17, 14],
        'V' => [17, 17, 17, 17, 17, 10, 4],
        'W' => [17, 17, 17, 21, 21, 21, 10],
        'X' => [17, 17, 10, 4, 10, 17, 17],
        'Y' => [17, 17, 10, 4, 4, 4, 4],
        'Z' => [31, 1, 2, 4, 8, 16, 31],
        '0' => [14, 17, 19, 21, 25, 17, 14],
        '1' => [4, 12, 4, 4, 4, 4, 14],
        '2' => [14, 17, 1, 2, 4, 8, 31],
        '3' => [30, 1, 1, 14, 1, 1, 30],
        '4' => [2, 6, 10, 18, 31, 2, 2],
        '5' => [31, 16, 16, 30, 1, 1, 30],
        '6' => [14, 16, 16, 30, 17, 17, 14],
        '7' => [31, 1, 2, 4, 8, 8, 8],
        '8' => [14, 17, 17, 14, 17, 17, 14],
        '9' => [14, 17, 17, 15, 1, 1, 14],
        ':' => [0, 4, 4, 0, 4, 4, 0],
        '.' => [0, 0, 0, 0, 0, 4, 4],
        '-' => [0, 0, 0, 31, 0, 0, 0],
        '(' => [2, 4, 8, 8, 8, 4, 2],
        ')' => [8, 4, 2, 2, 2, 4, 8],
        '>' => [16, 8, 4, 2, 4, 8, 16],
        _ => [0; 7],
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interactions_and_pixels() {
        let mut p = Panel::default();
        assert_eq!(p.flags() & 64, 0);
        p.toggle_visible();
        assert!(!p.pointer_down(0.0, 0.0));
        assert!(p.pointer_down(30.0, 70.0));
        assert!(!p.settings.animate);
        p.pointer_down(50.0, 160.0);
        assert_eq!(p.settings.orientation, Orientation::Beam);
        p.pointer_down(45.0, 350.0);
        p.pointer_move(162.1875, 350.0);
        p.pointer_up();
        assert!((p.settings.emitter[0] + 1.0).abs() < 0.0001);
        assert_eq!(p.pixels().len(), (WIDTH * HEIGHT * 4) as usize);
        assert_eq!(p.flags() & 12, 8); // beam
        p.settings.reorient_primary = true;
        p.settings.reorient_secondary = false;
        p.settings.mlab_fragments = 8;
        p.settings.texture = ParticleTexture::Logo;
        assert_eq!(p.flags(), 64 | 128 | 1 | 2 | 8 | 48);
    }
}
