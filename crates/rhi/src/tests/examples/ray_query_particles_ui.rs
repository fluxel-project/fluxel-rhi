use super::*;

#[test]
fn interactions_and_pixels() {
    let mut panel = Panel::default();
    assert_eq!(panel.flags() & 64, 0);
    panel.toggle_visible();
    assert!(!panel.pointer_down(0.0, 0.0));
    assert!(panel.pointer_down(30.0, 70.0));
    assert!(!panel.settings.animate);
    panel.pointer_down(50.0, 160.0);
    assert_eq!(panel.settings.orientation, Orientation::Beam);
    panel.pointer_down(45.0, 350.0);
    panel.pointer_move(162.1875, 350.0);
    panel.pointer_up();
    assert!((panel.settings.emitter[0] + 1.0).abs() < 0.0001);
    assert_eq!(panel.pixels().len(), (WIDTH * HEIGHT * 4) as usize);
    assert_eq!(panel.flags() & 12, 8);
    panel.settings.reorient_primary = true;
    panel.settings.reorient_secondary = false;
    panel.settings.mlab_fragments = 8;
    panel.settings.texture = ParticleTexture::Logo;
    assert_eq!(panel.flags(), 64 | 128 | 1 | 2 | 8 | 48);
}
