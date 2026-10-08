#![cfg(target_os = "android")]

#[path = "../../01_triangle.rs"]
pub mod triangle;

pub use triangle::common;

pub fn create_example() -> triangle::TriangleExample {
    triangle::create_example()
}
