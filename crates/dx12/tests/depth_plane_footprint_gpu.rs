//! Observe the native per-plane footprint used for depth/stencil transfers.
#![cfg(all(windows, feature = "hardware-tests"))]
use windows::Win32::Graphics::{Direct3D::D3D_FEATURE_LEVEL_11_0, Direct3D12::*, Dxgi::Common::*};
#[test]
fn native_plane_footprints() {
    let mut device = None::<ID3D12Device>;
    // SAFETY: the output is a valid COM interface slot.
    unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut device).unwrap() };
    let device = device.unwrap();
    for format in [
        DXGI_FORMAT_R24G8_TYPELESS,
        DXGI_FORMAT_R32G8X24_TYPELESS,
        DXGI_FORMAT_R32_TYPELESS,
        DXGI_FORMAT_R16_TYPELESS,
    ] {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: 8,
            Height: 8,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        };
        for plane in
            0..if format == DXGI_FORMAT_R24G8_TYPELESS || format == DXGI_FORMAT_R32G8X24_TYPELESS {
                2
            } else {
                1
            }
        {
            let mut layout = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut rows = 0;
            let mut row_size = 0;
            let mut total = 0;
            // SAFETY: the descriptor and output slots are valid for this call.
            unsafe {
                device.GetCopyableFootprints(
                    &desc,
                    plane,
                    1,
                    0,
                    Some(&mut layout),
                    Some(&mut rows),
                    Some(&mut row_size),
                    Some(&mut total),
                );
            }
            let expected_format = if plane == 1 {
                DXGI_FORMAT_R8_TYPELESS
            } else if format == DXGI_FORMAT_R16_TYPELESS {
                DXGI_FORMAT_R16_TYPELESS
            } else {
                DXGI_FORMAT_R32_TYPELESS
            };
            assert_eq!(layout.Footprint.Format, expected_format);
            assert_eq!(rows, 8);
            assert_eq!(
                row_size,
                if plane == 1 {
                    8
                } else if format == DXGI_FORMAT_R16_TYPELESS {
                    16
                } else {
                    32
                }
            );
            assert!(layout.Footprint.RowPitch >= row_size as u32);
            assert_eq!(layout.Footprint.RowPitch % 256, 0);
            println!(
                "resource={format:?} plane={plane} format={:?} rows={rows} row_size={row_size} pitch={} total={total}",
                layout.Footprint.Format, layout.Footprint.RowPitch
            );
        }
    }
}
