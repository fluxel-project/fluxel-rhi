//! Windows display-configuration observation for an HDR output's SDR white.

use std::mem::size_of;

use windows::Win32::{
    Devices::Display::{
        DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL, DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME,
        DISPLAYCONFIG_DEVICE_INFO_HEADER, DISPLAYCONFIG_MODE_INFO, DISPLAYCONFIG_PATH_INFO,
        DISPLAYCONFIG_SDR_WHITE_LEVEL, DISPLAYCONFIG_SOURCE_DEVICE_NAME,
        DisplayConfigGetDeviceInfo, GetDisplayConfigBufferSizes, QDC_ONLY_ACTIVE_PATHS,
        QueryDisplayConfig,
    },
    Foundation::{ERROR_INSUFFICIENT_BUFFER, ERROR_SUCCESS, LUID},
};

pub(super) fn white_level(output: &[u16]) -> Option<f32> {
    let mut level = None;
    for path in active_paths()? {
        let source = DISPLAYCONFIG_SOURCE_DEVICE_NAME {
            header: DISPLAYCONFIG_DEVICE_INFO_HEADER {
                r#type: DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME,
                size: size_of::<DISPLAYCONFIG_SOURCE_DEVICE_NAME>() as u32,
                adapterId: path.sourceInfo.adapterId,
                id: path.sourceInfo.id,
            },
            ..Default::default()
        };
        let mut source = source;
        // SAFETY: the header starts a live, writable source-name record of
        // the exact type and size specified in that header.
        if unsafe { DisplayConfigGetDeviceInfo(&mut source.header) } != ERROR_SUCCESS.0 as i32 {
            return None;
        }
        if source.viewGdiDeviceName.split(|&unit| unit == 0).next()
            != output.split(|&unit| unit == 0).next()
        {
            continue;
        }
        let value = target_white_level(path.targetInfo.adapterId, path.targetInfo.id)?;
        // A clone group can have multiple targets for one GDI source name.
        // Publish a reference only when every matching target agrees.
        if level.is_some_and(|previous| previous != value) {
            return None;
        }
        level = Some(value);
    }
    level.map(|value| value as f32 * (80.0 / 1000.0))
}

fn target_white_level(adapter: LUID, id: u32) -> Option<u32> {
    let mut value = DISPLAYCONFIG_SDR_WHITE_LEVEL {
        header: DISPLAYCONFIG_DEVICE_INFO_HEADER {
            r#type: DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL,
            size: size_of::<DISPLAYCONFIG_SDR_WHITE_LEVEL>() as u32,
            adapterId: adapter,
            id,
        },
        ..Default::default()
    };
    // SAFETY: the header starts a live, writable SDR white record of the
    // exact type and size specified in that header.
    let status = unsafe { DisplayConfigGetDeviceInfo(&mut value.header) };
    (status == ERROR_SUCCESS.0 as i32 && value.SDRWhiteLevel != 0).then_some(value.SDRWhiteLevel)
}

fn active_paths() -> Option<Vec<DISPLAYCONFIG_PATH_INFO>> {
    for _ in 0..3 {
        let mut path_count = 0;
        let mut mode_count = 0;
        // SAFETY: both outputs are initialized local counters of the required type.
        let status = unsafe {
            GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &mut path_count, &mut mode_count)
        };
        if status != ERROR_SUCCESS || path_count == 0 || mode_count == 0 {
            return None;
        }
        let mut paths = vec![DISPLAYCONFIG_PATH_INFO::default(); path_count as usize];
        let mut modes = vec![DISPLAYCONFIG_MODE_INFO::default(); mode_count as usize];
        // SAFETY: each array is writable for the element count supplied to
        // the synchronous query, which also reports their used counts.
        let status = unsafe {
            QueryDisplayConfig(
                QDC_ONLY_ACTIVE_PATHS,
                &mut path_count,
                paths.as_mut_ptr(),
                &mut mode_count,
                modes.as_mut_ptr(),
                None,
            )
        };
        if status == ERROR_INSUFFICIENT_BUFFER {
            continue;
        }
        if status != ERROR_SUCCESS {
            return None;
        }
        paths.truncate(path_count as usize);
        return Some(paths);
    }
    None
}
