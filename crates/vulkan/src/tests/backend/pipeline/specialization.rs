//! Unit tests for the corresponding implementation module.

use super::*;

#[test]
fn resolves_grouped_spec_id_without_exposing_group_as_constant() {
    let words = [
        0x0723_0203,
        0x0001_0000,
        0,
        10,
        0,
        (2 << 16) | 73,
        3,
        (4 << 16) | 71,
        3,
        1,
        42,
        (3 << 16) | 74,
        3,
        7,
        (4 << 16) | 71,
        8,
        1,
        9,
    ];
    let ids = specialization_ids(&words).unwrap();
    assert_eq!(ids.get(&7), Some(&42));
    assert_eq!(ids.get(&8), Some(&9));
    assert!(!ids.contains_key(&3));
}

#[test]
fn native_info_is_absent_without_overrides_and_present_with_an_override() {
    let empty = Specialization {
        entries: Vec::new(),
        data: Vec::new(),
    };
    assert!(empty.native_info().is_none());

    let populated = Specialization {
        entries: vec![vk::SpecializationMapEntry {
            constant_id: 7,
            offset: 0,
            size: 4,
        }],
        data: vec![1, 0, 0, 0],
    };
    let info = populated.native_info().expect("override keeps native info");
    assert_eq!(info.map_entry_count, 1);
    assert_eq!(info.data_size, 4);
}
