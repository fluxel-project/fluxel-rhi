//! Unit tests for the selected cooperative-matrix memory-layout admission.

use super::*;

fn instruction(opcode: u32, operands: &[u32]) -> Vec<u32> {
    let count = u32::try_from(operands.len() + 1).unwrap();
    let mut words = vec![(count << 16) | opcode];
    words.extend_from_slice(operands);
    words
}

fn module(
    layout: u32,
    stride: Option<u32>,
    element_index: u32,
    element_index_high_word: Option<u32>,
) -> Vec<u32> {
    module_with_block(layout, stride, element_index, element_index_high_word, true)
}

fn module_with_block(
    layout: u32,
    stride: Option<u32>,
    element_index: u32,
    element_index_high_word: Option<u32>,
    has_block_decoration: bool,
) -> Vec<u32> {
    let mut words = vec![0x0723_0203, 0x0001_0000, 0, 16, 0];
    // %4 is the cooperative-matrix layout, %5 the explicit stride, and
    // %7/%8 are the zero/nonzero indexes in the admitted access-chain form.
    words.extend(instruction(21, &[1, 32, 0]));
    words.extend(instruction(43, &[1, 4, layout]));
    words.extend(instruction(43, &[1, 5, stride.unwrap_or(16)]));
    words.extend(instruction(43, &[1, 7, 0]));
    if let Some(high_word) = element_index_high_word {
        words.extend(instruction(21, &[14, 64, 0]));
        words.extend(instruction(43, &[14, 8, element_index, high_word]));
    } else {
        words.extend(instruction(43, &[1, 8, element_index]));
    }
    // A StorageBuffer variable backed by a one-member struct whose member is
    // a runtime array at byte offset zero, matching the GPU artifact.
    words.extend(instruction(29, &[9, 10]));
    words.extend(instruction(30, &[11, 9]));
    words.extend(instruction(32, &[12, 12, 11]));
    words.extend(instruction(32, &[13, 12, 10]));
    words.extend(instruction(59, &[12, 3, 12]));
    if has_block_decoration {
        words.extend(instruction(71, &[11, 2]));
    }
    words.extend(instruction(72, &[11, 0, 35, 0]));
    words.extend(instruction(65, &[13, 6, 3, 7, 8]));
    let mut load = vec![1, 2, 6, 4];
    let mut store = vec![6, 2, 4];
    if stride.is_some() {
        load.push(5);
        store.push(5);
    }
    words.extend(instruction(4457, &load));
    words.extend(instruction(4458, &store));
    words
}

#[test]
fn rejects_matching_struct_without_block_decoration() {
    assert!(matches!(
        validate_cooperative_matrix_layout(&module_with_block(0, Some(16), 0, None, false)),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn admits_row_major_explicit_stride_sixteen() {
    assert!(validate_cooperative_matrix_layout(&module(0, Some(16), 0, None)).is_ok());
}

#[test]
fn rejects_column_major_layout() {
    assert!(matches!(
        validate_cooperative_matrix_layout(&module(1, Some(16), 0, None)),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn rejects_missing_explicit_stride() {
    assert!(matches!(
        validate_cooperative_matrix_layout(&module(0, None, 0, None)),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn rejects_non_selected_stride() {
    assert!(matches!(
        validate_cooperative_matrix_layout(&module(0, Some(8), 0, None)),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn rejects_malformed_spirv() {
    let mut words = module(0, Some(16), 0, None);
    words.push(4457);
    assert!(matches!(
        validate_cooperative_matrix_layout(&words),
        Err(Error::Shader(_))
    ));
}

#[test]
fn rejects_mixed_valid_and_short_cooperative_instructions() {
    let mut short_load = module(0, Some(16), 0, None);
    short_load.extend(instruction(4457, &[1, 2, 6]));
    assert!(matches!(
        validate_cooperative_matrix_layout(&short_load),
        Err(Error::Shader(_))
    ));

    let mut short_store = module(0, Some(16), 0, None);
    short_store.extend(instruction(4458, &[6, 2]));
    assert!(matches!(
        validate_cooperative_matrix_layout(&short_store),
        Err(Error::Shader(_))
    ));
}

#[test]
fn rejects_nonzero_access_chain_offset() {
    assert!(matches!(
        validate_cooperative_matrix_layout(&module(0, Some(16), 1, None)),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn rejects_64_bit_index_with_nonzero_high_word() {
    assert!(matches!(
        validate_cooperative_matrix_layout(&module(0, Some(16), 0, Some(1))),
        Err(Error::Unsupported(_))
    ));
}
