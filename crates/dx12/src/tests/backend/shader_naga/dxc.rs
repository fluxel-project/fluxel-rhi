use super::*;

#[test]
fn definitions_are_finite_and_injection_free() {
    assert_eq!(
        definitions(&[("VALUE".into(), SpecializationValue::U32(7))]).unwrap(),
        ["VALUE=(7u)"]
    );
    assert!(definitions(&[("X -T cs_6_0".into(), SpecializationValue::U32(1))]).is_err());
    assert!(definitions(&[("X".into(), SpecializationValue::F32(f32::NAN))]).is_err());
    assert_eq!(
        definitions(&[("HALF".into(), SpecializationValue::F16Bits(0x3e00))]).unwrap(),
        ["HALF=(1.5)"]
    );
    assert!(definitions(&[("HALF".into(), SpecializationValue::F16Bits(0x7c00))]).is_err());
    assert_eq!(
        definitions(&[("D".into(), SpecializationValue::F64(1.5))]).unwrap(),
        ["D=(1.5L)"]
    );
    assert_eq!(
        definitions(&[("I".into(), SpecializationValue::I64(i64::MIN))]).unwrap(),
        ["I=((-9223372036854775807ll-1ll))"]
    );
}

#[test]
fn direct_hlsl_stage_profiles_compile() {
    let source = "float4 vs(float4 position : POSITION) : SV_Position { return position; }\nfloat4 ps() : SV_Target { return float4(1, 0, 0, 1); }\n[numthreads(1, 1, 1)] void cs() {}";
    for (entry, profile) in [("vs", "vs_6_0"), ("ps", "ps_6_0"), ("cs", "cs_6_0")] {
        assert!(
            compile_hlsl(source, entry, profile, &[])
                .unwrap()
                .starts_with(b"DXBC")
        );
    }
}

#[test]
fn direct_hlsl_preserves_double_and_i64_literals() {
    let source =
        "[numthreads(1,1,1)] void cs() { double value = DOUBLE_VALUE; int64_t minimum = MINIMUM; }";
    let constants = [
        (
            "DOUBLE_VALUE".into(),
            SpecializationValue::F64(16_777_217.0),
        ),
        ("MINIMUM".into(), SpecializationValue::I64(i64::MIN)),
    ];
    assert!(
        compile_hlsl(source, "cs", "cs_6_0", &constants)
            .unwrap()
            .starts_with(b"DXBC")
    );
}
