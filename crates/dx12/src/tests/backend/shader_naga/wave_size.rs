use super::*;

#[test]
fn inserts_only_before_the_exact_compute_entry() {
    let source = "void helper() {}\nvoid main_() {}\nvoid main() {}\n";
    assert_eq!(
        inject(source, "main", 32).unwrap(),
        "void helper() {}\nvoid main_() {}\n[WaveSize(32)]\nvoid main() {}\n"
    );
}

#[test]
fn rejects_missing_or_invalid_entry() {
    assert!(inject("void other() {}\n", "main", 32).is_err());
    assert!(inject("void main() {}\n", "main", 3).is_err());
}

#[test]
fn direct_hlsl_finds_definition_not_comment_string_or_prototype() {
    let source = "// void main() {}\nconst char* note = \"void main() {}\";\nvoid main();\n[numthreads(1, 1, 1)] void main() {}";
    let result = inject_direct_hlsl(source, "main", 32).unwrap();
    assert!(result.contains("[numthreads(1, 1, 1)] [WaveSize(32)] void main()"));
    assert_eq!(result.matches("WaveSize").count(), 1);
}

#[test]
fn direct_hlsl_handles_multiline_and_rejects_duplicate_or_preprocessor() {
    let multiline = "[numthreads(1, 1, 1)]\nvoid\nmain()\n{}";
    assert!(
        inject_direct_hlsl(multiline, "main", 32)
            .unwrap()
            .contains("[WaveSize(32)] void\nmain")
    );
    assert!(
        inject_direct_hlsl(
            "[WaveSize(16)] [numthreads(1,1,1)] void main() {}",
            "main",
            32
        )
        .is_err()
    );
    assert!(inject_direct_hlsl("#if 0\nvoid main() {}\n#endif", "main", 32).is_err());
}

#[test]
fn direct_hlsl_ignores_method_decoy_and_rejects_multiple_global_entries() {
    let source = "struct Decoy { void main() {} };\nvoid main() {}";
    let result = inject_direct_hlsl(source, "main", 32).unwrap();
    assert!(result.contains("struct Decoy { void main() {} };\n[WaveSize(32)] void main()"));
    assert!(inject_direct_hlsl("void main() {}\nvoid main() {}", "main", 32).is_err());
}
