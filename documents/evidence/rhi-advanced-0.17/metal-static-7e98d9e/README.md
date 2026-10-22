# Metal static verification for 7e98d9e

Source commit: `7e98d9ebbf03685d76cb412aa625af1ed5a2257b`.
Host: Windows MSVC. No Apple GPU was available or exercised.

| Command | Result | Log SHA-256 |
| --- | --- | --- |
| `cargo test -p fluxel-rhi-core -p fluxel-rhi-metal --lib` | 52 Core and 57 Metal tests passed | `core-metal-tests.log`: `0093b2310dc3d4e3a07c1880d7b9ec99224b27d89224d952ae5f3180745c11cb` |
| `cargo +1.87 check -p fluxel-rhi-metal --target aarch64-apple-darwin --lib` | Passed | `apple-arm64-msrv.log`: `a905345a90ddc157b63ecbee3cde890835d78819c7377bd2c2edc186eb82ade1` |
| `cargo +1.87 check -p fluxel-rhi-metal --target x86_64-apple-darwin --lib` | Passed | `apple-x64-msrv.log`: `2d52eff659ee930d5a1aef074714194f984ef78a17ce75ff602a15a4814ad265` |

The Metal statistics and ICB paths were also reviewed independently for
capability gating, query state, resource retention, and ordered completion;
neither review found a definite P0/P1 in the static path. Compilation and
review do not prove native execution, pass statistics values, synchronization,
or draw pixels. The fixed eleven-field query, ray pipeline/shader table, and
sparse mapping are not advertised on Metal.
