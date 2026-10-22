# DX12 native call performance screening

This is an engineering screening, not a frozen release benchmark. It compares
fixed release executables on Windows, AMD Radeon 780M, `x86_64-pc-windows-msvc`,
Rust 1.87.0, with the D3D12 debug layer disabled. Each of six runs used one
process/device, three warmups and nine samples per workload. The table shows
the median of three run medians; each run's range and native call counts are in
[the raw logs](performance/) and [the summary](performance/ab-summary.json).
GPU completion waits are outside the measured record and submit intervals.
Each iteration checks the completion frontier and GPU readback; uploads check all
128,000 bytes. A separate debug-layer smoke passed before timing.

The workloads are diagnostic synthetic workloads, not a prediction of an
application's frame time. Recording includes encoder, work and submission-plan
construction; shader/pipeline setup for the binding case is outside the timed
region. The binding case dispatches once after 500 repeated bindings.

| Workload | Metric | Baseline median | Candidate median |
| --- | --- | ---: | ---: |
| 500 same-lane steps | submit CPU | 1.2045 ms | 0.6872 ms |
| One step, 500 native lists | submit CPU | 0.5655 ms | 0.4698 ms |
| 500 writes of 256 bytes | record CPU | 181.9065 ms | 1.0400 ms |
| 500 same-heap bindings | record CPU | 0.7412 ms | 0.7075 ms |

The repeated-binding latency difference is within observed run variation; the
native descriptor-heap set count fell from 500 to 1. The 500-step signal count
fell from 501 to 1, and the 500-list execute-call count fell from 500 to 1.
For the upload case the warmed candidate reports no newly created or mapped
upload resources in the measured sample, versus 500 of each in the baseline;
this is a warm reuse result, not a cold-allocation result. All nine samples in
each recorded run report the same counts for their case. Candidate counter
arrays have an additional final `upload_capacity_bytes` field; the first seven
fields match the baseline schema.

The baseline executable SHA-256 is
`54c2d7d2454d161c62a2011322de6fa19a78a85f0c59caf648f072d2506289b3`;
the candidate executable SHA-256 is
`3f216e38ac2ff5c323ae7d3a2941bfd586ceaa07a3ac671ba1a51893b3b60ace`.
The baseline was built from working-tree content on HEAD
`66e77710034dbcb1c8afd458106715f6c67f57e8`; its recorded source manifest hash
is `86c04fbaf630b9e1fd594d0f3f6dbb6926918ab6bb7ce366c142df530d386a4a`.
The manifest identifies file hashes but the complete pre-optimization source
snapshot or patch was not retained, so the baseline source cannot be fully
reconstructed from this evidence. The original executables are retained in the
local performance directory for adjacent A/B runs; they are not committed.
The candidate binary is a screening build and is not identified as the final
candidate commit.
