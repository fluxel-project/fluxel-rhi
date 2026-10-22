# Example 19 dropdown controls

Windows MSVC, AMD Radeon 780M, driver 32.0.21028.2002, 1000 x 709. Source and executable hashes are recorded in sha256.json. DX12 debug and Vulkan validation were enabled. The input probe paused the simulation, opened the orientation, texture and fragment lists and selected beam, logo and eight fragments. Both processes remained alive until the probe terminated them; their logs contain no diagnostics. The DX12 final panel image was opened for inspection.

Two standalone UI tests passed, including direct list selection and editing an emitter value. The probe attempted to use posted WM_CHAR messages for numerical input; the retained final screenshot still shows the old value, because this did not produce the host text event. Numeric input requires a real keyboard or suitable OS input injection for GPU validation.

This remains a CPU bitmap UI, not Dear ImGui: font, styling, layout and some input semantics differ. The latest baseline appearance comparison against the corrected reference is in ../demo19-camera-controls/report.md. This work does not establish pixel-identical or behavior-identical UI.
