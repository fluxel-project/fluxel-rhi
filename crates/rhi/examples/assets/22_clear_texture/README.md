# Standalone texture clear: HelloTriangle assets

The HLSL and DXIL files are the same HelloTriangle assets used by example 17.
The HLSL derives from Microsoft DirectX-Graphics-Samples revision
`e5975f9b0744fc5096dd593ed72bf8f5c004164f`; the included MIT license applies.
The RHI shader adjusts the clip-space vertical convention. See example 17's
asset README for the source and compilation details.

Example 22 clears the offscreen RGBA8 texture with `clear_texture`, then loads
it in a raster pass and draws the triangle. Set `FLUXEL_CLEAR_READBACK` to save
the first completed frame as a PPM image.
