//! Native GL query tokens used by context discovery.
pub const VENDOR: u32 = 0x1F00;
pub const RENDERER: u32 = 0x1F01;
pub const VERSION: u32 = 0x1F02;
pub const EXTENSIONS: u32 = 0x1F03;
pub const SHADING_LANGUAGE_VERSION: u32 = 0x8B8C;
pub const NUM_EXTENSIONS: u32 = 0x821D;
pub const MAX_TEXTURE_SIZE: u32 = 0x0D33;
pub const MAX_3D_TEXTURE_SIZE: u32 = 0x8073;
pub const MAX_ARRAY_TEXTURE_LAYERS: u32 = 0x88FF;
pub const MAX_CUBE_MAP_TEXTURE_SIZE: u32 = 0x851C;
pub const MAX_RENDERBUFFER_SIZE: u32 = 0x84E8;
pub const MAX_COLOR_ATTACHMENTS: u32 = 0x8CDF;
pub const MAX_DRAW_BUFFERS: u32 = 0x8824;
pub const MAX_VERTEX_ATTRIBS: u32 = 0x8869;
pub const MAX_VIEWPORT_DIMS: u32 = 0x0D3A;
pub const MAX_VIEWPORTS: u32 = 0x825B;
pub const MAX_VERTEX_TEXTURE_IMAGE_UNITS: u32 = 0x8B4C;
pub const MAX_TEXTURE_IMAGE_UNITS: u32 = 0x8872;
pub const MAX_COMBINED_TEXTURE_IMAGE_UNITS: u32 = 0x8B4D;
pub const MAX_UNIFORM_BUFFER_BINDINGS: u32 = 0x8A2F;
pub const MAX_UNIFORM_BLOCK_SIZE: u32 = 0x8A30;
pub const UNIFORM_BUFFER_OFFSET_ALIGNMENT: u32 = 0x8A34;
pub const MAX_VERTEX_UNIFORM_BLOCKS: u32 = 0x8A2B;
pub const MAX_FRAGMENT_UNIFORM_BLOCKS: u32 = 0x8A2D;
pub const MAX_COMPUTE_UNIFORM_BLOCKS: u32 = 0x91BB;
pub const MAX_COMBINED_UNIFORM_BLOCKS: u32 = 0x8A2E;
pub const MAX_SHADER_STORAGE_BUFFER_BINDINGS: u32 = 0x90DD;
pub const MAX_SHADER_STORAGE_BLOCK_SIZE: u32 = 0x90DE;
pub const SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT: u32 = 0x90DF;
pub const MAX_VERTEX_SHADER_STORAGE_BLOCKS: u32 = 0x90D6;
pub const MAX_FRAGMENT_SHADER_STORAGE_BLOCKS: u32 = 0x90DA;
pub const MAX_COMPUTE_SHADER_STORAGE_BLOCKS: u32 = 0x90DB;
pub const MAX_COMBINED_SHADER_STORAGE_BLOCKS: u32 = 0x90DC;
pub const MAX_IMAGE_UNITS: u32 = 0x8F38;
pub const MAX_COMBINED_IMAGE_UNIFORMS: u32 = 0x90CF;
pub const MAX_SAMPLES: u32 = 0x8D57;
pub const MAX_COLOR_TEXTURE_SAMPLES: u32 = 0x910E;
pub const MAX_DEPTH_TEXTURE_SAMPLES: u32 = 0x910F;
pub const MAX_INTEGER_SAMPLES: u32 = 0x9110;
/// Views one attachment may serve in one pass, as defined by the multiview
/// extension family (the same registry value WebGL2 exposes).
pub const MAX_MULTIVIEW_VIEWS: u32 = 0x9631;
pub const MAX_COMPUTE_WORK_GROUP_COUNT: u32 = 0x91BE;
pub const MAX_COMPUTE_WORK_GROUP_SIZE: u32 = 0x91BF;
pub const MAX_COMPUTE_WORK_GROUP_INVOCATIONS: u32 = 0x90EB;
pub const MAX_TEXTURE_MAX_ANISOTROPY_EXT: u32 = 0x84FF;
/// The framebuffer bound for drawing, which tells the drawable queries
/// below whether they would answer about the surface at all.
pub const DRAW_FRAMEBUFFER_BINDING: u32 = 0x8CA6;
/// The draw framebuffer itself, as the target an attachment query is asked
/// about. The same object the binding query above reads the identity of.
pub const DRAW_FRAMEBUFFER: u32 = 0x8CA9;
pub const SAMPLE_BUFFERS: u32 = 0x80A8;
pub const SAMPLES: u32 = 0x80A9;
/// The drawable's colour buffers and its depth and stencil attachments.
///
/// The component widths are read from an attachment rather than from the
/// default-framebuffer `GL_*_BITS` queries this table used to carry: those
/// were deprecated in 3.0 and removed from the core profile in 3.1, so on
/// the core profile this crate accepts they answer `GL_INVALID_ENUM`, and
/// `glGetFramebufferAttachmentParameteriv` is the question the profile
/// still answers. The samples pair above is unaffected -- the core profile
/// kept it, which is visible on the hardware this was found on, where the
/// six bit queries failed and these two answered.
///
/// Values cross-checked against two independent bindings, because a wrong
/// token here is a question asked about nothing: `glow` 0.18 and `web-sys`
/// 0.3 agree on all six sizes and on the object type, and the two colour
/// buffers are consistent with the `FRONT`/`BACK` values the latter records
/// (`GL_FRONT` is `0x0404`, so `GL_FRONT_LEFT`/`GL_BACK_LEFT` are the two
/// entries below it).
pub const BACK: u32 = 0x0405;
pub const BACK_LEFT: u32 = 0x0402;
pub const FRONT_LEFT: u32 = 0x0400;
pub const DEPTH: u32 = 0x1801;
pub const STENCIL: u32 = 0x1802;
pub const FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE: u32 = 0x8CD0;
pub const FRAMEBUFFER_ATTACHMENT_RED_SIZE: u32 = 0x8212;
pub const FRAMEBUFFER_ATTACHMENT_GREEN_SIZE: u32 = 0x8213;
pub const FRAMEBUFFER_ATTACHMENT_BLUE_SIZE: u32 = 0x8214;
pub const FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE: u32 = 0x8215;
pub const FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE: u32 = 0x8216;
pub const FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE: u32 = 0x8217;
