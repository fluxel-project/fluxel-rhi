// DirectX equivalent of shaders/glsl/screenshot/mesh.{vert,frag}.
// The constant-buffer layout deliberately mirrors the C++ sample's UBO.
cbuffer Transform : register(b0)
{
    float4x4 projection;
    float4x4 view;
    float4x4 model;
};

struct VertexInput
{
    float3 position : LOCATION0;
    float3 normal   : LOCATION1;
    float4 color    : LOCATION2;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float3 normal   : NORMAL0;
    float3 color    : COLOR0;
    float3 view_vec : TEXCOORD0;
    float3 light_vec : TEXCOORD1;
};

VertexOutput vs_main(VertexInput input)
{
    VertexOutput output;
    float4 model_position = mul(model, float4(input.position, 1.0));
    float4 view_position = mul(view, model_position);
    output.position = mul(projection, view_position);
    // The C++ GLSL source transforms normals by its `model` field only.  Its
    // CPU UBO writes `projection, model, view` while GLSL declares
    // `projection, view, model`, so this is the camera matrix in both native
    // shader routes.
    output.normal = mul((float3x3)model, input.normal);
    output.color = input.color.rgb;
    output.view_vec = -view_position.xyz;
    // The source Vulkan loader reflects mesh Y, whereas Fluxel keeps model
    // space WebGPU-style and reflects the viewport.  Reflect the source light
    // through the same Y axis so lighting remains in the model's coordinates.
    output.light_vec = float3(1.0, 1.0, 1.0) - view_position.xyz;
    return output;
}

float4 ps_main(VertexOutput input) : SV_Target
{
    float3 normal = normalize(input.normal);
    float3 light = normalize(input.light_vec);
    float3 view = normalize(input.view_vec);
    float3 reflected = reflect(-light, normal);
    float3 ambient = float3(0.1, 0.1, 0.1);
    float3 diffuse = max(dot(normal, light), 0.0).xxx;
    float3 specular = pow(max(dot(reflected, view), 0.0), 16.0).xxx * 0.75;
    return float4((ambient + diffuse) * input.color + specular, 1.0);
}
