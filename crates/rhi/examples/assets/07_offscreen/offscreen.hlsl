// Matches the captured C++ GLSL offscreen scene and Fluxel's framebuffer axes.
cbuffer Transform : register(b0) {
    float4x4 projection;
    float4x4 view;
    float4x4 model;
    float4 lightPos;
};

struct VertexInput {
    float3 position : LOCATION0;
    float3 normal : LOCATION1;
    float2 uv : LOCATION2;
    float3 color : LOCATION3;
};

struct PhongVertex {
    float4 position : SV_Position;
    float clip : SV_ClipDistance0;
    float3 normal : NORMAL0;
    float3 color : COLOR0;
    float3 eyePos : TEXCOORD0;
    float3 lightVec : TEXCOORD1;
};

PhongVertex phong_vs(VertexInput input) {
    PhongVertex output;
    float4 world = mul(model, float4(input.position, 1.0));
    float4 eye = mul(view, world);
    output.position = mul(projection, eye);
    // The captured C++ reference uses its GLSL path, whose clip plane is zero.
    output.clip = 0.0;
    output.normal = input.normal;
    output.color = input.color;
    output.eyePos = eye.xyz;
    output.lightVec = normalize(lightPos.xyz - eye.xyz);
    return output;
}

float4 phong_ps(PhongVertex input) : SV_Target {
    float3 eye = normalize(-input.eyePos);
    float3 reflected = normalize(reflect(-input.lightVec, input.normal));
    float4 ambient = float4(0.1, 0.1, 0.1, 1.0);
    float4 diffuse = max(dot(input.normal, input.lightVec), 0.0).xxxx;
    float4 specular = 0.0;
    if (dot(input.eyePos, input.normal) < 0.0) {
        specular = float4(0.5, 0.5, 0.5, 1.0)
            * pow(max(dot(reflected, eye), 0.0), 16.0) * 0.75;
    }
    return (ambient + diffuse) * float4(input.color, 1.0) + specular;
}

struct MirrorVertex {
    float4 position : SV_Position;
    float4 projected : TEXCOORD0;
};

MirrorVertex mirror_vs(VertexInput input) {
    MirrorVertex output;
    output.projected = mul(projection, mul(view, mul(model, float4(input.position, 1.0))));
    output.position = output.projected;
    return output;
}

Texture2D reflectionImage : register(t1);
SamplerState reflectionSampler : register(s2);

float4 mirror_ps(MirrorVertex input, bool frontFacing : SV_IsFrontFace) : SV_Target {
    if (!frontFacing) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    float2 projected = input.projected.xy / input.projected.w * 0.5 + 0.5;
    // Fluxel's +Y-up NDC maps to a top-left framebuffer origin.
    projected.y = 1.0 - projected.y;
    float4 color = 0.0;
    [unroll] for (int x = -3; x <= 3; ++x) {
        [unroll] for (int y = -3; y <= 3; ++y) {
            color += reflectionImage.Sample(
                reflectionSampler, projected + float2(x, y) / 512.0) / 49.0;
        }
    }
    return float4(color.rgb, 1.0);
}
