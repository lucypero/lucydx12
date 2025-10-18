#pragma pack_matrix(row_major)

struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uvs : TEXCOORD;
};

struct PSInput {
    float4 position : SV_POSITION;
    float3 frag_pos_world: POSITION;
    float3 frag_normal: NORMAL;
    float2 uvs : TEXCOORD0;
};

cbuffer ConstantBuffer : register(b0) {
    float4x4 wvp;
    float3 light_pos;
    float light_int;
    float someValue;
};

PSInput VSMain(VSInput the_input) {
    PSInput result;
    result.position = mul(float4(the_input.position, 1.0f), wvp);
    result.frag_pos_world = float3(the_input.position);
    result.frag_normal = the_input.normal;
    result.uvs = the_input.uvs.xy;
    return result;
}

Texture2D<float4> myTexture : register(t1);
SamplerState mySampler : register(s0);

float4 PSMain(PSInput input) : SV_TARGET {
    float4 pixelColor = myTexture.Sample(mySampler, input.uvs); // we need to pass UVs too

    float3 norm = normalize(input.frag_normal);
    float3 light_dir = normalize(light_pos - input.frag_pos_world);

    float diff = max(dot(norm, light_dir), 0.0f);
    float3 diffuse = diff * light_int;

    float amb_val = 0.05;

    float4 ambient = float4(amb_val, amb_val, amb_val, 0.0);

    return pixelColor * float4(diffuse, 1.0f) + pixelColor * ambient;
    // return float4(1,0,0,1);
}
