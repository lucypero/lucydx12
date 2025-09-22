#pragma pack_matrix(row_major)

struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uvs : TEXCOORD;
};

struct PSInput {
    float4 position : SV_POSITION;
    float2 uvs : TEXCOORD0;
};

cbuffer ConstantBuffer : register(b0) {
    float4x4 wvp;
    float someValue;
};

PSInput VSMain(VSInput the_input) {
    PSInput result;
    result.position = mul(float4(the_input.position, 1.0f), wvp);
    result.uvs = the_input.uvs.xy;
    return result;
}

Texture2D<float4> myTexture : register(t1);
SamplerState mySampler : register(s0);

float4 PSMain(PSInput input) : SV_TARGET {
    float4 pixelColor = myTexture.Sample(mySampler, input.uvs); // we need to pass UVs too
    return pixelColor;
    // return float4(1,0,0,1);
}
