#pragma pack_matrix(row_major)

struct VSInput {
    float3 position : POSITION;
    float2 uvs : TEXCOORD;
    float4 color : COLOR;
};

struct PSInput {
    float4 position : SV_POSITION;
    float2 uvs : TEXCOORD0;
    float4 color : COLOR;
};

cbuffer ConstantBuffer : register(b0) {
    float4x4 wvp;
    float someValue;
};

PSInput VSMain(VSInput the_input) {
    PSInput result;
    // result.position = mul(wvp, float4(the_input.position, 1.0f));
    result.position = mul(float4(the_input.position, 1.0f), wvp);
    result.uvs = the_input.uvs.xy;
    result.color = the_input.color;
    return result;
}

Texture2D<float4> myTexture : register(t1);
SamplerState mySampler : register(s0);

float4 PSMain(PSInput input) : SV_TARGET {
    float4 pixelColor = myTexture.Sample(mySampler, input.uvs); // we need to pass UVs too
    return pixelColor;
    // return float4(1,0,0,1);
    //return input.color; 
}
