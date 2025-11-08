// This is the final pass that reads from the g-buffers, calculates lighting, and
//  outputs the final image.

#pragma pack_matrix(column_major)

Texture2D<float4> albedo : register(t1);
Texture2D<float4> normal : register(t2);
Texture2D<float4> position : register(t3);

SamplerState mySampler : register(s0);

struct PSInput
{
    float4 position : SV_Position; // Clip-space position
    float2 uvs : TEXCOORD0;        // UV coordinates passed to PS
};

PSInput VSMain(uint VertexID : SV_VertexID)
{
    PSInput output;

    const float2 positions[3] = {
        float2(-1.0, 3.0),  // Top-left (covers the top edge)
        float2(3.0, -1.0),  // Bottom-right (covers the right edge)
        float2(-1.0, -1.0), // Bottom-left
    };

    const float2 uvs[3] = {
        float2(0.0, -1.0), // Corresponding UVs to map positions to [0, 1] UV space
        float2(2.0, 1.0),
        float2(0.0, 1.0),
    };

    // Use the VertexID to look up the hardcoded data
    output.position = float4(positions[VertexID].xy, 0.0, 1.0);
    output.uvs = uvs[VertexID];

    return output;
}

float4 PSMain(PSInput input) : SV_TARGET
{
    float4 pixelColor = albedo.Sample(mySampler, input.uvs);
    float4 normalColor = normal.Sample(mySampler, input.uvs);
    float4 positionColor = position.Sample(mySampler, input.uvs);

    return float4(pixelColor.xyz * normalColor.xyz * positionColor.xyz, 1.0);
}
