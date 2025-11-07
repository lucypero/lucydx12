// This is the final pass that reads from the g-buffers, calculates lighting, and
//  outputs the final image.

#pragma pack_matrix(column_major)

struct PSInput
{
    float4 position : SV_Position; // Clip-space position
    float2 uvs : TEXCOORD0;        // UV coordinates passed to PS
};

PSInput VSMain(uint VertexID : SV_VertexID)
{
    PSInput output;
    output.uvs = float2(1, 1);
    output.position = float4(1, 1, 1, 1.0);
    return output;
}

float4 PSMain(PSInput input) : SV_TARGET
{
    return float4(0.0, 1.0, 0.0, 1.0);
}
