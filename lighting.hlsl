// This is the final pass that reads from the g-buffers, calculates lighting, and
//  outputs the final image.

#pragma pack_matrix(column_major)

Texture2D<float4> albedo : register(t1);
Texture2D<float4> normal : register(t2);
Texture2D<float4> position : register(t3);

SamplerState mySampler : register(s0);

cbuffer ConstantBuffer : register(b0)
{
    float4x4 view;
    float4x4 projection;
    float3 light_pos;
    float light_int;
    float3 view_pos;
    float time;
    bool place_texture;
};

struct PSInput
{
    float4 position : SV_Position;
    float2 uvs : TEXCOORD0;
};

PSInput VSMain(uint VertexID : SV_VertexID)
{
    PSInput output;

    const float2 positions[3] = {
        float2(-1.0, 3.0),
        float2(3.0, -1.0),
        float2(-1.0, -1.0),
    };

    const float2 uvs[3] = {
        float2(0.0, -1.0),
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
    float3 pixelColor = albedo.Sample(mySampler, input.uvs).xyz;
    float3 normalColor = normal.Sample(mySampler, input.uvs).xyz;
    float3 positionColor = position.Sample(mySampler, input.uvs).xyz;

    float3 norm = normalize(normalColor);
    float3 light_dir = normalize(light_pos - positionColor);

    // diffuse

    float diff = max(dot(norm, light_dir), 0.0f);
    float3 diffuse = diff * light_int;

    float amb_val = 0.05;

    float3 ambient = float3(amb_val, amb_val, amb_val);

    // Specular calculation

    float specularStrength = 0.5;

    float3 viewDir = normalize(view_pos - positionColor);
    float3 reflectDir = reflect(light_dir, norm);

    float3 spec_color = float3(1, 1, 1);

    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 128);
    float3 specular = specularStrength * spec * spec_color;

    float3 result = (ambient + diffuse + specular) * pixelColor;

    return float4(result, 1.0);
    // return float4(pixelColor.xyz * normalColor.xyz * positionColor.xyz, 1.0);
}
