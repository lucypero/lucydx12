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
    float3 albedoColor = albedo.Sample(mySampler, input.uvs).xyz;
    float3 normalColor = normal.Sample(mySampler, input.uvs).xyz;
    float3 positionColor = position.Sample(mySampler, input.uvs).xyz;

    // calculating normal
    
    float3 norm = 0.0f;
    {
        norm = normalize(normalColor); // range: [0.0, 1.0]
        
        // unmap normal value back to -1 to 1 range
        norm = (norm * 2.0f) - 1.0f;
        
        // normalize again to be safe
        norm = normalize(norm);
    }
    
    float3 light_dir = normalize(light_pos - positionColor);

    // calculating diffuse
    float3 diffuse = 0;
    {
        float diff = max(dot(norm, light_dir), 0.0f);
        diffuse = diff * light_int;
    }

    // calculating ambient
    float3 ambient = 0;
    {
        float amb_val = 0.05;
        ambient = float3(amb_val, amb_val, amb_val);
    }

    // Specular calculation
    
    float3 specular = 0;
    {
        float specularStrength = 0.5;
    
        float3 viewDir = normalize(view_pos - positionColor);
        float3 reflectDir = reflect(light_dir, norm);
    
        float3 spec_color = float3(1, 1, 1);
    
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 128);
        specular = specularStrength * spec * spec_color;
    }

    float3 result = (ambient + diffuse + specular) * albedoColor;
    
    // -- Display g buffer 1
    // return float4(albedoColor, 1.0);
    // -- Display g buffer 2
    // return float4(normalColor, 1.0);
    // -- Display g buffer 3
    // return float4(positionColor, 1.0);
    
    // -- Display Final image
    return float4(result, 1.0);
    
}
