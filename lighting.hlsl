// This is the final pass that reads from the g-buffers, calculates lighting, and
//  outputs the final image.

#pragma pack_matrix(column_major)
#include "shader_common.hlsl"

SamplerState mySampler : register(s0);

// Assuming: 
// uv: [0, 1] across the screen
// depth: sampled from your depth buffer [0, 1]

float4 GetWorldPosition(float2 uv, float depth, float4x4 invViewProj) {
    // Convert UV to [-1, 1] NDC range. 
    // Note: Y is often flipped depending on your API (Vulkan vs DX)
    float2 ndcXY = uv * 2.0 - 1.0;
    ndcXY.y = -ndcXY.y; // Flip Y for DirectX

    // Create the NDC position vector
    float4 ndcPos = float4(ndcXY, depth, 1.0);

    // Transform by Inverse View-Projection
    float4 worldPos = mul(invViewProj, ndcPos);

    // Perspective Divide
    return worldPos / worldPos.w;
}

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
	AllSrvsIndices srv_indexes = get_srvs_from_heap();
	
	ConstantBuffer<GeneralConstants> general_constants = ResourceDescriptorHeap[srv_indexes.general_constants_idx];
	
	// g buffer
	Texture2D<float4> albedo = ResourceDescriptorHeap[srv_indexes.g_buffer_color_idx];
	Texture2D<float4> normal = ResourceDescriptorHeap[srv_indexes.g_buffer_normal_idx];
	
	// depth
	Texture2D<float4> depthTexture = ResourceDescriptorHeap[srv_indexes.depth_idx];

    float3 albedoColor = albedo.Sample(mySampler, input.uvs).xyz;
    float3 normalColor = normal.Sample(mySampler, input.uvs).xyz;
    // float3 worldPosition = position.Sample(mySampler, input.uvs).xyz;
    // float3 worldPosition = position.Sample(mySampler, input.uvs).xyz;
    
    // In the Pixel Shader
    float depth = depthTexture.Sample(mySampler, input.uvs).r;
    
    // float depth = ???
    
    
    float3 worldPosition = GetWorldPosition(input.uvs, depth, general_constants.inverse_view_proj).xyz;

    // calculating normal
    
    float3 norm = 0.0f;
    {
        norm = normalize(normalColor); // range: [0.0, 1.0]
        
        // unmap normal value back to -1 to 1 range
        norm = (norm * 2.0f) - 1.0f;
        
        // normalize again to be safe
        norm = normalize(norm);
    }
    
    float3 light_dir = normalize(general_constants.light_pos - worldPosition);

    // calculating diffuse
    float3 diffuse = 0;
    {
        float diff = max(dot(norm, light_dir), 0.0f);
        diffuse = diff * general_constants.light_int;
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
    
        float3 viewDir = normalize(general_constants.view_pos - worldPosition);
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
    // return float4(worldPosition, 1.0);
    
    // -- Display Final image
    return float4(result, 1.0);
    // return float4(depth, 0.0, 0.0, 1.0);
}
