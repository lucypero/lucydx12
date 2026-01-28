// this is the first pass that populates all the g-buffers.

#pragma pack_matrix(column_major)
#include "shader_common.hlsl"

struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uvs : TEXCOORD0;
    float2 uvs_2 : TEXCOORD1;
    // instance data
    float4 worldM0  : WORLDMATRIX0; // Per-instance data (Slot 1)
    float4 worldM1  : WORLDMATRIX1;
    float4 worldM2  : WORLDMATRIX2;
    float4 worldM3  : WORLDMATRIX3;
    float3 color : COLOR;
};

struct PSInput {
    float4 position : SV_POSITION;
    float3 frag_pos_world: POSITION;
    float3 frag_normal: NORMAL;
    float2 uvs : TEXCOORD0;
    float2 uvs_2 : TEXCOORD1;
    float3 color: COLOR;
};

struct Material {
	uint base_color_index;
	uint base_color_uv_index;
};

struct MeshTransform
{
    float4x4 model; 
};

SamplerState mySampler : register(s0);

struct DrawConstants {
    uint mesh_index;
    uint material_index;
};

ConstantBuffer<DrawConstants> draw_constants : register(b1);

// cbv index is 3
// structured buffer index is %v 5

PSInput VSMain(VSInput the_input) {

	AllSrvsIndices srv_indexes = get_srvs_from_heap();

	ConstantBuffer<GeneralConstants> general_constants = ResourceDescriptorHeap[srv_indexes.general_constants_idx];
	StructuredBuffer<MeshTransform> mesh_transforms = ResourceDescriptorHeap[srv_indexes.mesh_transforms_idx];

    PSInput result;

    // use this for instanced drawing
    // float4x4 world_matrix = float4x4(the_input.worldM0, the_input.worldM1, the_input.worldM2, the_input.worldM3);
    // world_matrix = transpose(world_matrix);
    
    float4x4 world_matrix = mesh_transforms[draw_constants.mesh_index].model;

    float4 pos = float4(the_input.position, 1.0f);

    float4 world_position = mul(world_matrix, pos);
    // sometimes u have to flip it like this:
    // world_position.y = -world_position.y;
    // float4x4 world_position = mul(wvp, world_matrix);

    float4 view_position = mul(general_constants.view, world_position);

    // result.position = mul(pos, world_position);
    result.position = mul(general_constants.projection, view_position);

    result.frag_pos_world = world_position.xyz;
    
    // transforming normals by the normal matrix (a transformed world matrix)
    // this does not handle non-uniform scaling
    // TODO deal with that.
    result.frag_normal = mul((float3x3)world_matrix, the_input.normal);
    
    result.uvs = the_input.uvs.xy;
    // result.uvs.y = 1.0f - result.uvs.y;
    
    result.uvs_2 = the_input.uvs_2.xy;
    // result.uvs_2.y = 1.0f - result.uvs_2.y;
    
    result.color = the_input.color;
    return result;
}

struct PSOutput {
    // Target 0: Albedo Color (RGB) and Specular Intensity (A)
    // Common formats: DXGI_FORMAT_R8G8B8A8_UNORM
    float4 albedoSpecRT : SV_Target0; 

    // Target 1: World-Space Normals (RGB)
    // We use DXGI_FORMAT_R8G8B8A8_UNORM or similar. Normals are usually packed.
    float4 normalRT   : SV_Target1; 
};

PSOutput PSMain(PSInput input) {

	AllSrvsIndices srv_indexes = get_srvs_from_heap();

	ConstantBuffer<GeneralConstants> general_constants = ResourceDescriptorHeap[srv_indexes.general_constants_idx];
    PSOutput output;
    
    StructuredBuffer<Material> materials = ResourceDescriptorHeap[srv_indexes.materials_idx];
    
    Material mat = materials[draw_constants.material_index];

    float4 pixelColor = float4(input.color, 1.0);
    Texture2D<float4> baseColorTexture = ResourceDescriptorHeap[mat.base_color_index];
    
    float2 base_color_uvs = input.uvs;
    if(mat.base_color_uv_index != 0) {
    	base_color_uvs = input.uvs_2;
    }
    
    pixelColor = baseColorTexture.Sample(mySampler, base_color_uvs);

    float3 norm = normalize(input.frag_normal);
    
    // writing to all gbuffers
    output.albedoSpecRT = pixelColor;
    output.normalRT.rgb = (norm * 0.5f) + 0.5f;
    output.normalRT.a = 1.0f;
    
    return output;
}
