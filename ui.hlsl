// this renders all the gizmos (light position, etc)

#pragma pack_matrix(column_major)
#include "shader_common.hlsl"

struct VSInput {
	float3 position : POSITION;
};

struct PSInput {
	float4 position : SV_POSITION;
};

struct DrawConstants {
	uint mesh_index;
	uint material_index;
};

struct MeshTransform
{
	float4x4 model; 
};

ConstantBuffer<DrawConstants> draw_constants : register(b1);

PSInput VSMain(VSInput the_input) {
	
	AllSrvsIndices srv_indexes = get_srvs_from_heap();
	ConstantBuffer<GeneralConstants> general_constants = ResourceDescriptorHeap[srv_indexes.general_constants_idx];
	StructuredBuffer<MeshTransform> mesh_transforms = ResourceDescriptorHeap[srv_indexes.mesh_transforms_idx];
	
	float4x4 world_matrix = mesh_transforms[draw_constants.mesh_index].model;
	float4 pos = float4(the_input.position, 1.0f);
	float4 world_position = mul(world_matrix, pos);
	float4 view_position = mul(general_constants.view, world_position);
	
	PSInput result;
	
	result.position = mul(general_constants.projection, view_position);
	return result;
}

float4 PSMain(PSInput input) : SV_TARGET {
	return float4(1.0, 1.0, 1.0, 1.0);
}
