// this renders all the gizmos (light position, etc)

#pragma pack_matrix(column_major)
#include "shader_common.hlsl"

struct VSInput {
	float3 position : POSITION;
	// instance data
	float4 worldM0  : WORLDMATRIX0;
	float4 worldM1  : WORLDMATRIX1;
	float4 worldM2  : WORLDMATRIX2;
	float4 worldM3  : WORLDMATRIX3;
	float4 color: COLOR0;
};

struct PSInput {
	float4 position : SV_POSITION;
	float4 color: COLOR;
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
	
	// instanced drawing
	float4x4 world_matrix = float4x4(the_input.worldM0, the_input.worldM1, the_input.worldM2, the_input.worldM3);
	world_matrix = transpose(world_matrix);
	// float4x4 world_matrix = mesh_transforms[draw_constants.mesh_index].model;
	
	float4 pos = float4(the_input.position, 1.0f);
	float4 world_position = mul(world_matrix, pos);
	float4 view_position = mul(general_constants.view, world_position);
	
	PSInput result;
	
	result.position = mul(general_constants.projection, view_position);
	result.color = the_input.color;
	return result;
}

float4 PSMain(PSInput input) : SV_TARGET {
	return float4(input.color);
}
