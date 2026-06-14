// this renders shadowmaps

#pragma pack_matrix(column_major)
#include "src/shaders/shader_common.hlsl"

struct VSInput {
	float3 position : POSITION;
	float3 normal : NORMAL;
	float4 tangent : TANGENT;
	float2 uvs : TEXCOORD;
	float2 uvs_2 : TEXCOORD_SECOND_UV;
};

struct PSInput {
	float4 position : SV_POSITION;
};

struct MeshTransform
{
	float4x4 model; 
};

// just do position processing
PSInput VSMain(VSInput the_input) {

	ConstantBuffer<GeneralConstants> general_constants = ResourceDescriptorHeap[cbv_index];
	StructuredBuffer<MeshTransform> mesh_transforms = ResourceDescriptorHeap[general_constants.current_scene_mesh_transforms_idx];

	PSInput result;

	float4x4 world_matrix = mesh_transforms[draw_constants.mesh_index].model;
	float4 pos = float4(the_input.position, 1.0f);
	float4 world_position = mul(world_matrix, pos);
	float4 view_position = mul(general_constants.light_view, world_position);

	// result.position = mul(pos, world_position);
	result.position = mul(general_constants.light_projection, view_position);
	return result;
}

void PSMain(PSInput input) : SV_TARGET {
	return;
}
