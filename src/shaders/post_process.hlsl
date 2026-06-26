// Post processing step. Compute shader.
// FXAA test.

#pragma pack_matrix(column_major)
#include "src/shaders/shader_common.hlsl"

[numthreads(1, 1, 1)]
void CSMain(
	uint3 groupID          : SV_GroupID,           // ID of the current thread group
	uint3 groupThreadID    : SV_GroupThreadID,     // ID of the thread within its group
	uint3 dispatchThreadID : SV_DispatchThreadID,  // Global pixel/data coordinate on the GPU
	uint groupIndex        : SV_GroupIndex         // Flattened 1D index of thread within group
) {

	ConstantBuffer<GeneralConstants> general_constants = ResourceDescriptorHeap[cbv_index];
	RWTexture2D<float4> result_texture = ResourceDescriptorHeap[general_constants.compute_out_idx];
	// lighting pass output. we'll process this.
	Texture2D<float3> in_texture = ResourceDescriptorHeap[general_constants.lighting_out_srv_idx];

	// Get the dimensions of the bound texture resource
	uint width, height;
	result_texture.GetDimensions(width, height);

	float3 light_out_color = in_texture[dispatchThreadID.xy];

	// do nothing
	result_texture[dispatchThreadID.xy] = float4(light_out_color.r, light_out_color.g, light_out_color.b, 1.0f);
}
