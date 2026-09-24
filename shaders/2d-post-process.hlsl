#pragma pack_matrix(column_major)
#include "shaders/2d-common.hlsl"

// TODO do 8x8 thread groups
[numthreads(1, 1, 1)]
void CSMain(
	uint3 groupID          : SV_GroupID,           // ID of the current thread group
	uint3 groupThreadID    : SV_GroupThreadID,     // ID of the thread within its group
	uint3 dispatchThreadID : SV_DispatchThreadID,  // Global pixel/data coordinate on the GPU
	uint groupIndex        : SV_GroupIndex         // Flattened 1D index of thread within group
) {
	ConstantBuffer<GeneralConstants> general_constants = ResourceDescriptorHeap[cbv_index];
	RWTexture2D<float4> result_texture = ResourceDescriptorHeap[general_constants.tx_idx_post_process_out];
	// lighting pass output. we'll process this.
	Texture2D<float4> in_texture = ResourceDescriptorHeap[general_constants.tx_idx_quad_out];


	float4 original_color = in_texture[dispatchThreadID.xy];
	original_color.r *= 0.2;
	
	// write to result reading in_texture
	result_texture[dispatchThreadID.xy] = original_color;
}
