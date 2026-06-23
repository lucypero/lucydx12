#pragma pack_matrix(column_major)
#include "src/shaders/shader_common.hlsl"

// Post processing step. Compute shader.
// FXAA test.


// 1. RESOURCE BINDINGS
// RWTexture2D means a Read-Write 2D Texture (Unordered Access View / UAV)
RWTexture2D<float4> ResultTexture : register(u0);

[numthreads(8, 8, 1)]
void CSMain(
	// 3. SYSTEM VALUE INPUTS
	uint3 groupID          : SV_GroupID,           // ID of the current thread group
	uint3 groupThreadID    : SV_GroupThreadID,     // ID of the thread within its group
	uint3 dispatchThreadID : SV_DispatchThreadID,  // Global pixel/data coordinate on the GPU
	uint groupIndex        : SV_GroupIndex         // Flattened 1D index of thread within group
) {
	// Get the dimensions of the bound texture resource
	uint width, height;
	ResultTexture.GetDimensions(width, height);

}
