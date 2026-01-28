#pragma once

struct GeneralConstants {
    float4x4 view;
    float4x4 projection;
    float4x4 inverse_view_proj;
    float3 light_pos;
    float light_int;
    float3 view_pos;
    float time;
};

struct AllSrvsIndices {
	int general_constants_idx;
	int mesh_transforms_idx;
	int materials_idx;
	
	// g buffer
	int g_buffer_color_idx;
	int g_buffer_normal_idx;
	
	
	// depth 
	
	int depth_idx;
};

AllSrvsIndices get_srvs_from_heap() {
	AllSrvsIndices idxs;
	
	idxs.g_buffer_color_idx = 0;
	idxs.g_buffer_normal_idx = 1;
	
	idxs.depth_idx = 2;
	idxs.materials_idx = 3;
	idxs.general_constants_idx = 4;
	idxs.mesh_transforms_idx = 5;
	
	return idxs;
}
