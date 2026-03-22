#pragma once

struct GeneralConstants {
    float4x4 view;
    float4x4 projection;
    float4x4 inverse_view_proj;
    float3 light_pos;
    float light_int;
    float3 view_pos;
    float time;
    uint current_scene_materials_idx;
    uint current_scene_mesh_transforms_idx;
};

struct AllSrvsIndices {
	int general_constants_idx;
	
	// g buffer
	int g_buffer_color_idx;
	int g_buffer_normal_idx;
	int g_buffer_ao_rough_metal_idx;
	
	// depth 
	int depth_idx;
};

AllSrvsIndices get_srvs_from_heap() {
	AllSrvsIndices idxs;
	
	idxs.g_buffer_color_idx = 0;
	idxs.g_buffer_normal_idx = 1;
	idxs.g_buffer_ao_rough_metal_idx = 2;
	
	idxs.depth_idx = 3;
	idxs.general_constants_idx = 4;
	
	return idxs;
}
