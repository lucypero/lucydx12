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
	
	// (for slug text)
	int param_struct_idx;
	int curve_texture_idx;
	int band_texture_idx;
};

AllSrvsIndices get_srvs_from_heap() {
	AllSrvsIndices idxs;
	
	idxs.g_buffer_color_idx = 0;
	idxs.g_buffer_normal_idx = 1;
	idxs.g_buffer_ao_rough_metal_idx = 2;
	
	idxs.general_constants_idx = 3;
	idxs.depth_idx = 4;
	
	idxs.param_struct_idx = 5;
	
	// not done yet
	idxs.curve_texture_idx = 6;
	idxs.band_texture_idx = 7;
	
	return idxs;
}
