#pragma once

// Indices for different core things
struct AllSrvsIndices {
};

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
    
    // Other srv indices
    int general_constants_idx;
	
	// g buffer
	int g_buffer_color_idx;
	int g_buffer_normal_idx;
	int g_buffer_ao_rough_metal_idx;
	
	// depth 
	int depth_idx;
};

struct DrawConstants {
	uint mesh_index;
	uint material_index;
};

int GetCBVIndex() {
	return 3;
}
