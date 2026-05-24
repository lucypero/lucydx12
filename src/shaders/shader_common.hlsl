#pragma once

// Draw Constants for mesh drawing. this comes in as 2 32bit root constants
struct DrawConstants {
	uint mesh_index;
	uint material_index;
};

/// Root Parameters
SamplerState mySampler : register(s0);
int cbv_index: register (b0);
ConstantBuffer<DrawConstants> draw_constants : register(b1);

// Constant Buffer Struct Definition
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
	
	// g buffer
	int g_buffer_color_idx;
	int g_buffer_normal_idx;
	int g_buffer_ao_rough_metal_idx;
	
	// depth 
	int depth_idx;
};
