// this is the first pass that populates all the g-buffers.

#pragma pack_matrix(column_major)
#include "shader_common.hlsl"

struct VSInput {
	float3 position : POSITION;
	float3 normal : NORMAL;
	float4 tangent : TANGENT;
	float2 uvs : TEXCOORD0;
	float2 uvs_2 : TEXCOORD1;
	// instance data
	float4 worldM0  : WORLDMATRIX0; // Per-instance data (Slot 1)
	float4 worldM1  : WORLDMATRIX1;
	float4 worldM2  : WORLDMATRIX2;
	float4 worldM3  : WORLDMATRIX3;
	float3 color : COLOR;
};

struct TextureUV {
	uint texture_id;
	uint uv_id;
};

struct Material {
	TextureUV base_color;
	TextureUV metallic_roughness;
	TextureUV normal;
};

struct MeshTransform
{
	float4x4 model; 
};

SamplerState mySampler : register(s0);

struct DrawConstants {
	uint mesh_index;
	uint material_index;
};

ConstantBuffer<DrawConstants> draw_constants : register(b1);

struct PSInput {
	float4 position : SV_POSITION;
	float3 frag_pos_world: POSITION;
	float3 normal_world: NORMAL;
	float3 tangent_world: TANGENT;
	float3 bitangent_world: BITANGENT;
	float2 uvs : TEXCOORD0;
	float2 uvs_2 : TEXCOORD1;
	float3 color: COLOR;
};

PSInput VSMain(VSInput the_input) {

	AllSrvsIndices srv_indexes = get_srvs_from_heap();
	
	ConstantBuffer<GeneralConstants> general_constants = ResourceDescriptorHeap[srv_indexes.general_constants_idx];
	StructuredBuffer<MeshTransform> mesh_transforms = ResourceDescriptorHeap[srv_indexes.mesh_transforms_idx];
	
	PSInput result;
	
	// use this for instanced drawing
	// float4x4 world_matrix = float4x4(the_input.worldM0, the_input.worldM1, the_input.worldM2, the_input.worldM3);
	// world_matrix = transpose(world_matrix);
	
	float4x4 world_matrix = mesh_transforms[draw_constants.mesh_index].model;
	
	float4 pos = float4(the_input.position, 1.0f);
	
	float4 world_position = mul(world_matrix, pos);
	// sometimes u have to flip it like this:
	// world_position.y = -world_position.y;
	// float4x4 world_position = mul(wvp, world_matrix);
	
	float4 view_position = mul(general_constants.view, world_position);
	
	// result.position = mul(pos, world_position);
	result.position = mul(general_constants.projection, view_position);
	
	result.frag_pos_world = world_position.xyz;
	
	// transforming normals by the normal matrix (a transformed world matrix)
	// this does not handle non-uniform scaling
	// TODO deal with that.
	float3 N = mul((float3x3)world_matrix, the_input.normal);
	result.normal_world = N;
	
	float3 T = mul((float3x3)world_matrix, the_input.tangent.xyz);
	
	// 1. Gram-Schmidt Re-orthogonalization
	// This ensures T is exactly 90 degrees to N
	T = normalize(T - dot(T, N) * N);
	
	result.tangent_world = T;
	
	// Calculate Bitangent (B = N x T)
	result.bitangent_world = cross(N, T) * the_input.tangent.w * -1;
	
	// Optional: If your mesh has mirrored UVs, you often store a 'w' 
	// component in the tangent to flip the bitangent direction
	// float3 bitangentWS = cross(normalWS, tangentWS) * input.Tangent.w;
	
	result.uvs = the_input.uvs.xy;
	// result.uvs.y = 1.0f - result.uvs.y;
	
	result.uvs_2 = the_input.uvs_2.xy;
	// result.uvs_2.y = 1.0f - result.uvs_2.y;
	
	result.color = the_input.color;
	return result;
}



struct PSOutput {
	float4 albedoRT : SV_Target0; 
	float4 normalRT : SV_Target1; 
	// X channel: AO --- Y channel: roughness --- Z channel: metalness
	float4 AoRoughMetalRT : SV_Target2; 
};

PSOutput PSMain(PSInput input) {
	AllSrvsIndices srv_indexes = get_srvs_from_heap();
	
	ConstantBuffer<GeneralConstants> general_constants = ResourceDescriptorHeap[srv_indexes.general_constants_idx];
	PSOutput output;
	
	StructuredBuffer<Material> materials = ResourceDescriptorHeap[srv_indexes.materials_idx];
	
	Material mat = materials[draw_constants.material_index];
	
	
	// Albedo map
	{
		float4 albedoColor;
		Texture2D<float4> baseColorTexture = ResourceDescriptorHeap[mat.base_color.texture_id];
		
		float2 base_color_uvs = input.uvs;
		if(mat.base_color.uv_id != 0) {
			base_color_uvs = input.uvs_2;
		}
	
		albedoColor = baseColorTexture.Sample(mySampler, base_color_uvs);
		output.albedoRT = albedoColor;
	}
	
	// Normal map
	{
		float3 N = normalize(input.normal_world); // N is normal vector
		float3 T = normalize(input.tangent_world); // T is tangent vector
		float3 B = normalize(input.bitangent_world); // B is bitangent vector
		
		T = normalize(T - dot(T, N) * N);
		// Recompute B to ensure a perfectly orthogonal system
		B = cross(N, T);
		
		// Form the matrix
		float3x3 TBN = float3x3(T, B, N);
		TBN = transpose(TBN);
		
		
		// Sample and transform
		Texture2D<float4> normalMapTexture = ResourceDescriptorHeap[mat.normal.texture_id];
		
		float2 normal_map_uv = input.uvs;
		if (mat.normal.uv_id != 0) {
			normal_map_uv = input.uvs_2;
		}
		
		float2 normalXY = normalMapTexture.Sample(mySampler, normal_map_uv).rg;
		normalXY = normalXY * 2.0 - 1.0;
		
		// flipping green channel
		normalXY.y *= -1;
		
		// 2. Reconstruct Z (the blue channel)
		// Since the normal is a unit vector, z = sqrt(1 - x^2 - y^2)
		// We use saturate to ensure we don't sqrt a negative number due to precision
		float normalZ = sqrt(saturate(1.0 - dot(normalXY, normalXY)));
		
		// 3. Final reconstructed normal in Tangent Space
		float3 normalSample = float3(normalXY, normalZ);
		
		const float normalStrength = 0.5f;
		normalSample.xy *= normalStrength;
		float3 worldNormal = normalize(mul(TBN, normalSample));
		// float3 worldNormal = normalize(mul(normalSample, TBN));
		
		// enconding normal to map
		float4 normalColor;
		normalColor.xyz = worldNormal;
		normalColor.xyz = (normalColor.xyz * 0.5f) + 0.5f;
		normalColor.a = 1.0f;
		
		output.normalRT = normalColor;
		
		// uncomment to skip normal map
		// output.normalRT = float4(N * 0.5 + 0.5, 1.0);
	}
	
	// AO + Rough + Metalness map
	{
		float4 aoRoughMetalColor;
		Texture2D<float4> metalRoughTexture = ResourceDescriptorHeap[mat.metallic_roughness.texture_id];
		
		float2 base_color_uvs = input.uvs;
		if(mat.metallic_roughness.uv_id != 0) {
			base_color_uvs = input.uvs_2;
		}
	
		aoRoughMetalColor = metalRoughTexture.Sample(mySampler, base_color_uvs);
		// AO (x channel)
		output.AoRoughMetalRT.x = 1.0f;
		
		// Roughness (y channel)
		output.AoRoughMetalRT.y = aoRoughMetalColor.y; // roughness is at the Y channel in the texture
		
		// Metalness (z channel)
		output.AoRoughMetalRT.z = aoRoughMetalColor.x; // metalness is at the X channel in the texture
		
		output.AoRoughMetalRT.w = 1.0f;
	}
	
	return output;
}
