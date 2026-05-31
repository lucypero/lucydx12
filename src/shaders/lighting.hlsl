// This is the final pass that reads from the g-buffers, calculates lighting, and
//  outputs the final image.

#pragma pack_matrix(column_major)
#include "src/shaders/shader_common.hlsl"

// Light struct

enum LightType : uint32_t {
	Directional,
	Point,
};

struct Light {
	LightType type;
	float3 position;

	float radius;
	float3 direction;

	float intensity;
	float3 color;
};

struct PSInput
{
	float4 position : SV_Position;
	float2 uvs : TEXCOORD0;
};


// Assuming: 
// uv: [0, 1] across the screen
// depth: sampled from your depth buffer [0, 1]

float4 GetWorldPosition(float2 uv, float depth, float4x4 invViewProj) {
	// Convert UV to [-1, 1] NDC range. 
	// Note: Y is often flipped depending on your API (Vulkan vs DX)
	float2 ndcXY = uv * 2.0 - 1.0;
	ndcXY.y = -ndcXY.y; // Flip Y for DirectX

	// Create the NDC position vector
	float4 ndcPos = float4(ndcXY, depth, 1.0);

	// Transform by Inverse View-Projection
	float4 worldPos = mul(invViewProj, ndcPos);

	// Perspective Divide
	return worldPos / worldPos.w;
}

PSInput VSMain(uint VertexID : SV_VertexID)
{
	PSInput output;

	const float2 positions[3] = {
		float2(-1.0, 3.0),
		float2(3.0, -1.0),
		float2(-1.0, -1.0),
	};

	const float2 uvs[3] = {
		float2(0.0, -1.0),
		float2(2.0, 1.0),
		float2(0.0, 1.0),
	};

	// Use the VertexID to look up the hardcoded data
	output.position = float4(positions[VertexID].xy, 0.0, 1.0);
	output.uvs = uvs[VertexID];

	return output;
}


// helper code

// 1. Normal Distribution Function (GGX)
// Approximates the amount of microfacets aligned with the Halfway vector
float DistributionGGX(float3 N, float3 H, float roughness) {
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = max(dot(N, H), 0.0);
	float NdotH2 = NdotH * NdotH;

	float num = a2;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = 3.14159 * denom * denom;

	return num / denom;
}

// 2. Geometry Function (Smith's method)
// Approximates self-shadowing of the microfacets
float GeometrySchlickGGX(float NdotV, float roughness) {
	float r = (roughness + 1.0);
	float k = (r * r) / 8.0;
	return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(float3 N, float3 V, float3 L, float roughness) {
	float NdotV = max(dot(N, V), 0.0);
	float NdotL = max(dot(N, L), 0.0);
	float ggx2 = GeometrySchlickGGX(NdotV, roughness);
	float ggx1 = GeometrySchlickGGX(NdotL, roughness);
	return ggx1 * ggx2;
}

// 3. Fresnel Equation (Schlick's approximation)
// Ratio of light reflected vs refracted
float3 FresnelSchlick(float cosTheta, float3 F0) {
	return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Helper function to turn a 0.0 - 1.0 value into a rainbow
float3 DebugHueGradient(float t)
{
	// These coefficients create a standard "Spectral" rainbow
	// Formula: color = a + b * cos(2 * PI * (c * t + d))
	float3 a = float3(0.5, 0.5, 0.5);
	float3 b = float3(0.5, 0.5, 0.5);
	float3 c = float3(1.0, 1.0, 1.0);
	float3 d = float3(0.0, 0.33, 0.67);

	return a + b * cos(6.28318 * (c * t + d));
}

float3 ComputeDirectionalLight(Light light, float3 worldPosition, float3 norm, float3 albedoColor, float3 aoRoughMetalColor, float3 view_pos) {

	float3 light_dir = normalize(light.direction);

	// calculating diffuse
	float3 diffuse = 0;
	{
		float diff = max(dot(norm, light_dir), 0.0f);
		diffuse = diff;
	}

	// Specular calculation
	float3 specular = 0;
	{
		float roughness = aoRoughMetalColor.y;

		float3 N = norm;
		float3 V = normalize(view_pos - worldPosition);
		float3 L = normalize(light_dir);
		float3 H = normalize(V + L);

		// F0 is the base reflectivity (0.04 for non-metals)
		float3 F0 = float3(0.04, 0.04, 0.04); 

		// Cook-Torrance BRDF components
		float D = DistributionGGX(N, H, roughness);
		float G = GeometrySmith(N, V, L, roughness);
		float3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);

		// Final Specular calculation
		float3 numerator = D * G * F;
		float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // prevent div by zero
		specular = numerator / denominator;

		float NdotL = max(dot(N, L), 0.0);
		specular = specular * float3(1,1,1) * NdotL;
	}

	return (diffuse + specular) * albedoColor * light.intensity;
};

float3 ComputePointLight(Light light, float3 worldPosition, float3 norm, float3 albedoColor, float3 aoRoughMetalColor, float3 view_pos) {

	float3 light_dir = normalize(light.position - worldPosition);

	// do light thing here

	// calculating diffuse
	float3 diffuse = 0;
	{
		float diff = max(dot(norm, light_dir), 0.0f);
		diffuse = diff * light.intensity;
	}

	// Specular calculation
	float3 specular = 0;
	{
		float roughness = aoRoughMetalColor.y;

		float3 N = norm;
		float3 V = normalize(view_pos - worldPosition);
		float3 L = normalize(light_dir);
		float3 H = normalize(V + L);

		// F0 is the base reflectivity (0.04 for non-metals)
		float3 F0 = float3(0.04, 0.04, 0.04); 

		// Cook-Torrance BRDF components
		float D = DistributionGGX(N, H, roughness);
		float G = GeometrySmith(N, V, L, roughness);
		float3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);

		// Final Specular calculation
		float3 numerator = D * G * F;
		float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // prevent div by zero
		specular = numerator / denominator;

		float NdotL = max(dot(N, L), 0.0);
		specular = specular * float3(1,1,1) * NdotL;
	}

	return (diffuse + specular) * albedoColor;
};

//  / helper code

float4 PSMain(PSInput input) : SV_TARGET
{
	ConstantBuffer<GeneralConstants> general_constants = ResourceDescriptorHeap[cbv_index];

	// g buffer
	Texture2D<float4> albedo = ResourceDescriptorHeap[general_constants.g_buffer_color_idx];
	Texture2D<float4> normal = ResourceDescriptorHeap[general_constants.g_buffer_normal_idx];
	Texture2D<float4> ao_rough_metal = ResourceDescriptorHeap[general_constants.g_buffer_ao_rough_metal_idx];

	// depth
	Texture2D<float4> depthTexture = ResourceDescriptorHeap[general_constants.depth_idx];

	float3 albedoColor = albedo.Sample(mySampler, input.uvs).xyz;
	float3 normalColor = normal.Sample(mySampler, input.uvs).xyz;
	float3 aoRoughMetalColor = ao_rough_metal.Sample(mySampler, input.uvs).xyz;

	// In the Pixel Shader
	float depth = depthTexture.Sample(mySampler, input.uvs).r;

	float3 worldPosition = GetWorldPosition(input.uvs, depth, general_constants.inverse_view_proj).xyz;

	// calculating normal

	float3 norm = 0.0f;
	{
		norm = normalize(normalColor); // range: [0.0, 1.0]

		// unmap normal value back to -1 to 1 range
		norm = (norm * 2.0f) - 1.0f;

		// normalize again to be safe
		norm = normalize(norm);
	}

	// Calculate all lights

	float3 result = 0;

	StructuredBuffer<Light> lights = ResourceDescriptorHeap[general_constants.light_sb_idx];

	float3 ambient = 0;
	// calculating ambient
	{
		float amb_val = 0.05;
		ambient = float3(amb_val, amb_val, amb_val);
	}

	result = ambient;

	for(int i = 0; i<general_constants.light_count; ++i) {

		Light light = lights[i];

		switch(light.type) {
		case Directional:
			result += ComputeDirectionalLight(light, worldPosition, norm, albedoColor, aoRoughMetalColor, general_constants.view_pos);
			break;
		case Point:
			result += ComputePointLight(light, worldPosition, norm, albedoColor, aoRoughMetalColor, general_constants.view_pos);
			break;
		}
	}

	// -- Display Final image

	// 2. Define the Minimap's position and size in UV space (0.0 to 1.0)
	// Let's place it in the top-right corner.
	float width  = 0.2;  // 20% of screen width
	float height = 0.2;  // 20% of screen height
	float posX   = 0.75; // Starts at 75% across the screen
	float posY   = 0.05; // Starts at 5% down the screen

	// Calculate the boundaries of the minimap box
	float minX = posX;
	float maxX = posX + width;
	float minY = posY;
	float maxY = posY + height;

	// 3. Check if the current pixel being rendered is inside the minimap box
	if (general_constants.draw_shadowmap && input.uvs.x >= minX && input.uvs.x <= maxX && 
		input.uvs.y >= minY && input.uvs.y <= maxY) 
	{
		// 4. Remap the full-screen UVs to local UVs (0.0 to 1.0) for the texture
		float2 localUV;
		localUV.x = (input.uvs.x - minX) / width;
		localUV.y = (input.uvs.y - minY) / height;

		// 5. Sample the minimap texture using the new local UVs
		Texture2D<float4> shadowmap = ResourceDescriptorHeap[general_constants.shadowmap_idx];
		float4 minimapColor = shadowmap.Sample(mySampler, localUV);

		// 6. Composite the minimap over the main scene
		// Using simple alpha blending here: lerp(background, foreground, foregroundAlpha)
		// result = lerp(result, minimapColor, minimapColor.a);

		// Alternatively, if your texture has no alpha channel and you just want to overwrite:
		result = minimapColor.xxx;

	}

	return float4(result, 1.0);
}
