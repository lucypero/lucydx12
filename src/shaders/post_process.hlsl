// Post processing step. Compute shader.
// FXAA test.

// Including FXAA

#ifdef FXAA_ENABLE

#define FXAA_PC 1
#define FXAA_HLSL_5 1
#define FXAA_QUALITY__PRESET 12

#include "src/shaders/FXAA3_11.hlsl"

#endif

// Rest of shader

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
	Texture2D<float4> in_texture = ResourceDescriptorHeap[general_constants.lighting_out_srv_idx];

	// Get the dimensions of the bound texture resource
	uint width, height;
	result_texture.GetDimensions(width, height);

	#ifdef FXAA_ENABLE
	float2 pixel_pos = float2(float(dispatchThreadID.x) / float(width), float(dispatchThreadID.y) / float(height));

	FxaaTex fxaa_tex;
	fxaa_tex.smpl = sampler_linear;
	fxaa_tex.tex = in_texture;

	// lucyfmt bugs out here ( when u pass {}, it unindents once). fix it later.
	FxaaFloat4 fxaa_out = FxaaPixelShader(
		pixel_pos, // pos
		0, // fxaaConsolePosPos
		fxaa_tex, // tex
		fxaa_tex, // fxaaConsole360TexExpBiasNegOne (not used)
		fxaa_tex, // fxaaConsole360TexExpBiasNegTwo (not used)
		// TODO(lucy): Bake this into a constant
		float2(1.0 / float(width), 1.0 / float(height)), // fxaaQualityRcpFrame (probably used)
		0, // fxaaConsoleRcpFrameOpt (not used)
		0, // fxaaConsoleRcpFrameOpt2 (not used)
		0, // fxaaConsole360RcpFrameOpt2 (not used)
		0.75, // fxaaQualitySubpix (probably used)
		0.125, // fxaaQualityEdgeThreshold (probably used)
		0.0312, // fxaaQualityEdgeThresholdMin (probably used)
		0, // fxaaConsoleEdgeSharpness (not used)
		0, // fxaaConsoleEdgeThreshold (not used)
		0, // fxaaConsoleEdgeThresholdMin (not used)
		0 // fxaaConsole360ConstDir (not used)
	);
	result_texture[dispatchThreadID.xy] = fxaa_out;

	#else
	float4 light_color_out = in_texture[dispatchThreadID.xy];
	result_texture[dispatchThreadID.xy] = light_color_out;
	result_texture[dispatchThreadID.xy].w = 1.0;
	#endif
}
