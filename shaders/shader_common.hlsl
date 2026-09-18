#pragma once

/// Root Parameters
SamplerState mySampler : register(s0);
// SamplerState sampler_shadowmap : register(s1);
SamplerComparisonState sampler_shadowmap : register(s1);
SamplerState sampler_linear : register(s2);

int cbv_index: register (b0); // index of my big CBV into the srv heap

struct MeshTransform
{
	float4x4 model; 
};

#include "src/shaders/gen/structs.gen.hlsl"

ConstantBuffer<DrawConstants> draw_constants : register(b1);
