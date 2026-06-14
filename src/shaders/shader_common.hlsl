#pragma once

/// Root Parameters
SamplerState mySampler : register(s0);
int cbv_index: register (b0); // index of my big CBV into the srv heap

struct MeshTransform
{
	float4x4 model; 
};

#include "src/shaders/gen/structs.gen.hlsl"

ConstantBuffer<DrawConstants> draw_constants : register(b1);
