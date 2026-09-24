#pragma once
#pragma pack_matrix(column_major)
#include "shaders/gen/lucy2d-structs.gen.hlsl"

SamplerState g_sampler : register(s1); // nearest neighbor sampler
int cbv_index: register (b0); // index of my big CBV into the srv heap
