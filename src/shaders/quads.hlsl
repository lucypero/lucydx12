#pragma pack_matrix(column_major)

// TODO auto-generate this struct from odin.
struct Sprite
{
	float2 pos;
	float2 size;
};

SamplerState g_sampler : register(s0);

struct GeneralConstants {
	uint sb_sprites_idx; // index of the sprite structured buffer into the resource heap
};

ConstantBuffer<GeneralConstants> g_draw_constants : register(b0);

struct VSOut {
	float4 pos   : SV_Position;
	float4 color : COLOR0;
};

VSOut VSMain(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
	StructuredBuffer<Sprite> sprites = ResourceDescriptorHeap[g_draw_constants.sb_sprites_idx];
	Sprite sprite = sprites[iid];
}

float4 PSMain() : SV_Target {

}
