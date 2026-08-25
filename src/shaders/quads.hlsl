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
	float2 inv_screen; // 1.0 / (width, height)
};

int cbv_index: register (b0); // index of my big CBV into the srv heap

struct VSOut {
	float4 pos   : SV_Position;
	float4 color : COLOR0;
};

VSOut VSMain(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
	ConstantBuffer<GeneralConstants> general_constants = ResourceDescriptorHeap[cbv_index];
	StructuredBuffer<Sprite> sprites = ResourceDescriptorHeap[general_constants.sb_sprites_idx];
	Sprite sprite = sprites[iid];

	// make the corners and stuff.. do some linear transforms

	VSOut output;
	output.color = float4(1,1,1,1);
	output.pos.z = 1;
	output.pos.w = 1;

	switch (vid) {
	case 0:
		output.pos.xy = sprite.pos;
		break;
	case 1:
		output.pos.xy = sprite.pos + float2(0, sprite.size.y);
		break;
	case 2:
		output.pos.xy = sprite.pos + float2(sprite.size.x, sprite.size.y);
		break;
	case 3:
		output.pos.xy = sprite.pos + float2(sprite.size.x, 0);
		break;
	}

	// Converting to homogeneus coord space
	output.pos.xy = output.pos.xy * general_constants.inv_screen * float2(2, -2) + float2(-1, 1);

	return output;
}

float4 PSMain(VSOut input) : SV_Target {
	return float4(1,0,0,1);
}
