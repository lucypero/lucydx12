#pragma pack_matrix(column_major)

// TODO auto-generate this struct from odin.
struct Sprite
{
	float2 pos;
	float2 size;
	float4 color;
	int tex_idx;
};

SamplerState g_sampler : register(s1); // nearest neighbor sampler
int cbv_index: register (b0); // index of my big CBV into the srv heap

struct GeneralConstants {
	uint sb_sprites_idx; // index of the sprite structured buffer into the resource heap
	float2 inv_screen; // 1.0 / (width, height)
};

struct VSOut {
	float4 pos   : SV_Position;
	float4 color : COLOR0;
	float2 uvs : TEXTUREUV;
	nointerpolation int tex_idx : TEXTUREIDX;
};

VSOut VSMain(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
	ConstantBuffer<GeneralConstants> general_constants = ResourceDescriptorHeap[cbv_index];
	StructuredBuffer<Sprite> sprites = ResourceDescriptorHeap[general_constants.sb_sprites_idx];
	Sprite sprite = sprites[iid];

	// make the corners and stuff.. do some linear transforms

	VSOut output;
	output.color = sprite.color;
	output.pos.z = 1;
	output.pos.w = 1;
	output.tex_idx = sprite.tex_idx;

	switch (vid) {
	case 0: // bottom left
		output.pos.xy = sprite.pos;
		output.uvs = float2(0, 1);
		break;
	case 1: // top left
		output.pos.xy = sprite.pos + float2(0, sprite.size.y);
		output.uvs = float2(0, 0);
		break;
	case 2: // bottom right
		output.pos.xy = sprite.pos + float2(sprite.size.x, 0);
		output.uvs = float2(1, 1);
		break;
	case 3: // top right
		output.pos.xy = sprite.pos + float2(sprite.size.x, sprite.size.y);
		output.uvs = float2(1, 0);
		break;
	}

	// Converting to homogeneus coord space
	output.pos.xy = output.pos.xy * general_constants.inv_screen * 2 - 1;
	// output.pos.xy = output.pos.xy * general_constants.inv_screen * float2(2, 2) + float2(-1, -1);

	return output;
}

float4 PSMain(VSOut input) : SV_Target {

	float4 color = input.color;

	if(input.tex_idx > 0) {
		Texture2D<float4> tex = ResourceDescriptorHeap[input.tex_idx];
		color = tex.Sample(g_sampler, input.uvs);
	}

	return color;
}
