#pragma pack_matrix(row_major)

struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uvs : TEXCOORD;
    // instance data
    float4 worldM0  : WORLDMATRIX0; // Per-instance data (Slot 1)
    float4 worldM1  : WORLDMATRIX1;
    float4 worldM2  : WORLDMATRIX2;
    float4 worldM3  : WORLDMATRIX3;
};

struct PSInput {
    float4 position : SV_POSITION;
    float3 frag_pos_world: POSITION;
    float3 frag_normal: NORMAL;
    float2 uvs : TEXCOORD0;
};

cbuffer ConstantBuffer : register(b0) {
    float4x4 wvp;
    float3 light_pos;
    float light_int;
    float3 view_pos;
    float time;
    bool place_texture;
};

PSInput VSMain(VSInput the_input) {
    PSInput result;

    float4x4 world_matrix = float4x4(the_input.worldM0, the_input.worldM1, the_input.worldM2, the_input.worldM3);
    result.position = mul(float4(the_input.position, 1.0f), wvp * world_matrix);
    result.frag_pos_world = float3(the_input.position);
    result.frag_normal = the_input.normal;
    result.uvs = the_input.uvs.xy;

    return result;
}

Texture2D<float4> myTexture : register(t1);
SamplerState mySampler : register(s0);

float4 PSMain(PSInput input) : SV_TARGET {

    float4 pixelColor = float4(1.0, 1.0, 1.0, 1.0);

    if(place_texture) {
        pixelColor = myTexture.Sample(mySampler, input.uvs);
    }

    float3 norm = normalize(input.frag_normal);
    float3 light_dir = normalize(light_pos - input.frag_pos_world);

    float diff = max(dot(norm, light_dir), 0.0f);
    float3 diffuse = diff * light_int;

    float amb_val = 0.05;

    float3 ambient = float3(amb_val, amb_val, amb_val);

    // Specular calculation

    float specularStrength = 0.5;

    float3 viewDir = normalize(view_pos - input.frag_pos_world);
    float3 reflectDir = reflect(light_dir, norm);  

    float3 spec_color = float3(1,1,1);

    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 128);
    float3 specular = specularStrength * spec * spec_color;

    float3 result = (ambient + diffuse + specular) * pixelColor.xyz;
    return float4(result, 1.0);

    // return pixelColor * float4(diffuse, 1.0f) + pixelColor * ambient;
    // return float4(1,0,0,1);
}
