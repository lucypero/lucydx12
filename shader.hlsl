#pragma pack_matrix(row_major)

struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uvs : TEXCOORD;
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
    float someValue;
};

PSInput VSMain(VSInput the_input) {
    PSInput result;
    result.position = mul(float4(the_input.position, 1.0f), wvp);
    result.frag_pos_world = float3(the_input.position);
    result.frag_normal = the_input.normal;
    result.uvs = the_input.uvs.xy;
    return result;
}

Texture2D<float4> myTexture : register(t1);
SamplerState mySampler : register(s0);

float4 PSMain(PSInput input) : SV_TARGET {
    float4 pixelColor = myTexture.Sample(mySampler, input.uvs);

    float3 norm = normalize(input.frag_normal);
    float3 light_dir = normalize(light_pos - input.frag_pos_world);

    float diff = max(dot(norm, light_dir), 0.0f);
    float3 diffuse = diff * light_int;

    float amb_val = 0.05;

    float3 ambient = float3(amb_val, amb_val, amb_val);

    // Specular calculation

    float specularStrength = 0.5;

    float3 viewDir = normalize(view_pos - input.frag_pos_world);
    float3 reflectDir = reflect(-light_dir, norm);  

    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    float3 specular = specularStrength * spec;   // * light color (we don't have that yet)


    float3 result = (ambient + diffuse + specular) * pixelColor.xyz;
    return float4(result, 1.0);

    // return pixelColor * float4(diffuse, 1.0f) + pixelColor * ambient;
    // return float4(1,0,0,1);
}
