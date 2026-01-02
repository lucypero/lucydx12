// this is the first pass that populates all the g-buffers.

#pragma pack_matrix(column_major)

struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uvs : TEXCOORD;
    // instance data
    float4 worldM0  : WORLDMATRIX0; // Per-instance data (Slot 1)
    float4 worldM1  : WORLDMATRIX1;
    float4 worldM2  : WORLDMATRIX2;
    float4 worldM3  : WORLDMATRIX3;
    float3 color : COLOR;
};

struct PSInput {
    float4 position : SV_POSITION;
    float3 frag_pos_world: POSITION;
    float3 frag_normal: NORMAL;
    float2 uvs : TEXCOORD0;
    float3 color: COLOR;
};

struct GeneralConstants {
    float4x4 view;
    float4x4 projection;
    float3 light_pos;
    float light_int;
    float3 view_pos;
    float time;
    bool place_texture;
};

struct MeshTransform
{
    float4x4 model; 
};

SamplerState mySampler : register(s0);

struct DrawConstants {
    uint mesh_index;
};

ConstantBuffer<DrawConstants> draw_constants : register(b1);

// cbv index is 3
// structured buffer index is %v 5

PSInput VSMain(VSInput the_input) {

	ConstantBuffer<GeneralConstants> general_constants = ResourceDescriptorHeap[3];
	StructuredBuffer<MeshTransform> mesh_transforms = ResourceDescriptorHeap[4];

    PSInput result;

    // use this for instanced drawing
    // float4x4 world_matrix = float4x4(the_input.worldM0, the_input.worldM1, the_input.worldM2, the_input.worldM3);
    // world_matrix = transpose(world_matrix);
    
    float4x4 world_matrix = mesh_transforms[draw_constants.mesh_index].model;

    float4 pos = float4(the_input.position, 1.0f);

    float4 world_position = mul(world_matrix, pos);
    // float4x4 world_position = mul(wvp, world_matrix);

    float4 view_position = mul(general_constants.view, world_position);

    // result.position = mul(pos, world_position);
    result.position = mul(general_constants.projection, view_position);

    result.frag_pos_world = world_position.xyz;
    
    // transforming normals by the normal matrix (a transformed world matrix)
    // this does not handle non-uniform scaling
    // TODO deal with that.
    result.frag_normal = mul((float3x3)world_matrix, the_input.normal);
    
    result.uvs = the_input.uvs.xy;
    result.color = the_input.color;
    return result;
}

struct PSOutput {
    // Target 0: Albedo Color (RGB) and Specular Intensity (A)
    // Common formats: DXGI_FORMAT_R8G8B8A8_UNORM
    float4 albedoSpecRT : SV_Target0; 

    // Target 1: World-Space Normals (RGB)
    // We use DXGI_FORMAT_R8G8B8A8_UNORM or similar. Normals are usually packed.
    float4 normalRT   : SV_Target1; 

    // Target 2: World-Space Position (XYZ) (or Depth/View-Space Position)
    // Common formats: DXGI_FORMAT_R16G16B16A16_FLOAT or DXGI_FORMAT_R32G32B32A32_FLOAT
    float4 positionRT : SV_Target2; 
};

PSOutput PSMain(PSInput input) {

	ConstantBuffer<GeneralConstants> general_constants = ResourceDescriptorHeap[3];
    PSOutput output;

    float4 pixelColor = float4(input.color, 1.0);
    // Texture2D<float4> someTexture = ResourceDescriptorHeap[9];
    // pixelColor = someTexture.Sample(mySampler, input.uvs);

    float3 norm = normalize(input.frag_normal);

    // writing to all gbuffers

    output.albedoSpecRT = pixelColor;
    
    // output.normalRT.rgb = norm;
    output.normalRT.rgb = (norm * 0.5f) + 0.5f;
    output.positionRT.rgb = input.frag_pos_world;

    output.normalRT.a = 1.0f;
    output.positionRT.a = 1.0f;

    // lighting: we won't use
    float3 light_dir = normalize(general_constants.light_pos - input.frag_pos_world);

    float diff = max(dot(norm, light_dir), 0.0f);
    float3 diffuse = diff * general_constants.light_int;

    float amb_val = 0.05;

    float3 ambient = float3(amb_val, amb_val, amb_val);

    // Specular calculation
    float specularStrength = 0.5;

    float3 viewDir = normalize(general_constants.view_pos - input.frag_pos_world);
    float3 reflectDir = reflect(light_dir, norm);  

    float3 spec_color = float3(1,1,1);

    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 128);
    float3 specular = specularStrength * spec * spec_color;

    float3 result = (ambient + diffuse + specular) * pixelColor.xyz;

    // return float4(result, 1.0);

    // end lighting 


    // return float4(1,0,0,1);

    return output;
}
