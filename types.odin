// project-wide public types

package main

import "core:mem/virtual"
import "core:sys/windows"
import "core:time"
import dx "vendor:directx/d3d12"
import dxgi "vendor:directx/dxgi"
import sdl "vendor:sdl2"
import "core:math"
import "core:math/linalg"
import "base:runtime"
import dxc "vendor:directx/dxc"
import dxma "libs/odin-d3d12ma"

PROFILE :: #config(PROFILE, false)

NUM_RENDERTARGETS :: 2

TURNS_TO_RAD :: math.PI * 2

v2 :: linalg.Vector2f32
v3 :: linalg.Vector3f32
v4 :: linalg.Vector4f32

dxm :: matrix[4, 4]f32

DXResourcePool :: [dynamic]^dx.IUnknown

gbuffer_shader_filename :: "shader.hlsl"
lighting_shader_filename :: "lighting.hlsl"
ui_shader_filename :: "ui.hlsl"

// window dimensions
WINDOW_WIDTH :: i32(2000)
WINDOW_HEIGHT :: i32(1000)

GBUFFER_COUNT :: 3

TEXTURE_WHITE_INDEX :: TEXTURE_INDEX_BASE - 1
TEXTURE_INDEX_BASE :: 400

MODEL_FILEPATH_TEAPOT :: "models/teapot.glb"
// model_filepath :: "models/main_sponza/NewSponza_Main_glTF_003.gltf"
MODEL_FILEPATH_TEST_SCENE :: "models/test_scene.glb"
// model_filepath :: "models/main_sponza/sponza_blender.glb"

// no decals (ruins solid rendering)
MODEL_FILEPATH_BIG_SPOZA_NO_DECALS :: "models/main_sponza/sponza_blender_no_decals.glb"
MODEL_FILEPATH_SPONZA :: "C:/Users/Lucy/third_party/glTF-Sample-Models/2.0/Sponza/glTF/Sponza.gltf"
MODEL_FILEPATH_TOYCAR :: "C:/Users/Lucy/third_party/glTF-Sample-Models/2.0/ToyCar/glTF/ToyCar.gltf"
MODEL_FILEPATH_NORMAL_MAP_TEST :: "models/normal_map_test.glb"
MODEL_FILEPATH_SUZANNE :: "C:/Users/Lucy/third_party/glTF-Sample-Models/2.0/Suzanne/glTF/Suzanne.gltf"
MODEL_FILEPATH_FLIGHTHELMET :: "C:/Users/Lucy/third_party/glTF-Sample-Models/2.0/FlightHelmet/glTF/FlightHelmet.gltf"




VertexData :: struct {
	pos: v3,
	normal: v3,
	tangent: v4,
	uv: v2,
	uv_2: v2,
}

// Data associated with a vertex buffer
// this could be an instance buffer too. it's the same to dx12.
VertexBuffer :: struct {
	buffer: ^dx.IResource,
	vbv: dx.VERTEX_BUFFER_VIEW,
	vertex_count: u32, // vertex count or instance count
	buffer_size: u32,
	buffer_stride: u32,
}

GBufferUnit :: struct {
	res: ^dx.IResource,
	rtv: dx.CPU_DESCRIPTOR_HANDLE,
	format: dxgi.FORMAT,
}

GBuffer :: struct {
	gb_albedo: GBufferUnit,
	gb_normal: GBufferUnit,
	gb_ao_rough_metal: GBufferUnit,
	rtv_heap: ^dx.IDescriptorHeap,
}

HotSwapState :: struct {
	// TODO: store more data here so u don't have to pass the data around in the hotswap methods
	last_write_time: time.Time,
	pso_swap: ^dx.IPipelineState,

	// index in the queue array to free the resource
	// i use this to swap the pointer when the pso gets hot swapped
	pso_index: int,
}

MAX_GIZMOS :: 20

Context :: struct {
	// sdl stuff
	window: ^sdl.Window,

	// imgui stuff
	imgui_descriptor_heap: ^dx.IDescriptorHeap,
	imgui_allocator: DescriptorHeapAllocator,

	// core stuff
	device: ^dx.IDevice,
	factory: ^dxgi.IFactory4,
	
	// Graphics core resources
	
	queue: ^dx.ICommandQueue,
	command_allocator: ^dx.ICommandAllocator,
	cmdlist: ^dx.IGraphicsCommandList,
	
	// Other
	
	swapchain: ^dxgi.ISwapChain3,
	dxc_compiler: ^dxc.ICompiler3,
	pipeline_gbuffer: ^dx.IPipelineState,
	constant_buffer_map: rawptr, //maps to our test constant buffer
	gbuffer_pass_root_signature: ^dx.IRootSignature,
	constant_buffer: ^dx.IResource,
	dxma_allocator: ^dxma.Allocator,
	// descriptor heap for the render target view
	swapchain_rtv_descriptor_heap: ^dx.IDescriptorHeap,
	frame_index: u32,
	targets: [NUM_RENDERTARGETS]^dx.IResource, // render targets
	gbuffer: GBuffer,

	// lighting pass resources
	pipeline_lighting: ^dx.IPipelineState,
	lighting_pass_root_signature: ^dx.IRootSignature,

	// fence stuff (for waiting to render frame)
	fence: ^dx.IFence,
	fence_value: u64,
	fence_event: windows.HANDLE,

	// descriptor heap for ALL our resources
	cbv_srv_uav_heap: ^dx.IDescriptorHeap,
	descriptor_count : u32, // count for how many descriptors are in the srv heap

	// depth buffer
	depth_stencil_res: ^dx.IResource,
	descriptor_heap_dsv: ^dx.IDescriptorHeap,

	// hot swap shader state
	lighting_hotswap: HotSwapState,
	gbuffer_hotswap: HotSwapState, // todo this one (make helper functions for setting initial state and swapping code)
	
	// --- gizmos drawing ------
	
	pso_gizmos : ^dx.IPipelineState,
	rs_gizmos : ^dx.IRootSignature,
	
	// --- gizmos drawing end ----
}

ModelMatrixData :: struct {
	model_matrix: dxm,
}

// all meshes use the same index/vertex buffer.
// so we just have to store the offset and index count to render a specific mesh
Mesh :: struct {
	primitives: []Primitive,
}

Primitive :: struct {
	index_offset: u32,
	index_count: u32,
	material_index: u32
}

// texture id into the srv heap. and the uv id used to sample the texture
TextureUV :: struct {
	texture_id: u32,
	uv_id: u32, // what uv to use to sample the texture
}

Material :: struct {
	base_color: TextureUV,
	metallic_roughness: TextureUV,
	normal: TextureUV,
}

// constant buffer data
ConstantBufferData :: struct #align (256) {
	view: dxm,
	projection: dxm,
	inverse_view_proj: dxm,
	light_pos: v3,
	light_int: f32,
	view_pos: v3,
	time: f32,
}

// testing

Scene :: struct {
	nodes: []Node,
	root_nodes: []int,
	mesh_count: uint,
	meshes: []Mesh,
	allocator: virtual.Arena,
	
	// dx resources
	sb_model_matrices: ^dx.IResource,
	sb_materials: ^dx.IResource,
	
	vertex_buffer: ^dx.IResource,
	index_buffer: ^dx.IResource,
	vertex_buffer_view: dx.VERTEX_BUFFER_VIEW,
	index_buffer_view: dx.INDEX_BUFFER_VIEW,
	
	// gizmos (put this somewhere else later)
	// TODO
	vb_gizmos_instance_data: VertexBuffer,
	
	resource_pool: DXResourcePool,
	
	fence_value: u64, // (set after it is ready) fence value to wait on for all scene resources to be uploaded to the GPU
	ready_value: u64, // when u get a resource back from the upload thread with this value, u know the scene is ready. (cpu)
	is_ready: bool // ready to be rendered (on cpu side)
}

Node :: struct {
	name: string,
	transform_t: v3,
	transform_r: v4,
	transform_s: v3,
	children: []int,
	parent: int, // -1 for no parent (root node)
	mesh: int, // mesh index to render. -1 for no mesh
}

// struct that holds instance data, for an instance rendering example
InstanceData :: struct #align (256) {
	world_mat: dxm,
	color: v4,
}
