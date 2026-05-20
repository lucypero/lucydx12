package main

import "core:flags"
import "core:thread"
import "core:mem/virtual"
import "core:debug/trace"
import "core:reflect"
import "core:fmt"
import "core:mem"
import "core:slice"
import "core:os"
import "core:strings"
import "core:sys/windows"
import "core:time"
import dx "vendor:directx/d3d12"
import dxgi "vendor:directx/dxgi"
import sdl "vendor:sdl2"
import "core:c"
import "core:math"
import "core:math/linalg"
import "base:runtime"
import "core:math/rand"
import "core:sync"
import dxc "vendor:directx/dxc"
import "core:prof/spall"
import dxma "../libs/odin-d3d12ma"

// imgui
import im "../libs/odin-imgui"
// imgui sdl2 implementation
import "../libs/odin-imgui/imgui_impl_sdl2"
// imgui dx12 implementation
import "../libs/odin-imgui/imgui_impl_dx12"

PROFILE :: #config(PROFILE, false)

NUM_RENDERTARGETS :: 2

TURNS_TO_RAD :: math.PI * 2

v2 :: linalg.Vector2f32
v3 :: linalg.Vector3f32
v4 :: linalg.Vector4f32

dxm :: matrix[4, 4]f32

DXResourcePool :: [dynamic]^dx.IUnknown

gbuffer_shader_filename :: "src/shaders/shader.hlsl"
lighting_shader_filename :: "src/shaders/lighting.hlsl"
ui_shader_filename :: "src/shaders/ui.hlsl"
text_shader_filename :: "src/shaders/text.hlsl"

// window dimensions
WINDOW_WIDTH :: 2000
WINDOW_HEIGHT :: 1000

GBUFFER_COUNT :: len(GBufferUnitName)

TEXTURE_WHITE_INDEX :: TEXTURE_INDEX_BASE - 1
TEXTURE_INDEX_BASE :: 400

MODEL_FILEPATH_TEAPOT :: "models/teapot.glb"
// model_filepath :: "models/main_sponza/NewSponza_Main_glTF_003.gltf"
MODEL_FILEPATH_TEST_SCENE :: "models/test_scene.glb"
// model_filepath :: "models/main_sponza/sponza_blender.glb"

GLTF_SAMPLES_DIR :: "models/glTF-Sample-Models/2.0"
// no decals (ruins solid rendering)
MODEL_FILEPATH_BIG_SPOZA_NO_DECALS :: "models/main_sponza/sponza_blender_no_decals.glb"
MODEL_FILEPATH_SPONZA :: GLTF_SAMPLES_DIR + "/Sponza/glTF/Sponza.gltf"
MODEL_FILEPATH_TOYCAR :: GLTF_SAMPLES_DIR + "/ToyCar/glTF/ToyCar.gltf"
MODEL_FILEPATH_NORMAL_MAP_TEST :: "models/normal_map_test.glb"
MODEL_FILEPATH_SUZANNE :: GLTF_SAMPLES_DIR + "/Suzanne/glTF/Suzanne.gltf"
MODEL_FILEPATH_FLIGHTHELMET :: GLTF_SAMPLES_DIR + "/FlightHelmet/glTF/FlightHelmet.gltf"

VertexData :: struct {
	pos: v3 `POSITION`,
	normal: v3 `NORMAL`,
	tangent: v4 `TANGENT`,
	uv: v2 `TEXCOORD`,
	uv_2: v2 `TEXCOORD_SECOND_UV`,
}

// Data associated with a vertex buffer
// this could be an instance buffer too. it's the same to dx12.
VertexBuffer :: struct {
	buffer: ^dx.IResource,
	gpu_pointer: rawptr, // only valid if it's on upload heap
	vbv: dx.VERTEX_BUFFER_VIEW,
	vertex_count: u32, // vertex count or instance count
	buffer_size: u32,
	buffer_stride: u32,
}

ConstantBufferUpload :: struct {
	buffer: ^dx.IResource,
	gpu_pointer: rawptr, // only valid if it's on upload heap
	buffer_size: u32,
	srv_index: int // index as constant buffer view in the uber heap
}

GBufferUnit :: struct {
	res: ^dx.IResource,
	rtv: dx.CPU_DESCRIPTOR_HANDLE,
	format: dxgi.FORMAT,
}

// NEVER CHANGE THE ORDER OF THESE.
// They are used as render targets in a shader. that's mainly why.
GBufferUnitName :: enum {
	Albedo,
	Normal,
	AO_Rough_Metal,
}

GBuffer :: struct {
	gbuffers: [GBufferUnitName]Texture,
}

MAX_GIZMOS :: 20

Context :: struct {
	/// SDL stuff
	window: ^sdl.Window,

	/// imgui stuff
	imgui_descriptor_heap: ^dx.IDescriptorHeap,
	imgui_allocator: DescriptorHeapAllocator,

	/// fence stuff (for waiting to render frame)
	fence: ^dx.IFence,
	fence_value: u64,
	fence_event: windows.HANDLE,

	/// descriptor heap for ALL our resources
	cbv_srv_uav_heap: UberDescriptorHeap,
	dsv_heap: UberDescriptorHeap,
	rtv_heap: UberDescriptorHeap,

	/// Other
	device: ^dx.IDevice,
	factory: ^dxgi.IFactory4,
	queue: ^dx.ICommandQueue,
	command_allocator: ^dx.ICommandAllocator,
	cmdlist: ^dx.IGraphicsCommandList,
	swapchain: ^dxgi.ISwapChain3,
	dxc_compiler: ^dxc.ICompiler3,
	dxma_allocator: ^dxma.Allocator,
	// descriptor heap for the render target view
	swapchain_rtv_descriptor_heap: ^dx.IDescriptorHeap,
	targets: [NUM_RENDERTARGETS]^dx.IResource, // render targets
	frame_index: int,
	depth_texture: Texture,
	gbuffer: GBuffer,
	root_signatures: [RootSignatureChoice]^dx.IRootSignature,
	psos: [PSOName]PSO,
	constant_buffer: ConstantBufferUpload,

	/// Shadowmap
	tx_shadowmap: Texture

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
	current_scene_materials_idx: u32,
	current_scene_mesh_transforms_idx: u32,
}

// testing
SceneStatus :: enum {Free, Loading, Ready, QueuedForDeletion}

Scene :: struct {
	path: string, // set this before scheduling upload
	nodes: []Node,
	root_nodes: []int,
	mesh_count: uint,
	uv_sphere_mesh: Mesh,
	meshes: []Mesh,
	allocator: virtual.Arena,

	material_srv_index: int,
	model_matrices_srv_index: int,

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

	// NOTE: This is a very thread-sensitive field. check that all modifications follow the right constraints.
	status: SceneStatus
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
	world_mat: dxm `WORLDMATRIX`,
	color: v4 `COLOR`
}

// ---- GLOBAL STATE ----

@(private="package") g_is_app_shutting_down: bool
@(private="package") g_dx_context: Context
@(private="package") g_resources_longterm: DXResourcePool
@(private="package") g_scenes: [3]Scene

// private global staet

g_global_trace_ctx: trace.Context
g_frame_dt : f64 = 0.2 // in ms
g_mesh_drawn_count: int = 0
g_start_time: time.Time
g_light_pos: v3
g_light_draw_gizmos: bool
g_light_int: f32
g_the_time_sec: f32
g_exit_app: bool

// Profiling stuff

when PROFILE {
@(private="package")
g_spall_ctx: spall.Context
@(thread_local)
@(private="package")
g_spall_buffer: spall.Buffer
}

// vertex buffer
Gizmos_Vertex_IA :: struct {
	pos: v3 `POSITION`,
}

// All our PSO's
PSOName :: enum {
	GBuffer_Pass,
	Lighting_Pass,
	Shadowmap,
	Gizmos,
	Text
}

// ----- //// GLOBAL STATE ------

cb_update :: proc() {

	// ticking cbv time value
	thetime := time.diff(g_start_time, time.now())
	g_the_time_sec = f32(thetime) / f32(time.Second)
	// if the_time_sec > 1 {
	// 	start_time = time.now()
	// }

	// sending constant buffer data
	view, projection := get_view_projection(g_cur_cam)

	active_scene, scene_is_active := get_first_active_scene()

	cbv_data := ConstantBufferData {
		view = view,
		projection = projection,
		inverse_view_proj = linalg.inverse(projection * view),
		light_pos = g_light_pos,
		light_int = g_light_int,
		view_pos = g_cur_cam.pos,
		time = g_the_time_sec,
		current_scene_materials_idx = scene_is_active ? cast(u32)active_scene.material_srv_index : 0,
		current_scene_mesh_transforms_idx = scene_is_active ? cast(u32)active_scene.model_matrices_srv_index : 0,
	}

	// sending data to the cpu mapped memory that the gpu can read
	// copy_to_buffer_already_mapped(g_dx_context.constant_buffer.gpu_pointer, slice.to_bytes([]ConstantBufferData{cbv_data}))
	copy_to_buffer_already_mapped_value(g_dx_context.constant_buffer.gpu_pointer, &cbv_data)
}

// initializes app data in Context struct
context_init :: proc(con: ^Context) {
	g_cur_cam = camera_init()
	g_light_pos = v3{0,2,0}
	g_light_draw_gizmos = true
	g_light_int = 1
}

byte_to_mb :: proc(bytecount: int) -> f32 {
	return cast(f32)bytecount / cast(f32)mem.Megabyte
}

tracking_allocator_report :: proc(allocator_name: string, track: mem.Tracking_Allocator, report_leaks_and_double_frees: bool) {
	lprintfln("=== %v - Memory Report ===", allocator_name)
	lprintfln("Peak Memory Used: %v MB", byte_to_mb(cast(int)track.peak_memory_allocated))
	lprintfln("Total Memory Allocated: %v MB", byte_to_mb(cast(int)track.total_memory_allocated))
	lprintfln("Total Memory Freed: %v MB", byte_to_mb(cast(int)track.total_free_count))

	if !report_leaks_and_double_frees do return

	// Check for leaks
	if len(track.allocation_map) > 1 { // skipping 1 because 1 is the tracker itself
		lprintfln("\n=== %v - MEMORY LEAKS (%v) ===", allocator_name, len(track.allocation_map))
		for _, entry in track.allocation_map {
			lprintfln("- %v bytes leaked at %v", entry.size, entry.location)
		}
	}

	// Check for bad frees (double frees, freeing wrong pointers)
	if len(track.bad_free_array) > 0 {
		lprintfln("\n=== %v, BAD FREES (%v) ===", allocator_name, len(track.bad_free_array))
		for entry in track.bad_free_array {
			lprintfln("- Bad free at %v", entry.location)
		}
	}
}

@(private="package")
g_track : mem.Tracking_Allocator

@(private="package")
main :: proc() {

	// set up memory
	{
		temp_arena : virtual.Arena
		temp_allocator : mem.Allocator

		alloc_err := virtual.arena_init_growing(&temp_arena, mem.Megabyte)
		assert(alloc_err == .None)
		temp_allocator = virtual.arena_allocator(&temp_arena)
		context.temp_allocator = temp_allocator
	}

	when ODIN_DEBUG {
	lprintln("Tracking Allocations...")
	mem.tracking_allocator_init(&g_track, context.allocator)
	context.allocator = mem.tracking_allocator(&g_track)

	temp_track: mem.Tracking_Allocator
	mem.tracking_allocator_init(&temp_track, context.temp_allocator)
	context.temp_allocator = mem.tracking_allocator(&temp_track)

	defer {
		tracking_allocator_report("context.allocator", g_track, true)
		// tracking_allocator_report("context.temp_allocator", temp_track, false)
		ar := cast(^virtual.Arena)context.temp_allocator.data
		arena_report("temp arena", ar^)
		mem.tracking_allocator_destroy(&g_track)
	}
	}

	// /set up memory

	// setting up upload thread
	upload_thread := thread.create_and_start(upload_thread_start)


	// setting up long term resource pool

	g_resources_longterm = make([dynamic]^dx.IUnknown)
	defer delete(g_resources_longterm)

	// destroy stray meshes (gizmo sphere)
	// (it's now in g_scene)

	trace.init(&g_global_trace_ctx)
	defer trace.destroy(&g_global_trace_ctx)

	ct := &g_dx_context

	// setting up profiling
	when PROFILE {
	SPALL_FILE :: "trace_test.spall"
	g_spall_ctx = spall.context_create(SPALL_FILE)
	defer {
		lprintfln("Spall profiling data written to: " + SPALL_FILE)
		spall.context_destroy(&g_spall_ctx)
	}

	buffer_backing := make([]u8, spall.BUFFER_DEFAULT_SIZE)
	defer delete(buffer_backing)

	g_spall_buffer = spall.buffer_create(buffer_backing, u32(sync.current_thread_id()))
	defer spall.buffer_destroy(&g_spall_ctx, &g_spall_buffer)

	spall.SCOPED_EVENT(&g_spall_ctx, &g_spall_buffer, #procedure)
	}

	// Init SDL and create window
	if err := sdl.Init(sdl.InitFlags{.TIMER, .AUDIO, .VIDEO, .EVENTS}); err != 0 {
		fmt.eprintln(err)
		return
	}

	defer sdl.Quit()
	ct.window = sdl.CreateWindow(
		"lucydx12",
		sdl.WINDOWPOS_UNDEFINED,
		sdl.WINDOWPOS_UNDEFINED,
		WINDOW_WIDTH,
		WINDOW_HEIGHT,
		{.ALLOW_HIGHDPI, .SHOWN, .RESIZABLE},
	)

	if ct.window == nil {
		fmt.eprintln(sdl.GetError())
		return
	}

	defer sdl.DestroyWindow(ct.window)


	init_dx()
	init_dx_user()
	context_init(ct)

	g_start_time = time.now()
	do_main_loop()
	g_is_app_shutting_down = true

	// cleanup
	{
		imgui_destoy()

		thread.destroy(upload_thread)

		// TODO destroy scenes ( wait for all gpu to be done) (actually we already are.)
		// scene_destroy(&g_scene)

		for &scene in g_scenes {
			st := scene_status_load(&scene.status)
			if (st == .Ready || st == .QueuedForDeletion) do scene_destroy(&scene)
		}

		#reverse for &i in g_resources_longterm {
			i->Release()
		}

		// this does nothing
		sdl.DestroyWindow(ct.window)

		when ODIN_DEBUG {

		debug_device: ^dx.IDebugDevice2
		ct.device->QueryInterface(dx.IDebugDevice2_UUID, (^rawptr)(&debug_device))
		// Finally, release the device (it is not in any pool)
		// The device will be freed after we release the debug device
		ct.device->Release()
		debug_device->ReportLiveDeviceObjects({.DETAIL, .IGNORE_INTERNAL})
		debug_device->Release()

		// DXGI report
		dxgi_debug: ^dxgi.IDebug1
		dxgi.DXGIGetDebugInterface1(0, dxgi.IDebug1_UUID, (^rawptr)(&dxgi_debug))
		dxgi_debug->ReportLiveObjects(dxgi.DEBUG_ALL, {})
		}
	}
}

// inits all basic dx resources.
init_dx :: proc() {
	hr: dx.HRESULT
	ct := &g_dx_context

	// Init DXGI factory. DXGI is the link between the window and DirectX
	factory: ^dxgi.IFactory4

	{
		flags: dxgi.CREATE_FACTORY

		when ODIN_DEBUG {
		flags += {.DEBUG}
		}

		hr = dxgi.CreateDXGIFactory2(flags, dxgi.IFactory4_UUID, cast(^rawptr)&factory)
		check(hr, "Failed creating factory")
		append(&g_resources_longterm, factory)
	}

	ct.factory = factory

	// Find the DXGI adapter (GPU)
	adapter: ^dxgi.IAdapter1
	error_not_found := dxgi.HRESULT(-142213123)

	// Debug layer
	when ODIN_DEBUG {
	debug_controller: ^dx.IDebug
	// continue here
	hr = dx.GetDebugInterface(dx.IDebug_UUID, (^rawptr)(&debug_controller))
	check(hr, "failed getting debug interface")

	debug_controller->EnableDebugLayer()
	debug_controller->Release()
	}

	for i: u32 = 0; factory->EnumAdapters1(i, &adapter) != error_not_found; i += 1 {
		desc: dxgi.ADAPTER_DESC1
		adapter->GetDesc1(&desc)
		append(&g_resources_longterm, adapter)
		if .SOFTWARE in desc.Flags {
			continue
		}

		device: ^dx.IDevice
		hr = dx.CreateDevice((^dxgi.IUnknown)(adapter), ._12_0, dx.IDevice_UUID, (^rawptr)(&device))

		if hr >= 0 {
			ct.device = device
			break
		} else {
			fmt.eprintfln("Failed to create device, err: %X", hr) // -2147467262
			// E_NOINTERFACE
			// no such interface supported
			return
		}
	}

	if adapter == nil {
		fmt.eprintln("Could not find hardware adapter")
		return
	}

	// set up logging callback
	when ODIN_DEBUG {
	info_queue: ^dx.IInfoQueue1
	ct.device->QueryInterface(dx.IInfoQueue1_UUID, (^rawptr)(&info_queue))
	cb_cookie: u32
	hr = info_queue->RegisterMessageCallback(dx_log_callback, {.IGNORE_FILTERS}, nil, &cb_cookie)
	info_queue->SetMuteDebugOutput(true)
	check(hr, "failed to register")
	info_queue->Release()
	}

	ct.dxc_compiler = dxc_init()

	// create dxma allocator
	{
		allocator_desc := dxma.ALLOCATOR_DESC {
			pDevice = ct.device,
			pAdapter = adapter,
			Flags = .NONE
		}

		check(dxma.CreateAllocator(&allocator_desc, &ct.dxma_allocator))
		append(&g_resources_longterm, cast(^dxgi.IUnknown)ct.dxma_allocator)
	}

	// Create command queue and allocator and list
	{
		check(ct.device->CreateCommandQueue(&{Type = .DIRECT}, dx.ICommandQueue_UUID, (^rawptr)(&ct.queue)))
		append(&g_resources_longterm, ct.queue)

		// The command allocator is used to create the commandlist that is used to tell the GPU what to draw
		check(ct.device->CreateCommandAllocator(.DIRECT, dx.ICommandAllocator_UUID, (^rawptr)(&ct.command_allocator)))
		append(&g_resources_longterm, ct.command_allocator)

		check(ct.device->CreateCommandList(
			0,
			.DIRECT,
			ct.command_allocator,
			nil,
			dx.ICommandList_UUID,
			(^rawptr)(&ct.cmdlist),
		))
		append(&g_resources_longterm, ct.cmdlist)
	}

	dx_upload_init()

	// Creating all uber descriptor heaps. So far, SRV and DSV uber heaps.
	{
		ct.cbv_srv_uav_heap = uber_heap_create(.CBV_SRV_UAV, &g_resources_longterm)
		ct.dsv_heap = uber_heap_create(.DSV, &g_resources_longterm)
		ct.rtv_heap = uber_heap_create(.RTV, &g_resources_longterm)
	}

	// Create the swapchain, it's the thing that contains render targets that we draw into.
	//  It has 2 render targets (NUM_RENDERTARGETS), giving us double buffering.
	ct.swapchain = create_swapchain(ct.factory, ct.queue, ct.window)

	ct.frame_index = cast(int)ct.swapchain->GetCurrentBackBufferIndex()

	// Create swapchain rtv heap
	{
		desc := dx.DESCRIPTOR_HEAP_DESC {
			NumDescriptors = NUM_RENDERTARGETS,
			Type = .RTV,
			Flags = {},
		}

		hr = ct.device->CreateDescriptorHeap(
			&desc,
			dx.IDescriptorHeap_UUID,
			(^rawptr)(&ct.swapchain_rtv_descriptor_heap),
		)
		check(hr, "Failed creating descriptor heap")
		ct.swapchain_rtv_descriptor_heap->SetName("lucy's swapchain RTV descriptor heap")
		append(&g_resources_longterm, ct.swapchain_rtv_descriptor_heap)
	}

	// Fetch the two render targets from the swapchain
	{
		rtv_descriptor_size: u32 = ct.device->GetDescriptorHandleIncrementSize(.RTV)
		rtv_descriptor_handle: dx.CPU_DESCRIPTOR_HANDLE
		ct.swapchain_rtv_descriptor_heap->GetCPUDescriptorHandleForHeapStart(&rtv_descriptor_handle)

		for i: u32 = 0; i < NUM_RENDERTARGETS; i += 1 {
			hr = ct.swapchain->GetBuffer(i, dx.IResource_UUID, (^rawptr)(&ct.targets[i]))
			ct.targets[i]->Release()
			check(hr, "Failed getting render target")
			ct.device->CreateRenderTargetView(ct.targets[i], nil, rtv_descriptor_handle)
			rtv_descriptor_handle.ptr += uint(rtv_descriptor_size)
		}
	}
}

create_root_signatures :: proc() {

	ct := &g_dx_context
	hr : dx.HRESULT

	root_parameters_len :: 2

	root_parameters: [root_parameters_len]dx.ROOT_PARAMETER

	root_parameters[0] = {
		ParameterType = .CBV,
		Descriptor = {ShaderRegister = 0, RegisterSpace = 0},
		ShaderVisibility = .ALL, // vertex, pixel, or both (all)
	}

	root_parameters[1] = {
		ParameterType = ._32BIT_CONSTANTS,
		Constants = {ShaderRegister = 1, RegisterSpace = 0, Num32BitValues = 2},
		ShaderVisibility = .ALL
	}

	sampler_desc := dx.STATIC_SAMPLER_DESC {
		Filter = .ANISOTROPIC,
		AddressU = .WRAP,
		AddressV = .WRAP,
		AddressW = .WRAP,
		MipLODBias = 0.0,
		MaxAnisotropy = 16,
		ComparisonFunc = .NEVER,
		BorderColor = .OPAQUE_BLACK,
		MinLOD = 0.0,
		MaxLOD = dx.FLOAT32_MAX,
		ShaderRegister = 0,
		RegisterSpace = 0,
		ShaderVisibility = .PIXEL,
	}

	desc := dx.VERSIONED_ROOT_SIGNATURE_DESC {
		Version = ._1_0,
		Desc_1_0 = {
			NumParameters = root_parameters_len,
			pParameters = &root_parameters[0],
			NumStaticSamplers = 1,
			pStaticSamplers = &sampler_desc,
		},
	}

	// BINDLESS MODE: ACTIVATED!!!!!
	desc.Desc_1_0.Flags = {.CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED, .ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT}

	serialized_desc: ^dx.IBlob
	hr = dx.SerializeVersionedRootSignature(&desc, &serialized_desc, nil)
	check(hr, "Failed to serialize root signature")
	hr = ct.device->CreateRootSignature(
		0,
		serialized_desc->GetBufferPointer(),
		serialized_desc->GetBufferSize(),
		dx.IRootSignature_UUID,
		(^rawptr)(&ct.root_signatures[.Standard]),
	)
	check(hr, "Failed creating root signature")
	append(&g_resources_longterm, ct.root_signatures[.Standard])
	serialized_desc->Release()

	// creating other root signatrues here when needed
}

init_dx_user :: proc() {
	ct := &g_dx_context
	hr : dx.HRESULT

	// Creating all root signatures
	create_root_signatures()

	// Creating G-Buffer textures and RTV's
	ct.gbuffer = create_gbuffer()

	// constant buffer
	ct.constant_buffer = create_constant_buffer_upload(size_of(ConstantBufferData), &g_resources_longterm, name = "general constants cbv")

	// shadowmap setup
	{
		// creating shadowmap texture (DSV and then SRV)
		// createte

		ct.tx_shadowmap = texture_create(nil, 1024, 1024, .D32_FLOAT,
			&g_resources_longterm, {.DSV, .SRV}, texture_name = "shadowmap")

		ct.psos[.Shadowmap] = pso_create("src/shaders/shadowmap.hlsl", PSOParameters {
			vertex_input = VertexData,
			blend_state = .Off,
			enable_depth = true,
			rtv_count = 0,
		}, pso_name = "Shadowmap pso")
	}


	gbuffer_rtv_formats := [8]dxgi.FORMAT {
		0 ..= 7 = .UNKNOWN,
	}

	for g_buffer, i in g_dx_context.gbuffer.gbuffers {
		gbuffer_rtv_formats[i] = g_buffer.format
	}

	ct.psos[.GBuffer_Pass] = pso_create(gbuffer_shader_filename, PSOParameters {
		vertex_input = VertexData,
		blend_state = .Off,
		enable_depth = true,
		rtv_count = GBUFFER_COUNT,
		rtv_formats = gbuffer_rtv_formats,
		// true because we flipped positions and normals in gltf to convert between coord systems.
		front_counter_clockwise = true,
	}, pso_name = "geometry pass PSO")

	ct.psos[.Lighting_Pass] = pso_create(lighting_shader_filename, PSOParameters {
		vertex_input = struct{},
		blend_state = .Off,
		enable_depth = false,
		rtv_count = 1,
		rtv_formats = {0 = .R8G8B8A8_UNORM, 1 ..=7 = .UNKNOWN},
	}, pso_name = "lighting pass PSO")

	ct.psos[.Gizmos] = pso_create(ui_shader_filename, PSOParameters {
		vertex_input = Gizmos_Vertex_IA,
		instance_vertex_input = typeid_of(InstanceData),
		blend_state = .Off,
		enable_depth = true,
		fill_mode = .Wireframe,
		rtv_count = 1,
		rtv_formats = {0 = .R8G8B8A8_UNORM, 1 ..=7 = .UNKNOWN},
	}, pso_name = "Gizmos PSO")

	// hr = ct.command_allocator->Reset(
	// hr = ct.cmdlist->Reset(ct.command_allocator, nil)
	create_depth_buffer()

	// TODO: delete this?
	close_and_execute_cmdlist()

	imgui_init()

	load_white_texture()

	SCENE_MAX_LEN :: 3

	// initting g_scenes
	for &scene in g_scenes {
		scene.allocator = arena_new()
	}

	scene_schedule_load(&g_scenes[0], MODEL_FILEPATH_SPONZA)

	// This fence is used to wait for frames to finish
	{
		hr = ct.device->CreateFence(ct.fence_value, {}, dx.IFence_UUID, (^rawptr)(&ct.fence))
		check(hr, "Failed to create fence")
		append(&g_resources_longterm, ct.fence)
		ct.fence_value += 1
		manual_reset: windows.BOOL = false
		initial_state: windows.BOOL = false
		ct.fence_event = windows.CreateEventW(nil, manual_reset, initial_state, nil)
		if ct.fence_event == nil {
			lprintln("Failed to create fence event")
			return
		}
	}
}

do_main_loop :: proc() {

	last_time := time.now()

	main_loop: for {
		when PROFILE do spall.SCOPED_EVENT(&g_spall_ctx, &g_spall_buffer, name = "main loop")

		for e: sdl.Event; sdl.PollEvent(&e); {

			imgui_impl_sdl2.ProcessEvent(&e)

			#partial switch e.type {
			case .QUIT:
				break main_loop
			case .WINDOWEVENT:
				if e.window.event == .CLOSE {
					break main_loop
				}
			}
		}

		imgui_impl_dx12.NewFrame()
		imgui_impl_sdl2.NewFrame()
		im.NewFrame()
		update()
		im.End()
		im.Render()
		render()
		free_all(context.temp_allocator)

		new_time := time.now()
		dur := time.diff(last_time, new_time)
		g_frame_dt = time.duration_milliseconds(dur)
		last_time = new_time

		if g_exit_app {
			break main_loop
		}
	}
}

create_swapchain :: proc(
	factory: ^dxgi.IFactory4,
	queue: ^dx.ICommandQueue,
	window: ^sdl.Window,
) -> (
	swapchain: ^dxgi.ISwapChain3,
) {

	// Get the window handle from SDL
	window_info: sdl.SysWMinfo
	sdl.GetWindowWMInfo(window, &window_info)
	window_handle := dxgi.HWND(window_info.info.win.window)


	desc := dxgi.SWAP_CHAIN_DESC1 {
		Width = u32(WINDOW_WIDTH),
		Height = u32(WINDOW_HEIGHT),
		Format = .R8G8B8A8_UNORM,
		SampleDesc = {Count = 1, Quality = 0},
		BufferUsage = {.RENDER_TARGET_OUTPUT},
		BufferCount = NUM_RENDERTARGETS,
		Scaling = .NONE,
		SwapEffect = .FLIP_DISCARD,
		AlphaMode = .UNSPECIFIED,
	}

	hr := factory->CreateSwapChainForHwnd(
		(^dxgi.IUnknown)(queue),
		window_handle,
		&desc,
		nil,
		nil,
		(^^dxgi.ISwapChain1)(&swapchain),
	)
	check(hr, "Failed to create swap chain")
	append(&g_resources_longterm, swapchain)

	return
}




dx_log_callback :: proc "c" (
	category: dx.MESSAGE_CATEGORY,
	severity: dx.MESSAGE_SEVERITY,
	id: dx.MESSAGE_ID,
	description: cstring,
	ctx: rawptr,
) {
	context = runtime.default_context()

	// Filtering by severity

	#partial switch severity {
	case .CORRUPTION, .ERROR, .WARNING:
	case:
		return
	}

	msg := string(description)

	// ignore if it tells me the device is live
	if id == .LIVE_DEVICE do return

	severity_string, _ := reflect.enum_name_from_value(severity)
	cat, _ := reflect.enum_name_from_value(category)

	lprintfln("%v: (%v) %v", severity_string, cat, msg)

	// printing stack trace
	if !trace.in_resolve(&g_global_trace_ctx) {
		buf: [64]trace.Frame
		max_frames_display :: 3
		frames := trace.frames(&g_global_trace_ctx, 1, buf[:])

		// filtering by frames where we actually have info
		real_counter := 0

		for f in frames {
			fl := trace.resolve(&g_global_trace_ctx, f, context.temp_allocator)
			if fl.loc.file_path == "" && fl.loc.line == 0 do continue
			if real_counter == 0 do lprintfln("At:")
			real_counter += 1
			if real_counter <= max_frames_display do lprintfln("--- %v - Frame %v", fl.loc, real_counter)
		}
	}
}

update :: proc() {

	c := &g_dx_context

	sdl.PumpEvents()
	keyboard := sdl.GetKeyboardStateAsSlice()

	if keyboard[sdl.Scancode.ESCAPE] == 1 {
		g_exit_app = true
	}

	for &pso in c.psos {
		pso_hotswap_watch(&pso)
	}

	// add all the others
	// TODO: just put all psos on an array.

	do_imgui_ui()
	camera_tick(keyboard)
}

do_imgui_ui :: proc() {

	im.Begin("lucydx12")

	im.DragFloat3("light pos", &g_light_pos, 0.1, -5000, 5000)
	im.DragFloat("light intensity", &g_light_int, 0.1, 0, 20)
	im.Checkbox("draw light gizmos", &g_light_draw_gizmos)
	im.DragFloat("cam speed", &g_cur_cam.speed, 0.0001, 0, 20)
	im.DragFloat("cam cruise speed", &g_cur_cam.cruising_speed, 0.0001, 0, 20)

	// Drawing delta time
	{
		sb := strings.builder_make_len_cap(0, 30, context.temp_allocator)
		fmt.sbprintfln(&sb, "DT: %.2f", g_frame_dt)
		dt_cstring := strings.to_cstring(&sb)
		im.Text(dt_cstring)
	}

	// Drawing cam position
	{
		sb := strings.builder_make_len_cap(0, 30, context.temp_allocator)
		fmt.sbprintfln(&sb, "cam position: %.2v", g_cur_cam.pos)
		dt_cstring := strings.to_cstring(&sb)
		im.Text(dt_cstring)
	}

	// const char* items[] = { "AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF", "GGGG", "HHHH", "IIIIIII", "JJJJ", "KKKKKKK" };
	//            static int item_current = 0;
	//            ImGui::Combo("combo", &item_current, items, IM_ARRAYSIZE(items));

	@static current_selected : c.int = 0
	items := [?]cstring{"sponza", "something else"}
	new_selected: c.int = current_selected
	im.ComboChar("scene", &new_selected, raw_data(&items), len(items))

	if current_selected != new_selected {
		current_selected = new_selected

		switch current_selected {
		case 0:
			scene_swap(MODEL_FILEPATH_SPONZA)
		case 1:
			scene_swap(MODEL_FILEPATH_FLIGHTHELMET)
		}
	}

	if im.Button("switch scenes") {
		current_selected = current_selected == 0 ? 1 : 0

		switch current_selected {
		case 0:
			scene_swap(MODEL_FILEPATH_SPONZA)
		case 1:
			scene_swap(MODEL_FILEPATH_FLIGHTHELMET)
		}
	}

	// im.ShowDemoWindow()
}


render :: proc() {
	ct := &g_dx_context
	hr: dx.HRESULT

	cb_update()

	g_mesh_drawn_count = 0

	ct.cmdlist->Reset(ct.command_allocator, nil)

	pso_gbuffer_render()
	pso_lighting_render()
	if g_light_draw_gizmos do pso_gizmos_render()

	render_imgui()

	// Cannot draw after this point!!

	// Transitioning the render target to "Present" state
	transition_resource(g_dx_context.targets[g_dx_context.frame_index],
		ct.cmdlist, {.RENDER_TARGET}, dx.RESOURCE_STATE_PRESENT, subresource = dx.RESOURCE_BARRIER_ALL_SUBRESOURCES)

	ct.cmdlist->Close()

	// execute
	cmdlists := [?]^dx.IGraphicsCommandList{ct.cmdlist}
	g_dx_context.queue->ExecuteCommandLists(len(cmdlists), (^^dx.ICommandList)(&cmdlists[0]))

	// present
	{
		when PROFILE do spall.SCOPED_EVENT(&g_spall_ctx, &g_spall_buffer, name = "Present")
		flags: dxgi.PRESENT
		params: dxgi.PRESENT_PARAMETERS
		hr = g_dx_context.swapchain->Present1(1, flags, &params)
		check(hr, "Present failed")
	}

	// wait for frame to finish
	{
		when PROFILE do spall.SCOPED_EVENT(&g_spall_ctx, &g_spall_buffer, name = "v-sync wait")

		current_fence_value := g_dx_context.fence_value

		hr = g_dx_context.queue->Signal(g_dx_context.fence, current_fence_value)
		check(hr, "Failed to signal fence")

		g_dx_context.fence_value += 1
		completed := g_dx_context.fence->GetCompletedValue()

		if completed < current_fence_value {
			hr = g_dx_context.fence->SetEventOnCompletion(current_fence_value, g_dx_context.fence_event)
			check(hr, "Failed to set event on completion flag")
			windows.WaitForSingleObject(g_dx_context.fence_event, windows.INFINITE)
		}

		ct.frame_index = cast(int)ct.swapchain->GetCurrentBackBufferIndex()
		check(ct.command_allocator->Reset())

		// swap PSO here if needed (hot reload of shaders)

		// destroy scenes queued for deletion (only of another scene is ready)

		is_a_scene_ready : bool

		for &scene in g_scenes {
			if scene_status_load(&scene.status) == .Ready {
				is_a_scene_ready = true
				break
			}
		}

		if is_a_scene_ready {
			for &scene in g_scenes {
				if scene_status_load(&scene.status) == .QueuedForDeletion {
					scene_destroy(&scene)
				}
			}
		}

		// hot swap handling
		for &pso in ct.psos {
			pso_hotswap_swap(&pso)
		}
	}
}

create_depth_buffer :: proc() {

	ct := &g_dx_context

	opt_clear := dx.CLEAR_VALUE {
		Format = .D32_FLOAT,
		DepthStencil = {Depth = 1.0, Stencil = 0},
	}

	srv_desc := dx.SHADER_RESOURCE_VIEW_DESC {
		Format = .R32_FLOAT,
		ViewDimension = .TEXTURE2D,
		Shader4ComponentMapping = dx.ENCODE_SHADER_4_COMPONENT_MAPPING(0, 1, 2, 3), // this is the default mapping
		Texture2D = {
			MostDetailedMip = 0,
			MipLevels = 1,
		}
	}

	ct.depth_texture = texture_create(nil,
		u64(WINDOW_WIDTH), u32(WINDOW_HEIGHT), .D32_FLOAT, &g_resources_longterm,
		{.DSV, .SRV}, opt_clear_value = &opt_clear, srv_desc = &srv_desc, res_flags = {.ALLOW_DEPTH_STENCIL})
}

imgui_init :: proc() {

	// need
	// sdl window
	//

	// initting dear imgui
	im.CHECKVERSION()
	im.CreateContext()
	io := im.GetIO()

	io.ConfigFlags += {.NavEnableKeyboard, .NavEnableGamepad}
	io.ConfigFlags += {.DockingEnable}
	io.ConfigFlags += {.ViewportsEnable}

	style := im.GetStyle()
	style.WindowRounding = 0
	style.Colors[im.Col.WindowBg].w = 1

	im.StyleColorsDark()

	imgui_impl_sdl2.InitForD3D(g_dx_context.window)


	// // Initialization data, for ImGui_ImplDX12_Init()
	// InitInfo :: struct {
	// 	Device:            ^d3d12.IDevice,
	// 	CommandQueue:      ^d3d12.ICommandQueue,
	// 	NumFramesInFlight: i32,
	// 	RTVFormat:         dxgi.FORMAT,          // RenderTarget format.
	// 	DSVFormat:         dxgi.FORMAT,          // DepthStencilView format.
	// 	UserData:          rawptr,

	// 	// Allocating SRV descriptors for textures is up to the application, so we provide callbacks.
	// 	// (current version of the backend will only allocate one descriptor, future versions will need to allocate more)
	// 	SrvDescriptorHeap:    ^d3d12.IDescriptorHeap,
	// 	SrvDescriptorAllocFn: proc "c" (info: ^InitInfo, out_cpu_desc_handle: ^d3d12.CPU_DESCRIPTOR_HANDLE, out_gpu_desc_handle: ^d3d12.GPU_DESCRIPTOR_HANDLE),
	// 	SrvDescriptorFreeFn:  proc "c" (info: ^InitInfo, cpu_desc_handle: d3d12.CPU_DESCRIPTOR_HANDLE, gpu_desc_handle: d3d12.GPU_DESCRIPTOR_HANDLE),
	// }

	// create a shader resource view  heap (srv)

	c := &g_dx_context


	// creating descriptor heap

	// if it goes above 3, we are dead
	srv_descriptor_heap_desc := dx.DESCRIPTOR_HEAP_DESC {
		NumDescriptors = 3,
		Type = .CBV_SRV_UAV,
		Flags = {.SHADER_VISIBLE},
	}

	hr := c.device->CreateDescriptorHeap(
		&srv_descriptor_heap_desc,
		dx.IDescriptorHeap_UUID,
		(^rawptr)(&g_dx_context.imgui_descriptor_heap),
	)
	check(hr, "could ont create imgui descriptor heap")
	g_dx_context.imgui_descriptor_heap->SetName("imgui's cbv srv uav descriptor heap")
	append(&g_resources_longterm, g_dx_context.imgui_descriptor_heap)

	g_dx_context.imgui_allocator = descriptor_heap_allocator_create(g_dx_context.imgui_descriptor_heap, .CBV_SRV_UAV)

	allocfn := proc "c" (
		info: ^imgui_impl_dx12.InitInfo,
		out_cpu_desc_handle: ^dx.CPU_DESCRIPTOR_HANDLE,
		out_gpu_desc_handle: ^dx.GPU_DESCRIPTOR_HANDLE,
	) {
		context = runtime.default_context()
		cpu, gpu := descriptor_heap_allocator_alloc(&g_dx_context.imgui_allocator)
		out_cpu_desc_handle.ptr = cpu.ptr
		out_gpu_desc_handle.ptr = gpu.ptr
	}

	freefn := proc "c" (
		info: ^imgui_impl_dx12.InitInfo,
		cpu_desc_handle: dx.CPU_DESCRIPTOR_HANDLE,
		gpu_desc_handle: dx.GPU_DESCRIPTOR_HANDLE,
	) {
		context = runtime.default_context()
		descriptor_heap_allocator_free(&g_dx_context.imgui_allocator, cpu_desc_handle, gpu_desc_handle)
	}


	dx12_init := imgui_impl_dx12.InitInfo {
		Device = g_dx_context.device,
		CommandQueue = g_dx_context.queue,
		// not sure what this is
		NumFramesInFlight = 2,
		RTVFormat = .R8G8B8A8_UNORM,
		DSVFormat = .D32_FLOAT,
		SrvDescriptorHeap = g_dx_context.imgui_descriptor_heap,
		SrvDescriptorAllocFn = allocfn,
		SrvDescriptorFreeFn = freefn,
	}

	imgui_impl_dx12.Init(&dx12_init)
}

imgui_destoy :: proc() {
	imgui_impl_sdl2.Shutdown() // here
	imgui_impl_dx12.Shutdown()
	im.DestroyContext()
}

// call this right before swapchain present
render_imgui :: proc() {

	// setting imgui's descriptor heap
	// if i don't do this, it errors out. seems like RenderDrawData doesn't set it
	//  by itself
	g_dx_context.cmdlist->SetDescriptorHeaps(1, &g_dx_context.imgui_descriptor_heap)

	// need graphics command list
	imgui_impl_dx12.RenderDrawData(im.GetDrawData(), g_dx_context.cmdlist)

	io := im.GetIO()

	if .ViewportsEnable in io.ConfigFlags {
		im.UpdatePlatformWindows()
		im.RenderPlatformWindowsDefault()
	}
}


// helpers

//unused

/*
get_descriptor_heap_gpu_address :: proc(
	heap: ^dx.IDescriptorHeap,
	offset: u32 = 0,
) -> (
	gpu_descriptor_handle: dx.GPU_DESCRIPTOR_HANDLE,
) {
	heap->GetGPUDescriptorHandleForHeapStart(&gpu_descriptor_handle)
	desc: dx.DESCRIPTOR_HEAP_DESC
	heap->GetDesc(&desc)
	increment := dx_context.device->GetDescriptorHandleIncrementSize(desc.Type)
	gpu_descriptor_handle.ptr += u64(offset * increment)
	return
}
*/

// gives you a transformation matrix given a position and scale and rot
get_world_mat :: proc(pos, scale: v3, rot_rads: f32 = 0, rot_vec: v3 = {1, 0, 0}) -> dxm {


	translation_mat := linalg.matrix4_translate_f32(pos)
	scale_mat := linalg.matrix4_scale_f32(scale)

	rot_mat := linalg.matrix4_rotate_f32(rot_rads, rot_vec)

	return translation_mat * scale_mat * rot_mat
}

//unused
/*
get_world_mat_quat :: proc(pos, scale: v3, rot_quat: quaternion128) -> dxm {

	translation_mat := linalg.matrix4_translate_f32(pos)
	scale_mat := linalg.matrix4_scale_f32(scale)

	rot_mat := linalg.matrix4_from_quaternion_f32(rot_quat)

	return translation_mat * scale_mat * rot_mat
}
*/

create_gbuffer_unit :: proc(format: dxgi.FORMAT, debug_name: string) -> Texture {
	opt_clear_value := dx.CLEAR_VALUE {
		Format = format,
		Color = {0,0,0,1}
	}

	return texture_create(nil, u64(WINDOW_WIDTH), u32(WINDOW_HEIGHT),
		format, &g_resources_longterm, {.SRV, .RTV}, texture_name = debug_name, 
		res_flags = {.ALLOW_RENDER_TARGET}, opt_clear_value = &opt_clear_value)
}

create_gbuffer :: proc() -> GBuffer {
	return GBuffer {
		gbuffers = [GBufferUnitName]Texture {
			.Albedo = create_gbuffer_unit(.R8G8B8A8_UNORM, "gbuffer - ALBEDO"),
			.Normal = create_gbuffer_unit(.R10G10B10A2_UNORM, "gbuffer - NORMALS"),
			.AO_Rough_Metal = create_gbuffer_unit(.R8G8B8A8_UNORM, "gbuffer - AO ROUGH METAL")
		}
	}
}

pso_gbuffer_render :: proc() {

	ct := &g_dx_context

	ct.cmdlist->SetPipelineState(ct.psos[.GBuffer_Pass].pipeline_state)
	ct.cmdlist->SetDescriptorHeaps(1, &ct.cbv_srv_uav_heap.heap)
	ct.cmdlist->SetGraphicsRootSignature(ct.psos[.GBuffer_Pass].root_signature)

	transition_resource(ct.depth_texture.buffer, ct.cmdlist, {.PIXEL_SHADER_RESOURCE}, {.DEPTH_WRITE})

	set_viewport_stuff()

	// Transitioning gbuffers from SRVs to render target
	transition_gbuffers(true)

	// Setting render targets. Clearing DSV and RTV.
	{
		rtv_handles := [GBUFFER_COUNT]dx.CPU_DESCRIPTOR_HANDLE {
			texture_get_rtv_cpu_address(g_dx_context.gbuffer.gbuffers[.Albedo]),
			texture_get_rtv_cpu_address(g_dx_context.gbuffer.gbuffers[.Normal]),
			texture_get_rtv_cpu_address(g_dx_context.gbuffer.gbuffers[.AO_Rough_Metal]),
		}
		dsv_handle := texture_get_dsv_cpu_address(ct.depth_texture)

		// setting depth buffer
		ct.cmdlist->OMSetRenderTargets(GBUFFER_COUNT, &rtv_handles[0], false, &dsv_handle)

		// clear backbuffer
		clearcolor := [?]f32{0, 0, 0, 1.0}

		// we should probably clear each gbuffer individually to a sane value...
		for rtv_handle in rtv_handles {
			ct.cmdlist->ClearRenderTargetView(rtv_handle, &clearcolor, 0, nil)
		}

		// clearing depth buffer
		ct.cmdlist->ClearDepthStencilView(dsv_handle, {.DEPTH, .STENCIL}, 1.0, 0, 0, nil)
	}

	// draw call
	ct.cmdlist->IASetPrimitiveTopology(.TRIANGLELIST)

	// drawing all scenes
	for &scene in g_scenes {
		st := scene_status_load(&scene.status)
		#partial switch st {
		case .Ready, .QueuedForDeletion:
		case:
			continue
		}

		queue_wait_on_upload_fence(ct.queue, scene.fence_value)

		// binding vertex buffer view and instance buffer view
		vertex_buffers_views := [?]dx.VERTEX_BUFFER_VIEW{scene.vertex_buffer_view}

		ct.cmdlist->IASetVertexBuffers(0, len(vertex_buffers_views), &vertex_buffers_views[0])
		ct.cmdlist->IASetIndexBuffer(&scene.index_buffer_view)

		// rendering each mesh individually
		// going through scene tree

		// drawing scene

		DrawConstants :: struct {
			mesh_index: u32,
			material_index: u32,
		}

		scene_walk(scene, nil, proc(node: Node, scene: Scene, data: rawptr) {
			ct := &g_dx_context

			if node.mesh == -1 {
				return
			}

			mesh_to_render := scene.meshes[node.mesh]

			for prim in mesh_to_render.primitives {
				dc := DrawConstants {
					mesh_index = u32(g_mesh_drawn_count),
					material_index = u32(prim.material_index),
				}
				ct.cmdlist->SetGraphicsRoot32BitConstants(1, 2, &dc, 0)
				ct.cmdlist->DrawIndexedInstanced(prim.index_count, 1, prim.index_offset, 0, 0)
			}
		})
	}

}

// to_render_target = false: transitions from render target to pixel shader resource
// to_render_target = true: viceversa
transition_gbuffers :: proc(to_render_target: bool) {

	ct := &g_dx_context
	res_barriers: [GBufferUnitName]dx.RESOURCE_BARRIER

	res_barriers[.Albedo] = dx.RESOURCE_BARRIER {
		Type = .TRANSITION,
		Flags = {},
		Transition = {
			pResource = nil,
			StateBefore = to_render_target ? {.PIXEL_SHADER_RESOURCE} : {.RENDER_TARGET},
			StateAfter = to_render_target ? {.RENDER_TARGET} : {.PIXEL_SHADER_RESOURCE},
			Subresource = dx.RESOURCE_BARRIER_ALL_SUBRESOURCES,
		},
	}

	// populating all res barriers with each gbuffer
	res_barriers[.Albedo] = res_barriers[.Albedo]
	res_barriers[.Albedo].Transition.pResource = ct.gbuffer.gbuffers[.Albedo].buffer

	res_barriers[.Normal] = res_barriers[.Albedo]
	res_barriers[.Normal].Transition.pResource = ct.gbuffer.gbuffers[.Normal].buffer

	res_barriers[.AO_Rough_Metal] = res_barriers[.Albedo]
	res_barriers[.AO_Rough_Metal].Transition.pResource = ct.gbuffer.gbuffers[.AO_Rough_Metal].buffer

	ct.cmdlist->ResourceBarrier(GBUFFER_COUNT, &res_barriers[cast(GBufferUnitName)0])
}

pso_lighting_render :: proc() {


	ct := &g_dx_context

	ct.cmdlist->SetPipelineState(ct.psos[.Lighting_Pass].pipeline_state)

	// Transitioning gbuffers from render target to SRVs
	transition_gbuffers(false)

	// here u have to transition the swapchain buffer so it is a RT
	{
		to_render_target_barrier := dx.RESOURCE_BARRIER {
			Type = .TRANSITION,
			Flags = {},
			Transition = {
				pResource = g_dx_context.targets[g_dx_context.frame_index],
				StateBefore = dx.RESOURCE_STATE_PRESENT,
				StateAfter = {.RENDER_TARGET},
				Subresource = dx.RESOURCE_BARRIER_ALL_SUBRESOURCES,
			},
		}

		ct.cmdlist->ResourceBarrier(1, &to_render_target_barrier)
	}

	transition_resource(ct.depth_texture.buffer, ct.cmdlist, {.DEPTH_WRITE}, {.PIXEL_SHADER_RESOURCE})

	// descriptor heap is directly accessed in the shader.
	//  so we don't need to set a descriptor table or set texture slots.
	ct.cmdlist->SetDescriptorHeaps(1, &ct.cbv_srv_uav_heap.heap)
	ct.cmdlist->SetGraphicsRootSignature(ct.psos[.Lighting_Pass].root_signature)

	set_viewport_stuff()

	// Setting render targets. Clearing RTV.
	{
		rtv_handles := [1]dx.CPU_DESCRIPTOR_HANDLE {
			get_descriptor_heap_cpu_address(ct.swapchain_rtv_descriptor_heap, ct.frame_index),
		}

		ct.cmdlist->OMSetRenderTargets(1, &rtv_handles[0], false, nil)

		// clear backbuffer
		clearcolor := [?]f32{0.05, 0.05, 0.05, 1.0}
		ct.cmdlist->ClearRenderTargetView(rtv_handles[0], &clearcolor, 0, nil)
	}

	// draw call
	ct.cmdlist->IASetPrimitiveTopology(.TRIANGLELIST)

	// 3. Draw 3 vertices (which triggers the VS 3 times)
	ct.cmdlist->DrawInstanced(3, 1, 0, 0)
}

// TODO: this is using resources from any loaded scene. change that.
pso_gizmos_render :: proc () {
	scene, ok:= get_first_active_scene() 
	if !ok do return

	ct := &g_dx_context

	// updating gizmo data (looking at lights)
	gizmos_count : u32 = 1
	{
		gizmos_instances := make([]InstanceData, gizmos_count, context.temp_allocator)

		gizmos_instances[0] = InstanceData {
			world_mat = get_world_mat(g_light_pos, 0.1),
			color = v4{1,0,0, 0.5}
		}

		copy_to_buffer_already_mapped(scene.vb_gizmos_instance_data.gpu_pointer, slice.to_bytes(gizmos_instances))
	}

	ct.cmdlist->SetPipelineState(ct.psos[.Gizmos].pipeline_state)

	// setting descriptor heap for our cbv srv uav's
	ct.cmdlist->SetDescriptorHeaps(1, &ct.cbv_srv_uav_heap.heap)

	// This state is reset everytime the cmd list is reset, so we need to rebind it
	ct.cmdlist->SetGraphicsRootSignature(ct.psos[.Gizmos].root_signature)

	// setting rtv and dsv

	transition_resource(ct.depth_texture.buffer, ct.cmdlist, {.PIXEL_SHADER_RESOURCE}, {.DEPTH_WRITE})
	defer {
		transition_resource(ct.depth_texture.buffer, ct.cmdlist, {.DEPTH_WRITE}, {.PIXEL_SHADER_RESOURCE})
	}

	rtv_handles := [1]dx.CPU_DESCRIPTOR_HANDLE {
		get_descriptor_heap_cpu_address(ct.swapchain_rtv_descriptor_heap, ct.frame_index),
	}

	dsv_handle := texture_get_dsv_cpu_address(ct.depth_texture)

	ct.cmdlist->OMSetRenderTargets(1, &rtv_handles[0], false, &dsv_handle)

	set_viewport_stuff()

	ct.cmdlist->IASetPrimitiveTopology(.TRIANGLELIST)

	// binding vertex buffer view and instance buffer view
	vertex_buffers_views := [?]dx.VERTEX_BUFFER_VIEW{scene.vertex_buffer_view, scene.vb_gizmos_instance_data.vbv}

	ct.cmdlist->IASetVertexBuffers(0, len(vertex_buffers_views), &vertex_buffers_views[0])
	ct.cmdlist->IASetIndexBuffer(&scene.index_buffer_view)

	// TEST: use first mesh primitive from main vertex buffer
	uv_sphere_primitive := scene.uv_sphere_mesh.primitives[0]
	ct.cmdlist->DrawIndexedInstanced(uv_sphere_primitive.index_count, gizmos_count, uv_sphere_primitive.index_offset, 0, 0)
}

// unused
/*
print_ref_count :: proc(obj: ^dx.IUnknown) {
	obj->AddRef()
	count := obj->Release()
	lprintfln("count: %v", count)
}
*/



arena_report :: proc(arena_name: string, arena: virtual.Arena) {
	lprintfln("===== Arena Report: name: \"%v\": total used: %vMB, total reserved: %vMB", arena_name, cast(f32)arena.total_used / cast(f32)mem.Megabyte,
		cast(f32)arena.total_reserved / cast(f32)mem.Megabyte)
}

scene_swap :: proc(new_scene: string) {
	// queue for deletion all active scenes. start loading on a free slot
	found_free : bool
	for &s in g_scenes {
		st := scene_status_load(&s.status)
		#partial switch st {
		case .Ready:
			scene_status_store(&s.status, .QueuedForDeletion)
		case .Free:
			if !found_free {
				scene_schedule_load(&s, new_scene)
			}
			found_free = true
		}
	}

	if !found_free {
		lprintln("could not find a free scene. scene swap FAILED")
		os.exit(1)
	}
}

scene_destroy :: proc(scene: ^Scene) {
	for r in scene.resource_pool {
		r->Release()
	}
	virtual.arena_free_all(&scene.allocator)
	scene_status_store(&scene.status, .Free)
}

// call this on scene slot u wanna start to load
scene_schedule_load :: proc(scene: ^Scene, scene_name: string) {
	if scene_status_load(&scene.status) != .Free {
		lprintfln("cannot schedule load for scene unless it's status=free")
		assert(false)
		return
	}

	virtual.arena_free_all(&scene.allocator)
	scene_allocator := virtual.arena_allocator(&scene.allocator)
	scene.path = strings.clone(scene_name, scene_allocator)

	// This "moves" the scene to the upload thread.
	// the upload thread will move the scene back to the main thread by setting status to "Ready"
	scene_status_store(&scene.status, .Loading)
}

get_first_active_scene :: proc() -> (scene: ^Scene, ok: bool) {
	for &scene in g_scenes {
		st := scene_status_load(&scene.status)
		#partial switch st {
		case .Ready, .QueuedForDeletion: // if we draw queued for deletion, a nice effect happens when u switch scenes.
			return &scene, true
		}
	}
	return nil, false
}
