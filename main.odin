package main

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
import img "vendor:stb/image"
import "core:c"
import "core:math"
import "core:math/linalg"
import "vendor:cgltf"
import sa "core:container/small_array"
import "base:runtime"
import "core:math/rand"
import "core:sync"
import dxc "vendor:directx/dxc"
import "core:prof/spall"

// imgui
import im "../odin-imgui"
// imgui sdl2 implementation
import "../odin-imgui/imgui_impl_sdl2"
// imgui dx12 implementation
import "../odin-imgui/imgui_impl_dx12"

// --- const definitions / aliases ---

PROFILE :: #config(PROFILE, false)

NUM_RENDERTARGETS :: 2

TURNS_TO_RAD :: math.PI * 2

v2 :: linalg.Vector2f32
v3 :: linalg.Vector3f32
v4 :: linalg.Vector4f32

dxm :: matrix[4, 4]f32

DXResourcePool :: sa.Small_Array(100, ^dx.IUnknown)

gbuffer_shader_filename :: "shader.hlsl"
lighting_shader_filename :: "lighting.hlsl"

gbuffer_count :: 3

// ---- all state ----

// profiling stuff

when PROFILE {
	spall_ctx: spall.Context
	@(thread_local)
	spall_buffer: spall.Buffer
}

// window dimensions
wx := i32(2000)
wy := i32(1000)

dx_context: Context
start_time: time.Time
light_pos: v3
light_int: f32
light_speed: f32
place_texture: bool
the_time_sec: f32
exit_app: bool

// last_write time for shaders

// dx resources to be freed at the end of the app
resources_longterm: DXResourcePool

// constant buffer data
ConstantBufferData :: struct #align (256) {
	view: dxm,
	projection: dxm,
	light_pos: v3,
	light_int: f32,
	view_pos: v3,
	time: f32,
	place_texture: b32,
}

// all meshes use the same index/vertex buffer.
// so we just have to store the offset and index count to render a specific mesh
Mesh :: struct {
	index_offset: u32,
	index_count: u32,
}

// testing
meshes: []Mesh

Scene :: struct {
	nodes: []Node,
	root_nodes: []int,
	mesh_count: uint,
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

scene: Scene

// struct that holds instance data, for an instance rendering example
InstanceData :: struct #align (256) {
	world_mat: dxm,
	color: v3,
}

get_most_of_cbv :: proc() -> ConstantBufferData {

	// ticking cbv time value
	thetime := time.diff(start_time, time.now())
	the_time_sec = f32(thetime) / f32(time.Second)
	// if the_time_sec > 1 {
	// 	start_time = time.now()
	// }

	// sending constant buffer data
	view, projection := get_view_projection(cur_cam)

	return ConstantBufferData {
		view = view,
		projection = projection,
		light_pos = light_pos,
		light_int = light_int,
		view_pos = cur_cam.pos,
		time = the_time_sec,
		place_texture = b32(place_texture),
	}
}

cb_update :: proc() {

	cbv_data := get_most_of_cbv()

	// sending data to the cpu mapped memory that the gpu can read
	mem.copy(dx_context.constant_buffer_map, (rawptr)(&cbv_data), size_of(cbv_data))
}

VertexData :: struct {
	pos: v3,
	normal: v3,
	uv: v2,
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
	gb_position: GBufferUnit,
	rtv_heap: ^dx.IDescriptorHeap,
	srv_heap: ^dx.IDescriptorHeap,
}

HotSwapState :: struct {
	// TODO: store more data here so u don't have to pass the data around in the hotswap methods
	last_write_time: os.File_Time,
	pso_swap: ^dx.IPipelineState,

	// index in the queue array to free the resource
	// i use this to swap the pointer when the pso gets hot swapped
	pso_index: int,
}

Context :: struct {
	// sdl stuff
	window: ^sdl.Window,

	// imgui stuff
	imgui_descriptor_heap: ^dx.IDescriptorHeap,
	imgui_allocator: DescriptorHeapAllocator,

	// core stuff
	device: ^dx.IDevice,
	factory: ^dxgi.IFactory4,
	queue: ^dx.ICommandQueue,
	swapchain: ^dxgi.ISwapChain3,
	command_allocator: ^dx.ICommandAllocator,
	dxc_compiler: ^dxc.ICompiler3,
	pipeline_gbuffer: ^dx.IPipelineState,
	cmdlist: ^dx.IGraphicsCommandList,
	constant_buffer_map: rawptr, //maps to our test constant buffer
	gbuffer_pass_root_signature: ^dx.IRootSignature,
	constant_buffer: ^dx.IResource,
	vertex_buffer_view: dx.VERTEX_BUFFER_VIEW,
	index_buffer_view: dx.INDEX_BUFFER_VIEW,
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

	// resources
	texture: ^dx.IResource,
	res_structured_buffer: ^dx.IResource,

	// descriptor heap for our cbv and srv (texture)
	descriptor_heap_cbv_srv_uav: ^dx.IDescriptorHeap,

	// vertex count
	vertex_count: u32,
	index_count: u32,

	// depth buffer
	depth_stencil_res: ^dx.IResource,
	descriptor_heap_dsv: ^dx.IDescriptorHeap,

	// instance buffer
	instance_buffer: VertexBuffer,

	meshes_to_render: int,

	// hot swap shader state
	lighting_hotswap: HotSwapState,
	gbuffer_hotswap: HotSwapState, // todo this one (make helper functions for setting initial state and swapping code)
}

// initializes app data in Context struct
context_init :: proc(con: ^Context) {
	cur_cam = camera_init()
	light_pos = v3{4.1, 3.5, 4.5}
	light_int = 1
	light_speed = 0.002
	con.meshes_to_render = len(meshes)
}

check :: proc(res: dx.HRESULT, message: string) {
	if (res >= 0) {
		return
	}

	fmt.printf("%v. Error code: %0x\n", message, u32(res))
	os.exit(-1)
}

main :: proc() {

	// setting up profiling
	when PROFILE {
		spall_ctx = spall.context_create("trace_test.spall")
		defer spall.context_destroy(&spall_ctx)

		buffer_backing := make([]u8, spall.BUFFER_DEFAULT_SIZE)
		defer delete(buffer_backing)

		spall_buffer = spall.buffer_create(buffer_backing, u32(sync.current_thread_id()))
		defer spall.buffer_destroy(&spall_ctx, &spall_buffer)

		spall.SCOPED_EVENT(&spall_ctx, &spall_buffer, #procedure)
	}

	// Init SDL and create window
	if err := sdl.Init(sdl.InitFlags{.TIMER, .AUDIO, .VIDEO, .EVENTS}); err != 0 {
		fmt.eprintln(err)
		return
	}

	defer sdl.Quit()
	dx_context.window = sdl.CreateWindow(
		"lucydx12",
		sdl.WINDOWPOS_UNDEFINED,
		sdl.WINDOWPOS_UNDEFINED,
		wx,
		wy,
		{.ALLOW_HIGHDPI, .SHOWN, .RESIZABLE},
	)

	if dx_context.window == nil {
		fmt.eprintln(sdl.GetError())
		return
	}

	defer sdl.DestroyWindow(dx_context.window)

	init_dx()

	device := dx_context.device


	hr: dx.HRESULT

	{
		desc := dx.COMMAND_QUEUE_DESC {
			Type = .DIRECT,
		}

		hr = device->CreateCommandQueue(&desc, dx.ICommandQueue_UUID, (^rawptr)(&dx_context.queue))
		check(hr, "Failed creating command queue")
		sa.push(&resources_longterm, dx_context.queue)
	}

	// Create the swapchain, it's the thing that contains render targets that we draw into.
	//  It has 2 render targets (NUM_RENDERTARGETS), giving us double buffering.
	dx_context.swapchain = create_swapchain(dx_context.factory, dx_context.queue, dx_context.window)

	dx_context.frame_index = dx_context.swapchain->GetCurrentBackBufferIndex()

	// Descriptors describe the GPU data and are allocated from a Descriptor Heap
	{
		desc := dx.DESCRIPTOR_HEAP_DESC {
			NumDescriptors = NUM_RENDERTARGETS,
			Type = .RTV,
			Flags = {},
		}

		hr = device->CreateDescriptorHeap(
			&desc,
			dx.IDescriptorHeap_UUID,
			(^rawptr)(&dx_context.swapchain_rtv_descriptor_heap),
		)
		check(hr, "Failed creating descriptor heap")
		dx_context.swapchain_rtv_descriptor_heap->SetName("lucy's swapchain RTV descriptor heap")
		sa.push(&resources_longterm, dx_context.swapchain_rtv_descriptor_heap)
	}

	// Fetch the two render targets from the swapchain

	{
		rtv_descriptor_size: u32 = device->GetDescriptorHandleIncrementSize(.RTV)
		rtv_descriptor_handle: dx.CPU_DESCRIPTOR_HANDLE
		dx_context.swapchain_rtv_descriptor_heap->GetCPUDescriptorHandleForHeapStart(&rtv_descriptor_handle)

		for i: u32 = 0; i < NUM_RENDERTARGETS; i += 1 {
			hr = dx_context.swapchain->GetBuffer(i, dx.IResource_UUID, (^rawptr)(&dx_context.targets[i]))
			dx_context.targets[i]->Release()
			check(hr, "Failed getting render target")
			device->CreateRenderTargetView(dx_context.targets[i], nil, rtv_descriptor_handle)
			rtv_descriptor_handle.ptr += uint(rtv_descriptor_size)
		}
	}

	// creating depth buffer

	// The command allocator is used to create the commandlist that is used to tell the GPU what to draw
	hr = device->CreateCommandAllocator(.DIRECT, dx.ICommandAllocator_UUID, (^rawptr)(&dx_context.command_allocator))
	check(hr, "Failed creating command allocator")
	sa.push(&resources_longterm, dx_context.command_allocator)

	// Creating G-Buffer textures and RTV's
	dx_context.gbuffer = create_gbuffer()

	// constant buffer
	{
		heap_properties := dx.HEAP_PROPERTIES {
			Type = .UPLOAD,
		}
		constant_buffer_desc := dx.RESOURCE_DESC {
			Width = size_of(ConstantBufferData),
			Dimension = .BUFFER,
			Height = 1,
			Layout = .ROW_MAJOR,
			Format = .UNKNOWN,
			DepthOrArraySize = 1,
			MipLevels = 1,
			SampleDesc = {Count = 1},
		}

		hr = device->CreateCommittedResource(
			&heap_properties,
			dx.HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES,
			&constant_buffer_desc,
			dx.RESOURCE_STATE_GENERIC_READ,
			nil,
			dx.IResource_UUID,
			(^rawptr)(&dx_context.constant_buffer),
		)

		check(hr, "failed creating constant buffer")
		dx_context.constant_buffer->SetName("lucy's constant buffer")
		sa.push(&resources_longterm, dx_context.constant_buffer)

		// empty range means the cpu won't read from it
		dx_context.constant_buffer->Map(0, &dx.RANGE{}, &dx_context.constant_buffer_map)
	}

	create_depth_buffer()

	/* 
	From https://docs.microsoft.com/en-us/windows/win32/direct3d12/root-signatures-overview:
	
	A root signature is configured by the app and links command lists to the resources the shaders require.
	The graphics command list has both a graphics and compute root signature. A compute command list will
	simply have one compute root signature. These root signatures are independent of each other.
	*/

	create_gbuffer_pass_root_signature()

	dx_context.instance_buffer = create_instance_buffer_example()

	create_gbuffer_pso_initial()
	create_lighting_pso_initial()

	// Create the commandlist that is reused further down.
	hr = device->CreateCommandList(
		0,
		.DIRECT,
		dx_context.command_allocator,
		dx_context.pipeline_gbuffer,
		dx.ICommandList_UUID,
		(^rawptr)(&dx_context.cmdlist),
	)
	check(hr, "Failed to create command list")
	hr = dx_context.cmdlist->Close()
	check(hr, "Failed to close command list")
	sa.push(&resources_longterm, dx_context.cmdlist)


	vertex_buffer: ^dx.IResource
	index_buffer: ^dx.IResource

	imgui_init()

	// creating and filling vertex and index buffers
	{
		// get vertex data from gltf file
		vertices, indices := do_gltf_stuff()
		dx_context.vertex_count = u32(len(vertices))

		// VERTEXDATA
		// vertex data and index data is in an upload heap.
		// This isn't optimal for geometry that doesn't change much.
		// If we want to make this fast, the vertex data needs to be in
		// a DEFAULT heap (vram). you transfer the data from an upload heap
		// to the default heap. but it's more complicated.
		heap_props := dx.HEAP_PROPERTIES {
			Type = .UPLOAD,
		}

		vertex_buffer_size := len(vertices) * size_of(vertices[0])

		resource_desc := dx.RESOURCE_DESC {
			Dimension = .BUFFER,
			Alignment = 0,
			Width = u64(vertex_buffer_size),
			Height = 1,
			DepthOrArraySize = 1,
			MipLevels = 1,
			Format = .UNKNOWN,
			SampleDesc = {Count = 1, Quality = 0},
			Layout = .ROW_MAJOR,
			Flags = {},
		}

		hr = device->CreateCommittedResource(
			&heap_props,
			{},
			&resource_desc,
			dx.RESOURCE_STATE_GENERIC_READ,
			nil,
			dx.IResource_UUID,
			(^rawptr)(&vertex_buffer),
		)
		check(hr, "Failed creating vertex buffer")
		sa.push(&resources_longterm, vertex_buffer)

		gpu_data: rawptr
		read_range: dx.RANGE

		hr = vertex_buffer->Map(0, &read_range, &gpu_data)
		check(hr, "Failed creating vertex buffer resource")

		mem.copy(gpu_data, &vertices[0], vertex_buffer_size)
		vertex_buffer->Unmap(0, nil)

		dx_context.vertex_buffer_view = dx.VERTEX_BUFFER_VIEW {
			BufferLocation = vertex_buffer->GetGPUVirtualAddress(),
			StrideInBytes = u32(vertex_buffer_size) / dx_context.vertex_count,
			SizeInBytes = u32(vertex_buffer_size),
		}

		// creating index buffer resource

		index_buffer_size := len(indices) * size_of(indices[0])
		dx_context.index_count = u32(len(indices))

		resource_desc.Width = u64(index_buffer_size)

		hr = device->CreateCommittedResource(
			&heap_props,
			{},
			&resource_desc,
			dx.RESOURCE_STATE_GENERIC_READ,
			nil,
			dx.IResource_UUID,
			(^rawptr)(&index_buffer),
		)
		check(hr, "failed index buffer")
		index_buffer->SetName("lucy's index buffer")
		sa.push(&resources_longterm, index_buffer)

		dx_context.index_buffer_view = dx.INDEX_BUFFER_VIEW {
			BufferLocation = index_buffer->GetGPUVirtualAddress(),
			SizeInBytes = u32(index_buffer_size),
			Format = .R32_UINT,
		}

		hr = index_buffer->Map(0, &dx.RANGE{}, &gpu_data)
		check(hr, "failed mapping")

		mem.copy(gpu_data, &indices[0], index_buffer_size)
		index_buffer->Unmap(0, nil)
	}

	create_sample_texture()
	create_structured_buffer(&resources_longterm)

	create_descriptor_heap_cbv_srv_uav()

	// This fence is used to wait for frames to finish
	{
		hr = device->CreateFence(dx_context.fence_value, {}, dx.IFence_UUID, (^rawptr)(&dx_context.fence))
		check(hr, "Failed to create fence")
		sa.push(&resources_longterm, dx_context.fence)
		dx_context.fence_value += 1
		manual_reset: windows.BOOL = false
		initial_state: windows.BOOL = false
		dx_context.fence_event = windows.CreateEventW(nil, manual_reset, initial_state, nil)
		if dx_context.fence_event == nil {
			fmt.println("Failed to create fence event")
			return
		}
	}

	context_init(&dx_context)


	// looping


	start_time = time.now()
	do_main_loop()

	// cleaning up

	imgui_destoy()

	#reverse for i in sa.slice(&resources_longterm) {
		i->Release()
	}

	// this does nothing
	sdl.DestroyWindow(dx_context.window)

	when ODIN_DEBUG {
		fmt.println("=== live object report start =====")
		debug_device: ^dx.IDebugDevice2
		dx_context.device->QueryInterface(dx.IDebugDevice2_UUID, (^rawptr)(&debug_device))
		// Finally, release the device (it is not in any pool)
		// The device will be freed after we release the debug device
		dx_context.device->Release()
		debug_device->ReportLiveDeviceObjects({.DETAIL, .IGNORE_INTERNAL})
		debug_device->Release()

		// DXGI report
		dxgi_debug: ^dxgi.IDebug1
		dxgi.DXGIGetDebugInterface1(0, dxgi.IDebug1_UUID, (^rawptr)(&dxgi_debug))
		dxgi_debug->ReportLiveObjects(dxgi.DEBUG_ALL, {})
		fmt.println("=== report end =====")
	}
}

g_frame_dt : f64 = 0.2 // in ms

do_main_loop :: proc() {
	
	last_time := time.now()
	
	main_loop: for {
		when PROFILE do spall.SCOPED_EVENT(&spall_ctx, &spall_buffer, name = "main loop")
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

		if exit_app {
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
		Width = u32(wx),
		Height = u32(wy),
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
	sa.push(&resources_longterm, swapchain)

	return
}

// inits dx factory device
init_dx :: proc() {
	hr: dx.HRESULT

	// Init DXGI factory. DXGI is the link between the window and DirectX
	factory: ^dxgi.IFactory4

	{
		flags: dxgi.CREATE_FACTORY

		when ODIN_DEBUG {
			flags += {.DEBUG}
		}

		hr = dxgi.CreateDXGIFactory2(flags, dxgi.IFactory4_UUID, cast(^rawptr)&factory)
		check(hr, "Failed creating factory")
		sa.push(&resources_longterm, factory)
	}

	dx_context.factory = factory

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
		sa.push(&resources_longterm, adapter)
		if .SOFTWARE in desc.Flags {
			continue
		}

		device: ^dx.IDevice
		hr = dx.CreateDevice((^dxgi.IUnknown)(adapter), ._12_0, dx.IDevice_UUID, (^rawptr)(&device))

		if hr >= 0 {
			dx_context.device = device
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
		dx_context.device->QueryInterface(dx.IInfoQueue1_UUID, (^rawptr)(&info_queue))
		cb_cookie: u32
		hr = info_queue->RegisterMessageCallback(lucy_log_callback, {}, nil, &cb_cookie)
		check(hr, "failed to register")
		info_queue->Release()
	}

	dx_context.dxc_compiler = dxc_init()
}

lucy_log_callback :: proc "c" (
	category: dx.MESSAGE_CATEGORY,
	severity: dx.MESSAGE_SEVERITY,
	id: dx.MESSAGE_ID,
	description: cstring,
	ctx: rawptr,
) {
	context = runtime.default_context()

	severity, ok_2 := reflect.enum_name_from_value(severity)
	cat, ok := reflect.enum_name_from_value(category)
	msg := string(description)

	fmt.printfln("%v: (%v) %v", severity, cat, msg)
}

get_projection_matrix :: proc(fov_rad: f32, screenWidth: i32, screenHeight: i32, near: f32, far: f32) -> dxm {
	f := math.tan_f32(fov_rad * 0.5)

	aspect := f32(screenWidth) / f32(screenHeight)

	return dxm {
		aspect / f,
		0.0,
		0.0,
		0.0,
		0.0,
		1 / f,
		0.0,
		0.0,
		0.0,
		0.0,
		far / (far - near),
		-(near * far) / (far - near),
		0.0,
		0.0,
		1.0,
		0.0,
	}
}

update :: proc() {

	c := &dx_context

	sdl.PumpEvents()
	keyboard := sdl.GetKeyboardStateAsSlice()

	light_pos.x = linalg.sin(the_time_sec * light_speed) * 2
	light_pos.z = linalg.cos(the_time_sec * light_speed) * 2

	if keyboard[sdl.Scancode.ESCAPE] == 1 {
		exit_app = true
	}

	hotswap_watch(
		&c.lighting_hotswap,
		c.lighting_pass_root_signature,
		lighting_shader_filename,
		pso_creation_proc = create_new_lighting_pso,
	)

	hotswap_watch(
		&c.gbuffer_hotswap,
		c.gbuffer_pass_root_signature,
		gbuffer_shader_filename,
		pso_creation_proc = create_new_gbuffer_pso,
	)

	// im.End()
	//

	// imgui stuff
	// im.ShowDemoWindow()
	
	draw_gui()


	camera_tick(keyboard)
	// fmt.printfln("%v", g_frame_dt)
}

draw_gui :: proc() {
	
	c := &dx_context
	im.Begin("lucydx12")

	im.DragFloat3("light pos", &light_pos, 0.1, -5, 5)
	im.DragFloat("light intensity", &light_int, 0.1, 0, 20)
	im.DragFloat("light speed", &light_speed, 0.0001, 0, 20)

	im.InputInt("mesh count to draw", (^i32)(&c.meshes_to_render))

	im.Checkbox("place texture", &place_texture)
	
	// Drawing delta time
	{
		sb := strings.builder_make_len_cap(0, 30, allocator = context.temp_allocator)
		fmt.sbprintfln(&sb, "DT: %.2f", g_frame_dt)
		dt_cstring := strings.to_cstring(&sb)
		im.Text(dt_cstring)
	}
	
	// if im.Button("Re-roll teapots") {
	// 	reroll_teapots()
	// }
}

g_mesh_drawn_count: int = 0
render :: proc() {

	c := &dx_context

	hr: dx.HRESULT

	cb_update()

	g_mesh_drawn_count = 0

	// case .WINDOWEVENT:
	// This is equivalent to WM_PAINT in win32 API
	// if e.window.event == .EXPOSED {
	hr = c.command_allocator->Reset()
	check(hr, "Failed resetting command allocator")

	hr = c.cmdlist->Reset(c.command_allocator, c.pipeline_gbuffer)
	check(hr, "Failed to reset command list")

	render_gbuffer_pass()

	render_lighting_pass()

	// Cannot draw after this point!!


	// Transitioning the render target to "Present" state

	{
		to_present_barrier := dx.RESOURCE_BARRIER {
			Type = .TRANSITION,
			Flags = {},
			Transition = {
				pResource = dx_context.targets[dx_context.frame_index],
				StateBefore = {.RENDER_TARGET},
				StateAfter = dx.RESOURCE_STATE_PRESENT,
				Subresource = dx.RESOURCE_BARRIER_ALL_SUBRESOURCES,
			},
		}

		c.cmdlist->ResourceBarrier(1, &to_present_barrier)
	}

	hr = c.cmdlist->Close()
	check(hr, "Failed to close command list")

	// execute
	cmdlists := [?]^dx.IGraphicsCommandList{c.cmdlist}
	dx_context.queue->ExecuteCommandLists(len(cmdlists), (^^dx.ICommandList)(&cmdlists[0]))

	// present
	{
		when PROFILE do spall.SCOPED_EVENT(&spall_ctx, &spall_buffer, name = "Present")
		flags: dxgi.PRESENT
		params: dxgi.PRESENT_PARAMETERS
		hr = dx_context.swapchain->Present1(1, flags, &params)
		check(hr, "Present failed")
	}

	// wait for frame to finish
	{
		when PROFILE do spall.SCOPED_EVENT(&spall_ctx, &spall_buffer, name = "v-sync wait")

		current_fence_value := dx_context.fence_value

		hr = dx_context.queue->Signal(dx_context.fence, current_fence_value)
		check(hr, "Failed to signal fence")

		dx_context.fence_value += 1
		completed := dx_context.fence->GetCompletedValue()

		if completed < current_fence_value {
			hr = dx_context.fence->SetEventOnCompletion(current_fence_value, dx_context.fence_event)
			check(hr, "Failed to set event on completion flag")
			windows.WaitForSingleObject(dx_context.fence_event, windows.INFINITE)
		}

		c.frame_index = c.swapchain->GetCurrentBackBufferIndex()

		// swap PSO here if needed (hot reload of shaders)

		// hot swap handling
		hotswap_swap(&c.lighting_hotswap, &c.pipeline_lighting)
		hotswap_swap(&c.gbuffer_hotswap, &c.pipeline_gbuffer)
	}
}

// creating resource and uploading it to the gpu
create_sample_texture :: proc() {

	ct := &dx_context

	// reading from image
	texture_width, texture_height, channels: c.int

	img_data := img.load("astrobot.png", &texture_width, &texture_height, &channels, 0)
	defer (img.image_free(img_data))

	if img_data == nil {
		fmt.eprintln("error reading image")
		os.exit(1)
	}

	// default heap (this is where the final texture will reside)

	heap_properties := dx.HEAP_PROPERTIES {
		Type = .DEFAULT,
	}
	texture_desc := dx.RESOURCE_DESC {
		Width = (u64)(texture_width),
		Height = (u32)(texture_height),
		Dimension = .TEXTURE2D,
		Layout = .UNKNOWN,
		Format = .R8G8B8A8_UNORM,
		DepthOrArraySize = 1,
		MipLevels = 1,
		SampleDesc = {Count = 1},
	}

	hr := dx_context.device->CreateCommittedResource(
		&heap_properties,
		dx.HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES,
		&texture_desc,
		{.COPY_DEST},
		nil,
		dx.IResource_UUID,
		(^rawptr)(&ct.texture),
	)

	check(hr, "failed creating texture")
	ct.texture->SetName("lucy's sample texture (it's now astrobot)")
	sa.push(&resources_longterm, ct.texture)

	// getting data from texture that we'll use later
	text_footprint: dx.PLACED_SUBRESOURCE_FOOTPRINT
	text_bytes: u64
	num_rows: u32
	row_size: u64

	dx_context.device->GetCopyableFootprints(
		&texture_desc,
		0,
		1,
		0,
		&text_footprint,
		&num_rows,
		&row_size,
		&text_bytes,
	)

	// creating upload heap and resource (needed to upload texture data from cpu to the default heap)

	heap_properties = dx.HEAP_PROPERTIES {
		Type = .UPLOAD,
	}

	texture_upload: ^dx.IResource
	upload_desc := dx.RESOURCE_DESC {
		Dimension = .BUFFER,
		Alignment = 0,
		Width = text_bytes, // size of the texture in bytes
		Height = 1,
		MipLevels = 1,
		Format = .UNKNOWN,
		Layout = .ROW_MAJOR,
		DepthOrArraySize = 1,
		SampleDesc = {Count = 1},
	}

	hr = dx_context.device->CreateCommittedResource(
		&heap_properties,
		dx.HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES,
		&upload_desc,
		dx.RESOURCE_STATE_GENERIC_READ,
		nil,
		dx.IResource_UUID,
		(^rawptr)(&texture_upload),
	)

	check(hr, "failed creating upload texture")
	texture_upload->SetName("lucy's texture upload resource, for tx uploading")
	defer texture_upload->Release()

	// here you do a Map and you memcpy the data to the upload resource.
	// you'll have to use an image library here to get the pixel data of an image.

	texture_map_start: rawptr
	texture_upload->Map(0, &dx.RANGE{}, &texture_map_start)
	texture_map_start_mp: [^]u8 = auto_cast texture_map_start

	for row in 0 ..< texture_height {
		mem.copy(
			texture_map_start_mp[u32(text_footprint.Footprint.RowPitch) * u32(row):],
			img_data[texture_width * channels * row:],
			int(texture_width * channels),
		)
	}

	// here you send the gpu command to copy the data to the texture resource.

	copy_location_src := dx.TEXTURE_COPY_LOCATION {
		pResource = texture_upload,
		Type = .PLACED_FOOTPRINT,
		PlacedFootprint = text_footprint,
	}

	copy_location_dst := dx.TEXTURE_COPY_LOCATION {
		pResource = ct.texture,
		Type = .SUBRESOURCE_INDEX,
		SubresourceIndex = 0,
	}

	dx_context.cmdlist->Reset(dx_context.command_allocator, ct.pipeline_gbuffer)
	dx_context.cmdlist->CopyTextureRegion(&copy_location_dst, 0, 0, 0, &copy_location_src, nil)

	transition_resource_from_copy_to_read(ct.texture, ct.cmdlist)

	execute_command_list_and_wait(ct.cmdlist, ct.queue)
}

gen_just_one_instance_data :: proc() -> []InstanceData {
	// returning one instance with no transformations
	// we're rendering sponza now. we only want one.


	instance_data := make([]InstanceData, 1, context.temp_allocator)

	world_mat: dxm
	world_mat = 1

	instance_data[0] = InstanceData {
		world_mat = world_mat,
		color = v3{1, 1, 1},
	}

	return instance_data
}

gen_teapot_instance_data :: proc() -> []InstanceData {
	teapot_count := 50

	instance_data := make([]InstanceData, teapot_count, context.temp_allocator)

	for &instance, i in instance_data {

		spread :: 10

		// fill it with random stuff
		x_pos := rand.float32_range(-spread, spread)
		y_pos := rand.float32_range(-spread, spread)
		z_pos := rand.float32_range(-spread, spread)

		scale := rand.float32_range(0.5, 5)

		rot_fac :: 1

		rot_val := rand.float32_range(0, math.TAU)
		rot_vec := v3 {
			rand.float32_range(-rot_fac, rot_fac),
			rand.float32_range(-rot_fac, rot_fac),
			rand.float32_range(-rot_fac, rot_fac),
		}

		rot_vec = linalg.vector_normalize(rot_vec)

		col_vec := v3{rand.float32_range(0.5, 1), rand.float32_range(0.5, 1), rand.float32_range(0.5, 1)}

		// x_pos = 0
		instance = InstanceData {
			world_mat = get_world_mat({x_pos, y_pos, z_pos}, {scale, scale, scale}, rot_val, rot_vec),
			color = col_vec,
		}
	}

	return instance_data
}

// creates instance buffer and fills it with some data
create_instance_buffer_example :: proc() -> VertexBuffer {

	// first: we create the data
	instance_data := gen_just_one_instance_data()
	// instance_data := gen_teapot_instance_data()

	// second: we create the DX buffer, passing the size we want. and stride
	//   it needs to return the vertex buffer view

	instance_data_size := len(instance_data) * size_of(instance_data[0])

	vb := create_vertex_buffer(size_of(instance_data[0]), u32(instance_data_size), pool = &resources_longterm)

	// third: we copy the data to the buffer (map and unmap)									
	copy_to_buffer(vb.buffer, &instance_data[0], instance_data_size)

	return vb
}

reroll_teapots :: proc() {

	instance_data := gen_teapot_instance_data()

	instance_data_size := len(instance_data) * size_of(instance_data[0])

	copy_to_buffer(dx_context.instance_buffer.buffer, &instance_data[0], instance_data_size)
}

// creates the descriptor heap that will hold all our cbv's srv's and uav's
// List of descriptors in it:
// - 0 - A test constant buffer (CBV). It has things like Time
// - 1 - A test texture (SRV). now, it's astrobot
// - 2 - A structured buffer (SRV). containing model matrices
create_descriptor_heap_cbv_srv_uav :: proc() {

	c := &dx_context

	// creating descriptor heap
	cbv_heap_desc := dx.DESCRIPTOR_HEAP_DESC {
		NumDescriptors = 3,
		Type = .CBV_SRV_UAV,
		Flags = {.SHADER_VISIBLE},
	}

	hr := c.device->CreateDescriptorHeap(
		&cbv_heap_desc,
		dx.IDescriptorHeap_UUID,
		(^rawptr)(&c.descriptor_heap_cbv_srv_uav),
	)
	check(hr, "failed creating descriptor heap")
	sa.push(&resources_longterm, c.descriptor_heap_cbv_srv_uav)

	c.descriptor_heap_cbv_srv_uav->SetName("lucy's cbv srv uav descriptor heap")

	// creating CBV for my test constant buffer (AT INDEX 0)
	cbv_desc := dx.CONSTANT_BUFFER_VIEW_DESC {
		BufferLocation = dx_context.constant_buffer->GetGPUVirtualAddress(),
		SizeInBytes = size_of(ConstantBufferData),
	}

	c.device->CreateConstantBufferView(&cbv_desc, get_descriptor_heap_cpu_address(c.descriptor_heap_cbv_srv_uav, 0))

	// creating SRV (my texture) (AT INDEX 1)
	c.device->CreateShaderResourceView(
		c.texture,
		nil,
		get_descriptor_heap_cpu_address(c.descriptor_heap_cbv_srv_uav, 1),
	)

	// creating SRV (structured buffer) (index 2)
	srv_desc := dx.SHADER_RESOURCE_VIEW_DESC {
		Format = .UNKNOWN,
		ViewDimension = .BUFFER,
		Shader4ComponentMapping = dx.ENCODE_SHADER_4_COMPONENT_MAPPING(0, 1, 2, 3), // this is the default mapping
		Buffer = {
			FirstElement = 0,
			NumElements = u32(scene.mesh_count),
			StructureByteStride = size_of(BufferThing),
			Flags = {},
		},
	}

	c.device->CreateShaderResourceView(
		c.res_structured_buffer,
		&srv_desc,
		get_descriptor_heap_cpu_address(c.descriptor_heap_cbv_srv_uav, 2),
	)
}

create_gbuffer_pass_root_signature :: proc() {

	root_parameters_len :: 3

	root_parameters: [root_parameters_len]dx.ROOT_PARAMETER

	// our test constant buffer
	root_parameters[0] = {
		ParameterType = .CBV,
		Descriptor = {ShaderRegister = 0, RegisterSpace = 0},
		ShaderVisibility = .ALL, // vertex, pixel, or both (all)
	}

	{
		// We'll define a descriptor range for our SRVs
		srv_range := dx.DESCRIPTOR_RANGE {
			RangeType = .SRV,
			NumDescriptors = 2,
			BaseShaderRegister = 1,
			RegisterSpace = 0,
			OffsetInDescriptorsFromTableStart = dx.DESCRIPTOR_RANGE_OFFSET_APPEND,
		}

		// our descriptor table for the texture
		root_parameters[1] = {
			ParameterType = .DESCRIPTOR_TABLE,
			DescriptorTable = {NumDescriptorRanges = 1, pDescriptorRanges = &srv_range},
			ShaderVisibility = .ALL, // TODO: look into separating them. one parameter for vertex. another for pixel.
		}
	}

	root_parameters[2] = {
		ParameterType = ._32BIT_CONSTANTS,
		Constants = {ShaderRegister = 1, RegisterSpace = 0, Num32BitValues = 1},
		ShaderVisibility = .VERTEX,
	}

	// our static sampler

	// We'll define a static sampler description
	sampler_desc := dx.STATIC_SAMPLER_DESC {
		Filter = .MIN_MAG_MIP_LINEAR, // Tri-linear filtering
		AddressU = .CLAMP, // Repeat the texture in the U direction
		AddressV = .CLAMP, // Repeat the texture in the V direction
		AddressW = .WRAP, // Repeat the texture in the W direction
		MipLODBias = 0.0,
		MaxAnisotropy = 0,
		ComparisonFunc = .NEVER,
		BorderColor = .OPAQUE_BLACK,
		MinLOD = 0.0,
		MaxLOD = dx.FLOAT32_MAX,
		ShaderRegister = 0, // This corresponds to the s0 register in the shader
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

	desc.Desc_1_0.Flags = {.ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT}
	serialized_desc: ^dx.IBlob
	hr := dx.SerializeVersionedRootSignature(&desc, &serialized_desc, nil)
	check(hr, "Failed to serialize root signature")
	hr = dx_context.device->CreateRootSignature(
		0,
		serialized_desc->GetBufferPointer(),
		serialized_desc->GetBufferSize(),
		dx.IRootSignature_UUID,
		(^rawptr)(&dx_context.gbuffer_pass_root_signature),
	)
	check(hr, "Failed creating root signature")
	sa.push(&resources_longterm, dx_context.gbuffer_pass_root_signature)
	serialized_desc->Release()
}


load_meshes :: proc(data: ^cgltf.data, vertices: ^[dynamic]VertexData, indices: ^[dynamic]u32) -> []Mesh {

	the_meshes := make_slice([]Mesh, len(data.meshes))
	index_count: u32
	index_count_total: u32 = 0

	for mesh, i in data.meshes {

		mesh_index_offset := index_count_total

		mesh_index_count: u32

		for prim in mesh.primitives {

			attr_position: cgltf.attribute
			attr_normal: cgltf.attribute
			attr_texcoord: cgltf.attribute

			for attribute in prim.attributes {
				#partial switch attribute.type {
				case .position:
					attr_position = attribute
				case .normal:
					attr_normal = attribute
				case .texcoord:
					attr_texcoord = attribute
				case:
				// it's outputting "unknown attribute COLOR_0" and it's annoying.
				//  so, i am commenting this error log.
				// fmt.eprintfln("Unkown gltf attribute: {}", attribute)
				}
			}

			for i in 0 ..< attr_position.data.count {
				vertex: VertexData
				ok: b32
				ok = cgltf.accessor_read_float(attr_position.data, i, &vertex.pos[0], 3)
				if !ok do fmt.eprintln("Error reading gltf position")
				ok = cgltf.accessor_read_float(attr_normal.data, i, &vertex.normal[0], 3)
				if !ok do fmt.eprintln("Error reading gltf normal")
				ok = cgltf.accessor_read_float(attr_texcoord.data, i, &vertex.uv[0], 2)
				if !ok do fmt.eprintln("Error reading gltf texcoord")

				position := v4{vertex.pos.x, vertex.pos.y, vertex.pos.z, 1}
				// vertex.pos = (mesh_mat * position).xyz
				vertex.pos = (position).xyz
				append(vertices, vertex)
				// vertices[i] = vertex
			}

			for i in 0 ..< prim.indices.count {
				append(indices, u32(cgltf.accessor_read_index(prim.indices, i)) + u32(index_count))
			}

			index_count += u32(attr_position.data.count)

			index_count_total += u32(prim.indices.count)
			mesh_index_count += u32(prim.indices.count)
		}

		the_meshes[i] = Mesh {
			index_offset = mesh_index_offset,
			index_count = mesh_index_count,
		}
	}

	return the_meshes
}

get_node_world_matrix :: proc(node: Node, scene: Scene) -> dxm {

	res: dxm = 1

	node_i := node

	for {

		boosted_t := node_i.transform_t * 1
		translation_mat := linalg.matrix4_translate_f32(boosted_t)

		boosted_s := node_i.transform_s * 1
		scale_mat := linalg.matrix4_scale_f32(boosted_s)

		rot_quat: quaternion128 = quaternion(
			w = node_i.transform_r[0],
			x = node_i.transform_r[1],
			y = node_i.transform_r[2],
			z = node_i.transform_r[3],
		)
		rot_mat := linalg.matrix4_from_quaternion_f32(rot_quat)

		// mesh_world : dxm = translation_mat * rot_mat * scale_mat
		// no rot
		mesh_world: dxm = translation_mat * rot_mat * scale_mat
		// mesh_world : dxm = scale_mat * rot_mat * translation_mat

		res = res * mesh_world
		// res = mesh_world * res
		// break

		if node_i.parent == -1 do break
		node_i = scene.nodes[node_i.parent]
	}

	return res
}

proc_walk :: proc(node: Node, scene: Scene, data: rawptr)

// Walks through the scene tree and runs a proc per node
scene_walk :: proc(scene: Scene, data: rawptr, thing_to_do: proc_walk) {
	nodes := scene.nodes

	for root_node in scene.root_nodes {
		node_i := scene.nodes[root_node]

		// algorithm state
		cur_child_i: uint = 0
		depth := 0
		child_i_levels: [10]uint
		children_are_explored: bool

		for {

			if !children_are_explored {

				// do the stuff here
				thing_to_do(node_i, scene, data)
			}

			if node_i.children == nil || children_are_explored {
				children_are_explored = false

				// go to next sibling
				cur_child_i += 1

				if node_i.parent == -1 {
					break
				}

				// if there is no next sibling, go up

				node_parent := nodes[node_i.parent]

				if cur_child_i >= len(node_parent.children) {

					depth -= 1

					// if the current's node's parent doesn't have a parent, we're done!
					if node_parent.parent == -1 {
						break
					}

					// check if this one has a sibling

					node_grandparent := nodes[node_parent.parent]

					node_i = nodes[node_grandparent.children[child_i_levels[depth]]]
					cur_child_i = child_i_levels[depth]
					children_are_explored = true
					continue
				}

				node_i = nodes[node_parent.children[cur_child_i]]
			} else {
				// go to first child
				child_i_levels[depth] = cur_child_i
				cur_child_i = 0
				node_i = nodes[node_i.children[cur_child_i]]
				depth += 1
			}
		}
	}
}

gltf_load_nodes :: proc(data: ^cgltf.data) -> Scene {

	// TODO: don't leak this
	nodes := make([]Node, len(data.nodes))
	root_node_count: int = 0

	for node, i in data.nodes {
		if node.parent == nil {
			root_node_count += 1
		}
	}

	// TODO: don't leak this
	root_nodes := make([]int, root_node_count)
	root_node_i := 0
	mesh_count: uint

	for node, i in data.nodes {
		// TODO: don't leak this
		node_children := make([]int, len(node.children))

		for n_child, i in node.children {
			node_children[i] = int(cgltf.node_index(data, n_child))
		}

		if node.mesh != nil {
			mesh_count += 1
		}

		nodes[i] = Node {
			name = strings.clone_from_cstring(node.name),
			transform_t = node.translation,
			transform_r = node.rotation,
			transform_s = node.scale,
			children = node_children,
			parent = node.parent == nil ? -1 : int(cgltf.node_index(data, node.parent)),
			mesh = node.mesh == nil ? -1 : int(cgltf.mesh_index(data, node.mesh)),
		}

		if node.parent == nil {
			root_nodes[root_node_i] = i
			root_node_i += 1
		}
	}

	return Scene{nodes = nodes, root_nodes = root_nodes, mesh_count = mesh_count}
}

do_gltf_stuff :: proc() -> (vertices: []VertexData, indices: []u32) {

	// model_filepath :: "models/teapot.glb"
	// model_filepath :: "models/main_sponza/NewSponza_Main_glTF_003.gltf"
	// model_filepath :: "models/test_scene.glb"
	// model_filepath :: "models/main_sponza/sponza_blender.glb"
	
	// no decals (ruins solid rendering)
	model_filepath :: "models/main_sponza/sponza_blender_no_decals.glb"
	
	model_filepath_c := strings.clone_to_cstring(model_filepath, context.temp_allocator)

	cgltf_options: cgltf.options

	data, ok := cgltf.parse_file(cgltf_options, model_filepath_c)
	defer cgltf.free(data)


	if ok != .success {
		fmt.eprintln("could not read glb")
	}

	load_buffers_result := cgltf.load_buffers(cgltf_options, data, model_filepath_c)
	if load_buffers_result != .success {
		fmt.eprintln("Error loading buffers from gltf: {} - {}", model_filepath, load_buffers_result)
	}

	// new stuff start

	// just taking the first scene

	assert(len(data.scenes) == 1)

	gltf_scene := data.scenes[0]

	vertices_dyn := make([dynamic]VertexData)
	indices_dyn := make([dynamic]u32)

	meshes = load_meshes(data, &vertices_dyn, &indices_dyn)

	fmt.printfln("mesh count: %v", len(meshes))

	scene = gltf_load_nodes(data)

	// printing nodes
	// scene_walk(scene, nil, proc(node: Node, scene: Scene, data: rawptr) {
	//     lprintfln("node name: %v", node.name)
	// })

	// for root_node in gltf_scene.nodes {
	//     gltf_load_nodes(data, root_node, &vertices_dyn, &indices_dyn)
	// }

	return vertices_dyn[:], indices_dyn[:]
}

create_depth_buffer :: proc() {

	c := &dx_context

	heap_properties := dx.HEAP_PROPERTIES {
		Type = .DEFAULT,
	}

	depth_stencil_desc := dx.RESOURCE_DESC {
		Dimension = .TEXTURE2D,
		Width = u64(wx),
		Height = u32(wy),
		DepthOrArraySize = 1,
		MipLevels = 1,
		Format = .D32_FLOAT,
		SampleDesc = {Count = 1},
		Layout = .UNKNOWN,
		Flags = {.ALLOW_DEPTH_STENCIL},
	}

	// define a clear value for the depth buffer
	opt_clear := dx.CLEAR_VALUE {
		Format = .D32_FLOAT,
		DepthStencil = {Depth = 1.0, Stencil = 0},
	}

	hr := c.device->CreateCommittedResource(
		&heap_properties,
		dx.HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES,
		&depth_stencil_desc,
		{.DEPTH_WRITE},
		&opt_clear,
		dx.IResource_UUID,
		(^rawptr)(&c.depth_stencil_res),
	)

	check(hr, "failed creating depth resource")
	c.depth_stencil_res->SetName("depth stencil texture")
	sa.push(&resources_longterm, c.depth_stencil_res)

	// depth stencil view descriptor heap

	// creating descriptor heap
	heap_desc := dx.DESCRIPTOR_HEAP_DESC {
		NumDescriptors = 1,
		Type = .DSV,
		Flags = {},
	}

	hr = c.device->CreateDescriptorHeap(&heap_desc, dx.IDescriptorHeap_UUID, (^rawptr)(&c.descriptor_heap_dsv))

	c.descriptor_heap_dsv->SetName("lucy's DSV (depth-stencil-view) descriptor heap")

	check(hr, "could not create descriptor heap for DSV")
	sa.push(&resources_longterm, c.descriptor_heap_dsv)

	// creating depth stencil view

	descriptor_handle: dx.CPU_DESCRIPTOR_HANDLE
	c.descriptor_heap_dsv->GetCPUDescriptorHandleForHeapStart(&descriptor_handle)

	dsv_desc := dx.DEPTH_STENCIL_VIEW_DESC {
		ViewDimension = .TEXTURE2D,
		Format = .D32_FLOAT,
	}

	c.device->CreateDepthStencilView(c.depth_stencil_res, &dsv_desc, descriptor_handle)
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

	imgui_impl_sdl2.InitForD3D(dx_context.window)


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

	c := &dx_context


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
		(^rawptr)(&dx_context.imgui_descriptor_heap),
	)
	check(hr, "could ont create imgui descriptor heap")
	dx_context.imgui_descriptor_heap->SetName("imgui's cbv srv uav descriptor heap")
	sa.push(&resources_longterm, dx_context.imgui_descriptor_heap)

	dx_context.imgui_allocator = descriptor_heap_allocator_create(dx_context.imgui_descriptor_heap, .CBV_SRV_UAV)

	allocfn := proc "c" (
		info: ^imgui_impl_dx12.InitInfo,
		out_cpu_desc_handle: ^dx.CPU_DESCRIPTOR_HANDLE,
		out_gpu_desc_handle: ^dx.GPU_DESCRIPTOR_HANDLE,
	) {
		context = runtime.default_context()
		cpu, gpu := descriptor_heap_allocator_alloc(&dx_context.imgui_allocator)
		out_cpu_desc_handle.ptr = cpu.ptr
		out_gpu_desc_handle.ptr = gpu.ptr
	}

	freefn := proc "c" (
		info: ^imgui_impl_dx12.InitInfo,
		cpu_desc_handle: dx.CPU_DESCRIPTOR_HANDLE,
		gpu_desc_handle: dx.GPU_DESCRIPTOR_HANDLE,
	) {
		context = runtime.default_context()
		descriptor_heap_allocator_free(&dx_context.imgui_allocator, cpu_desc_handle, gpu_desc_handle)
	}


	dx12_init := imgui_impl_dx12.InitInfo {
		Device = dx_context.device,
		CommandQueue = dx_context.queue,
		// not sure what this is
		NumFramesInFlight = 2,
		RTVFormat = .R8G8B8A8_UNORM,
		DSVFormat = .D32_FLOAT,
		SrvDescriptorHeap = dx_context.imgui_descriptor_heap,
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
imgui_update_after :: proc() {

	// setting imgui's descriptor heap
	// if i don't do this, it errors out. seems like RenderDrawData doesn't set it
	//  by itself
	dx_context.cmdlist->SetDescriptorHeaps(1, &dx_context.imgui_descriptor_heap)

	// need graphics command list
	imgui_impl_dx12.RenderDrawData(im.GetDrawData(), dx_context.cmdlist)

	io := im.GetIO()

	if .ViewportsEnable in io.ConfigFlags {
		im.UpdatePlatformWindows()
		im.RenderPlatformWindowsDefault()
	}
}


// helpers

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


get_descriptor_heap_cpu_address :: proc(
	heap: ^dx.IDescriptorHeap,
	offset: u32 = 0,
) -> (
	cpu_descriptor_handle: dx.CPU_DESCRIPTOR_HANDLE,
) {
	heap->GetCPUDescriptorHandleForHeapStart(&cpu_descriptor_handle)
	desc: dx.DESCRIPTOR_HEAP_DESC
	heap->GetDesc(&desc)
	increment := dx_context.device->GetDescriptorHandleIncrementSize(desc.Type)
	cpu_descriptor_handle.ptr += uint(offset * increment)
	return
}


// gives you a transformation matrix given a position and scale and rot
get_world_mat :: proc(pos, scale: v3, rot_rads: f32 = 0, rot_vec: v3 = {1, 0, 0}) -> dxm {


	translation_mat := linalg.matrix4_translate_f32(pos)
	scale_mat := linalg.matrix4_scale_f32(scale)

	rot_mat := linalg.matrix4_rotate_f32(rot_rads, rot_vec)

	return translation_mat * scale_mat * rot_mat
}

get_world_mat_quat :: proc(pos, scale: v3, rot_quat: quaternion128) -> dxm {

	translation_mat := linalg.matrix4_translate_f32(pos)
	scale_mat := linalg.matrix4_scale_f32(scale)

	rot_mat := linalg.matrix4_from_quaternion_f32(rot_quat)

	return translation_mat * scale_mat * rot_mat
}

// returns a vertex buffer view
create_vertex_buffer :: proc(stride_in_bytes, size_in_bytes: u32, pool: ^DXResourcePool) -> VertexBuffer {

	vb: ^dx.IResource

	// For now we'll just store stuff in an upload heap.
	// it's not optimal for most things but it's more practical for me
	heap_props := dx.HEAP_PROPERTIES {
		Type = .UPLOAD,
	}

	resource_desc := dx.RESOURCE_DESC {
		Dimension = .BUFFER,
		Alignment = 0,
		Width = u64(size_in_bytes),
		Height = 1,
		DepthOrArraySize = 1,
		MipLevels = 1,
		Format = .UNKNOWN,
		SampleDesc = {Count = 1, Quality = 0},
		Layout = .ROW_MAJOR,
		Flags = {},
	}

	hr := dx_context.device->CreateCommittedResource(
		&heap_props,
		{},
		&resource_desc,
		dx.RESOURCE_STATE_GENERIC_READ,
		nil,
		dx.IResource_UUID,
		(^rawptr)(&vb),
	)
	check(hr, "Failed creating vertex buffer")
	sa.push(pool, vb)

	vbv := dx.VERTEX_BUFFER_VIEW {
		BufferLocation = vb->GetGPUVirtualAddress(),
		StrideInBytes = stride_in_bytes,
		SizeInBytes = size_in_bytes,
	}

	return VertexBuffer {
		buffer = vb,
		vbv = vbv,
		vertex_count = size_in_bytes / stride_in_bytes,
		buffer_size = size_in_bytes,
		buffer_stride = stride_in_bytes,
	}
}

// copies data to a dx resource. then unmaps the memory
copy_to_buffer :: proc(buffer: ^dx.IResource, src: rawptr, len: int) {
	gpu_data: rawptr
	hr := buffer->Map(0, &dx.RANGE{}, &gpu_data)
	check(hr, "Failed mapping")
	mem.copy(gpu_data, src, len)
	buffer->Unmap(0, nil)
}

create_gbuffer :: proc() -> GBuffer {
	ct := &dx_context

	// creating rtv heap and srv heaps
	gb_rtv_dh: ^dx.IDescriptorHeap
	gb_srv_dh: ^dx.IDescriptorHeap

	desc := dx.DESCRIPTOR_HEAP_DESC {
		NumDescriptors = gbuffer_count,
		Type = .RTV,
		Flags = {},
	}

	hr := ct.device->CreateDescriptorHeap(&desc, dx.IDescriptorHeap_UUID, (^rawptr)(&gb_rtv_dh))
	check(hr, "Failed creating descriptor heap")
	gb_rtv_dh->SetName("lucy's g-buffer RTV descriptor heap")

	rtv_descriptor_size: u32 = ct.device->GetDescriptorHandleIncrementSize(.RTV)
	rtv_descriptor_handle_heap_start: dx.CPU_DESCRIPTOR_HANDLE
	gb_rtv_dh->GetCPUDescriptorHandleForHeapStart(&rtv_descriptor_handle_heap_start)

	// creating SRV descriptor heap

	desc = dx.DESCRIPTOR_HEAP_DESC {
		NumDescriptors = gbuffer_count,
		Type = .CBV_SRV_UAV,
		Flags = {.SHADER_VISIBLE},
	}

	hr = ct.device->CreateDescriptorHeap(&desc, dx.IDescriptorHeap_UUID, (^rawptr)(&gb_srv_dh))
	check(hr, "Failed creating descriptor heap")
	gb_rtv_dh->SetName("lucy's g-buffer CBV_SRV_UAV descriptor heap")

	// create texture resource and RTV's

	// TODO: look into creating a heap and resources separately.

	gb_albedo_format: dxgi.FORMAT = .R8G8B8A8_UNORM
	gb_normal_format: dxgi.FORMAT = .R10G10B10A2_UNORM
	gb_position_format: dxgi.FORMAT = .R16G16B16A16_FLOAT

	clear_value := dx.CLEAR_VALUE {
		Format = gb_albedo_format,
		Color = {0, 0, 0, 1.0},
	}

	// albedo color and specular
	gb_1_res := create_texture(
		u64(wx),
		u32(wy),
		gb_albedo_format,
		{.ALLOW_RENDER_TARGET},
		initial_state = {.PIXEL_SHADER_RESOURCE},
		pool = &resources_longterm,
	)

	gb_1_res->SetName("gbuffer unit 0: ALBEDO + SPECULAR")

	rtv_descriptor_handle_1: dx.CPU_DESCRIPTOR_HANDLE = rtv_descriptor_handle_heap_start
	rtv_descriptor_handle_1.ptr += uint(rtv_descriptor_size) * 0
	ct.device->CreateRenderTargetView(gb_1_res, nil, rtv_descriptor_handle_1)
	ct.device->CreateShaderResourceView(gb_1_res, nil, get_descriptor_heap_cpu_address(gb_srv_dh, 0))
	// u gotta release the whole heap

	// world normal data
	gb_2_res := create_texture(
		u64(wx),
		u32(wy),
		gb_normal_format,
		{.ALLOW_RENDER_TARGET},
		initial_state = {.PIXEL_SHADER_RESOURCE},
		pool = &resources_longterm,
	)
	gb_2_res->SetName("gbuffer unit 1: NORMAL")

	rtv_descriptor_handle_2: dx.CPU_DESCRIPTOR_HANDLE = rtv_descriptor_handle_heap_start
	rtv_descriptor_handle_2.ptr += uint(rtv_descriptor_size) * 1
	ct.device->CreateRenderTargetView(gb_2_res, nil, rtv_descriptor_handle_2)
	ct.device->CreateShaderResourceView(gb_2_res, nil, get_descriptor_heap_cpu_address(gb_srv_dh, 1))

	// world space position
	gb_3_res := create_texture(
		u64(wx),
		u32(wy),
		gb_position_format,
		{.ALLOW_RENDER_TARGET},
		initial_state = {.PIXEL_SHADER_RESOURCE},
		pool = &resources_longterm,
	)
	gb_3_res->SetName("gbuffer unit 2: WORLD SPACE POSITION")

	rtv_descriptor_handle_3: dx.CPU_DESCRIPTOR_HANDLE = rtv_descriptor_handle_heap_start
	rtv_descriptor_handle_3.ptr += uint(rtv_descriptor_size) * 2
	ct.device->CreateRenderTargetView(gb_3_res, nil, rtv_descriptor_handle_3)
	ct.device->CreateShaderResourceView(gb_3_res, nil, get_descriptor_heap_cpu_address(gb_srv_dh, 2))

	sa.push(&resources_longterm, gb_rtv_dh)
	sa.push(&resources_longterm, gb_srv_dh)

	return GBuffer {
		gb_albedo = GBufferUnit{res = gb_1_res, rtv = rtv_descriptor_handle_1, format = gb_albedo_format},
		gb_normal = GBufferUnit{res = gb_2_res, rtv = rtv_descriptor_handle_2, format = gb_normal_format},
		gb_position = GBufferUnit{res = gb_3_res, rtv = rtv_descriptor_handle_3, format = gb_position_format},
		rtv_heap = gb_rtv_dh,
		srv_heap = gb_srv_dh,
	}
}

// TODO organize this
BufferThing :: struct {
	model_matrix: dxm,
}


// this is the same as create_sample_texture
// it creates an upload heap, copies data to it, then transfers it to the default heap.
// make it specific for what u want now.
// then we can turn it into a helper function. later.
create_structured_buffer :: proc(pool: ^DXResourcePool) {

	ct := &dx_context



	buffer_size: u64 = size_of(BufferThing) * u64(scene.mesh_count)

	heap_properties := dx.HEAP_PROPERTIES {
		Type = .DEFAULT,
	}

	buffer_desc := dx.RESOURCE_DESC {
		Width = buffer_size,
		Height = 1,
		Dimension = .BUFFER,
		Layout = .ROW_MAJOR,
		Format = .UNKNOWN,
		DepthOrArraySize = 1,
		MipLevels = 1,
		SampleDesc = {Count = 1},
		Flags = {},
	}

	default_res: ^dx.IResource

	// CreateCommittedResource:          proc "system" (this: ^IDevice,
	//  pHeapProperties: ^HEAP_PROPERTIES, HeapFlags: HEAP_FLAGS, pDesc: ^RESOURCE_DESC, InitialResourceState: RESOURCE_STATES,
	//   pOptimizedClearValue: ^CLEAR_VALUE, riidResource: ^IID, ppvResource: ^rawptr) -> HRESULT,

	hr := ct.device->CreateCommittedResource(
		pHeapProperties = &heap_properties,
		HeapFlags = dx.HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES,
		pDesc = &buffer_desc,
		InitialResourceState = dx.RESOURCE_STATE_COMMON, // it will promote to copy_dest automatically later
		pOptimizedClearValue = nil,
		riidResource = dx.IResource_UUID,
		ppvResource = (^rawptr)(&default_res),
	)

	check(hr, "failed creating buffer")
	default_res->SetName("structured buffer - model matrices")
	sa.push(pool, default_res)

	// creating UPLOAD resource

	heap_properties.Type = .UPLOAD

	upload_res: ^dx.IResource

	// buffer desc is the same i think.
	// buffer_desc.

	hr = ct.device->CreateCommittedResource(
		&heap_properties,
		dx.HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES,
		&buffer_desc,
		dx.RESOURCE_STATE_GENERIC_READ,
		nil,
		dx.IResource_UUID,
		(^rawptr)(&upload_res),
	)

	check(hr, "failed creating upload buffer")
	upload_res->SetName("upload buffer for matrices")
	defer upload_res->Release()

	// Copying data from cpu to upload resource

	Data :: struct {
		sample_matrix_data: []BufferThing,
		mesh_i: uint,
	}

	data := Data {
		sample_matrix_data = make([]BufferThing, scene.mesh_count, allocator = context.temp_allocator),
		mesh_i = 0,
	}

	scene_walk(scene, &data, proc(node: Node, scene: Scene, data: rawptr) {
		if node.mesh == -1 do return
		data := cast(^Data)data
		data.sample_matrix_data[data.mesh_i].model_matrix = get_node_world_matrix(node, scene)
		data.mesh_i += 1
	})

	// TODO: check if this works.
	// if it works, make a helper method that just takes a slice. friendlier API
	copy_to_buffer(
		upload_res,
		&data.sample_matrix_data[0],
		size_of(data.sample_matrix_data[0]) * len(data.sample_matrix_data),
	)

	dx_context.cmdlist->Reset(ct.command_allocator, ct.pipeline_gbuffer)
	dx_context.cmdlist->CopyResource(default_res, upload_res)

	// transition resource to shader readable.

	transition_resource_from_copy_to_read(default_res, ct.cmdlist)

	execute_command_list_and_wait(ct.cmdlist, ct.queue)

	ct.res_structured_buffer = default_res
}

create_new_lighting_pso :: proc(root_signature: ^dx.IRootSignature, vs, ps: ^dxc.IBlob) -> ^dx.IPipelineState {

	c := &dx_context

	default_blend_state := dx.RENDER_TARGET_BLEND_DESC {
		BlendEnable = false,
		LogicOpEnable = false,
		SrcBlend = .ONE,
		DestBlend = .ZERO,
		BlendOp = .ADD,
		SrcBlendAlpha = .ONE,
		DestBlendAlpha = .ZERO,
		BlendOpAlpha = .ADD,
		LogicOp = .NOOP,
		RenderTargetWriteMask = u8(dx.COLOR_WRITE_ENABLE_ALL),
	}

	// the swapchain rtv
	rtv_formats := [8]dxgi.FORMAT {
		0 = .R8G8B8A8_UNORM,
		1 ..< 7 = .UNKNOWN,
	}

	pipeline_state_desc := dx.GRAPHICS_PIPELINE_STATE_DESC {
		pRootSignature = root_signature,
		VS = {pShaderBytecode = vs->GetBufferPointer(), BytecodeLength = vs->GetBufferSize()},
		PS = {pShaderBytecode = ps->GetBufferPointer(), BytecodeLength = ps->GetBufferSize()},
		StreamOutput = {},
		BlendState = {
			AlphaToCoverageEnable = false,
			IndependentBlendEnable = false,
			RenderTarget = {0 = default_blend_state, 1 ..< 7 = {}},
		},
		SampleMask = 0xFFFFFFFF,
		RasterizerState = {
			FillMode = .SOLID,
			CullMode = .BACK,
			FrontCounterClockwise = false,
			DepthBias = 0,
			DepthBiasClamp = 0,
			SlopeScaledDepthBias = 0,
			DepthClipEnable = true,
			MultisampleEnable = false,
			AntialiasedLineEnable = false,
			ForcedSampleCount = 0,
			ConservativeRaster = .OFF,
		},
		// enabling depth testing
		DepthStencilState = {DepthEnable = true, StencilEnable = false, DepthWriteMask = .ALL, DepthFunc = .LESS},
		// no input layout. we don't need a vertex buffer.
		InputLayout = {pInputElementDescs = nil, NumElements = 0},
		PrimitiveTopologyType = .TRIANGLE,
		NumRenderTargets = 1,
		RTVFormats = {0 = .R8G8B8A8_UNORM, 1 ..< 7 = .UNKNOWN},
		DSVFormat = .D32_FLOAT,
		SampleDesc = {Count = 1, Quality = 0},
	}

	pso: ^dx.IPipelineState

	hr := c.device->CreateGraphicsPipelineState(&pipeline_state_desc, dx.IPipelineState_UUID, (^rawptr)(&pso))
	check(hr, "Pipeline creation failed")
	pso->SetName("PSO for lighting pass")

	return pso
}


create_lighting_pso_initial :: proc() {

	c := &dx_context

	vs, ps, ok := compile_shader(c.dxc_compiler, lighting_shader_filename)

	if !ok {
		fmt.printfln("could not compile shader!! check logs")
		os.exit(1)
	}

	// create root signature
	create_lighting_root_signature()

	c.pipeline_lighting = create_new_lighting_pso(c.lighting_pass_root_signature, vs, ps)

	pso_index := sa.len(resources_longterm)
	sa.push(&resources_longterm, c.pipeline_lighting)

	hotswap_init(&c.lighting_hotswap, lighting_shader_filename, pso_index)

	vs->Release()
	ps->Release()
}

create_lighting_root_signature :: proc() {

	c := &dx_context


	// We'll define a descriptor range for our gbuffers
	texture_range := dx.DESCRIPTOR_RANGE {
		RangeType = .SRV,
		NumDescriptors = 3,
		BaseShaderRegister = 1, // Corresponds to t1 in the shader
		RegisterSpace = 0,
		OffsetInDescriptorsFromTableStart = dx.DESCRIPTOR_RANGE_OFFSET_APPEND,
	}

	root_parameters_len :: 2

	root_parameters: [root_parameters_len]dx.ROOT_PARAMETER

	// our test constant buffer
	root_parameters[0] = {
		ParameterType = .CBV,
		Descriptor = {ShaderRegister = 0, RegisterSpace = 0},
		ShaderVisibility = .ALL, // vertex, pixel, or both (all)
	}

	// our descriptor table for the texture
	root_parameters[1] = {
		ParameterType = .DESCRIPTOR_TABLE,
		DescriptorTable = {NumDescriptorRanges = 1, pDescriptorRanges = &texture_range},
		ShaderVisibility = .PIXEL,
	}

	// our static sampler

	// We'll define a static sampler description
	sampler_desc := dx.STATIC_SAMPLER_DESC {
		Filter = .MIN_MAG_MIP_LINEAR, // Tri-linear filtering
		AddressU = .CLAMP, // Repeat the texture in the U direction
		AddressV = .CLAMP, // Repeat the texture in the V direction
		AddressW = .WRAP, // Repeat the texture in the W direction
		MipLODBias = 0.0,
		MaxAnisotropy = 0,
		ComparisonFunc = .NEVER,
		BorderColor = .OPAQUE_BLACK,
		MinLOD = 0.0,
		MaxLOD = dx.FLOAT32_MAX,
		ShaderRegister = 0, // This corresponds to the s0 register in the shader
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

	// desc.Desc_1_0.Flags = {.ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT}
	desc.Desc_1_0.Flags = {}
	serialized_desc: ^dx.IBlob
	hr := dx.SerializeVersionedRootSignature(&desc, &serialized_desc, nil)
	check(hr, "Failed to serialize root signature")
	hr = c.device->CreateRootSignature(
		0,
		serialized_desc->GetBufferPointer(),
		serialized_desc->GetBufferSize(),
		dx.IRootSignature_UUID,
		(^rawptr)(&c.lighting_pass_root_signature),
	)
	check(hr, "Failed creating root signature")
	sa.push(&resources_longterm, c.lighting_pass_root_signature)
	serialized_desc->Release()
}

create_new_gbuffer_pso :: proc(root_signature: ^dx.IRootSignature, vs, ps: ^dxc.IBlob) -> ^dx.IPipelineState {

	c := &dx_context

	// This layout matches the vertices data defined further down
	// this has to include the instance data!!
	vertex_format := [?]dx.INPUT_ELEMENT_DESC {
		{
			SemanticName = "POSITION",
			Format = .R32G32B32_FLOAT,
			AlignedByteOffset = dx.APPEND_ALIGNED_ELEMENT,
			InputSlotClass = .PER_VERTEX_DATA,
		},
		{
			SemanticName = "NORMAL",
			Format = .R32G32B32_FLOAT,
			AlignedByteOffset = dx.APPEND_ALIGNED_ELEMENT,
			InputSlotClass = .PER_VERTEX_DATA,
		},
		{
			SemanticName = "TEXCOORD",
			Format = .R32G32_FLOAT,
			AlignedByteOffset = dx.APPEND_ALIGNED_ELEMENT,
			InputSlotClass = .PER_VERTEX_DATA,
		},
		// per-instance data
		{
			SemanticName = "WORLDMATRIX",
			SemanticIndex = 0,
			Format = .R32G32B32A32_FLOAT,
			InputSlot = 1,
			AlignedByteOffset = dx.APPEND_ALIGNED_ELEMENT,
			InputSlotClass = .PER_INSTANCE_DATA,
			InstanceDataStepRate = 1,
		},
		{
			SemanticName = "WORLDMATRIX",
			SemanticIndex = 1,
			Format = .R32G32B32A32_FLOAT,
			InputSlot = 1,
			AlignedByteOffset = dx.APPEND_ALIGNED_ELEMENT,
			InputSlotClass = .PER_INSTANCE_DATA,
			InstanceDataStepRate = 1,
		},
		{
			SemanticName = "WORLDMATRIX",
			SemanticIndex = 2,
			Format = .R32G32B32A32_FLOAT,
			InputSlot = 1,
			AlignedByteOffset = dx.APPEND_ALIGNED_ELEMENT,
			InputSlotClass = .PER_INSTANCE_DATA,
			InstanceDataStepRate = 1,
		},
		{
			SemanticName = "WORLDMATRIX",
			SemanticIndex = 3,
			Format = .R32G32B32A32_FLOAT,
			InputSlot = 1,
			AlignedByteOffset = dx.APPEND_ALIGNED_ELEMENT,
			InputSlotClass = .PER_INSTANCE_DATA,
			InstanceDataStepRate = 1,
		},
		{
			SemanticName = "COLOR",
			SemanticIndex = 0,
			Format = .R32G32B32_FLOAT,
			InputSlot = 1,
			AlignedByteOffset = dx.APPEND_ALIGNED_ELEMENT,
			InputSlotClass = .PER_INSTANCE_DATA,
			InstanceDataStepRate = 1,
		},
	}

	default_blend_state := dx.RENDER_TARGET_BLEND_DESC {
		BlendEnable = false,
		LogicOpEnable = false,
		SrcBlend = .ONE,
		DestBlend = .ZERO,
		BlendOp = .ADD,
		SrcBlendAlpha = .ONE,
		DestBlendAlpha = .ZERO,
		BlendOpAlpha = .ADD,
		LogicOp = .NOOP,
		RenderTargetWriteMask = u8(dx.COLOR_WRITE_ENABLE_ALL),
	}

	// all formats of the g buffers
	// HERE
	rtv_formats := [8]dxgi.FORMAT {
		0 = dx_context.gbuffer.gb_albedo.format,
		1 = dx_context.gbuffer.gb_normal.format,
		2 = dx_context.gbuffer.gb_position.format,
		3 ..< 7 = .UNKNOWN,
	}

	pipeline_state_desc := dx.GRAPHICS_PIPELINE_STATE_DESC {
		pRootSignature = root_signature,
		VS = {pShaderBytecode = vs->GetBufferPointer(), BytecodeLength = vs->GetBufferSize()},
		PS = {pShaderBytecode = ps->GetBufferPointer(), BytecodeLength = ps->GetBufferSize()},
		StreamOutput = {},
		BlendState = {
			AlphaToCoverageEnable = false,
			IndependentBlendEnable = false,
			RenderTarget = {0 ..< gbuffer_count = default_blend_state},
		},
		SampleMask = 0xFFFFFFFF,
		RasterizerState = {
			FillMode = .SOLID,
			CullMode = .BACK,
			FrontCounterClockwise = false,
			DepthBias = 0,
			DepthBiasClamp = 0,
			SlopeScaledDepthBias = 0,
			DepthClipEnable = true,
			MultisampleEnable = false,
			AntialiasedLineEnable = false,
			ForcedSampleCount = 0,
			ConservativeRaster = .OFF,
		},
		// enabling depth testing
		DepthStencilState = {DepthEnable = true, StencilEnable = false, DepthWriteMask = .ALL, DepthFunc = .LESS},
		InputLayout = {pInputElementDescs = &vertex_format[0], NumElements = u32(len(vertex_format))},
		PrimitiveTopologyType = .TRIANGLE,
		NumRenderTargets = gbuffer_count,
		RTVFormats = rtv_formats,
		DSVFormat = .D32_FLOAT,
		SampleDesc = {Count = 1, Quality = 0},
	}

	pso: ^dx.IPipelineState

	hr := c.device->CreateGraphicsPipelineState(&pipeline_state_desc, dx.IPipelineState_UUID, (^rawptr)(&pso))
	check(hr, "Pipeline creation failed")
	pso->SetName("PSO for gbuffer pass")

	return pso
}

// creates PSO for the first drawing pass that populates the gbuffer
create_gbuffer_pso_initial :: proc() {

	c := &dx_context

	vs, ps, ok := compile_shader(c.dxc_compiler, gbuffer_shader_filename)
	if !ok {
		fmt.printfln("could not compile shader!! check logs")
		os.exit(1)
	}

	c.pipeline_gbuffer = create_new_gbuffer_pso(c.gbuffer_pass_root_signature, vs, ps)

	pso_index := sa.len(resources_longterm)
	sa.push(&resources_longterm, c.pipeline_gbuffer)

	hotswap_init(&c.gbuffer_hotswap, gbuffer_shader_filename, pso_index)

	vs->Release()
	ps->Release()
}

render_gbuffer_pass :: proc() {

	c := &dx_context


	// This state is reset everytime the cmd list is reset, so we need to rebind it
	c.cmdlist->SetGraphicsRootSignature(dx_context.gbuffer_pass_root_signature)

	// setting descriptor heap for our cbv srv uav's
	c.cmdlist->SetDescriptorHeaps(1, &dx_context.descriptor_heap_cbv_srv_uav)

	// setting the root cbv that we set up in the root signature. root parameter 0
	c.cmdlist->SetGraphicsRootConstantBufferView(0, dx_context.constant_buffer->GetGPUVirtualAddress())

	// setting descriptor tables for our texture. root parameter 1
	{
		// setting the graphics root descriptor table
		// in the root signature, so that it points to
		// our SRV descriptor
		c.cmdlist->SetGraphicsRootDescriptorTable(
			1,
			get_descriptor_heap_gpu_address(dx_context.descriptor_heap_cbv_srv_uav, 1),
		)
	}

	{
		viewport := dx.VIEWPORT {
			Width = f32(wx),
			Height = f32(wy),
			MinDepth = 0,
			MaxDepth = 1,
		}

		scissor_rect := dx.RECT {
			left = 0,
			right = wx,
			top = 0,
			bottom = wy,
		}

		c.cmdlist->RSSetViewports(1, &viewport)
		c.cmdlist->RSSetScissorRects(1, &scissor_rect)
	}

	// Transitioning gbuffers from SRVs to render target
	{
		res_barriers: [3]dx.RESOURCE_BARRIER

		// res barrier template

		res_barriers[0] = dx.RESOURCE_BARRIER {
			Type = .TRANSITION,
			Flags = {},
			Transition = {
				pResource = nil,
				StateBefore = {.PIXEL_SHADER_RESOURCE},
				StateAfter = {.RENDER_TARGET},
				Subresource = dx.RESOURCE_BARRIER_ALL_SUBRESOURCES,
			},
		}

		// populating all res barriers with each gbuffer
		res_barriers[0] = res_barriers[0]
		res_barriers[0].Transition.pResource = c.gbuffer.gb_albedo.res

		res_barriers[1] = res_barriers[0]
		res_barriers[1].Transition.pResource = c.gbuffer.gb_normal.res

		res_barriers[2] = res_barriers[0]
		res_barriers[2].Transition.pResource = c.gbuffer.gb_position.res

		c.cmdlist->ResourceBarrier(3, &res_barriers[0])
	}

	// Setting render targets. Clearing DSV and RTV.
	{
		rtv_handles := [gbuffer_count]dx.CPU_DESCRIPTOR_HANDLE {
			dx_context.gbuffer.gb_albedo.rtv,
			dx_context.gbuffer.gb_normal.rtv,
			dx_context.gbuffer.gb_position.rtv,
		}
		dsv_handle := get_descriptor_heap_cpu_address(dx_context.descriptor_heap_dsv, 0)

		// setting depth buffer
		c.cmdlist->OMSetRenderTargets(gbuffer_count, &rtv_handles[0], false, &dsv_handle)

		// clear backbuffer
		clearcolor := [?]f32{0, 0, 0, 1.0}

		// we should probably clear each gbuffer individually to a sane value...
		c.cmdlist->ClearRenderTargetView(rtv_handles[0], &clearcolor, 0, nil)
		c.cmdlist->ClearRenderTargetView(rtv_handles[1], &clearcolor, 0, nil)
		c.cmdlist->ClearRenderTargetView(rtv_handles[2], &clearcolor, 0, nil)

		// clearing depth buffer
		c.cmdlist->ClearDepthStencilView(dsv_handle, {.DEPTH, .STENCIL}, 1.0, 0, 0, nil)
	}

	// draw call
	c.cmdlist->IASetPrimitiveTopology(.TRIANGLELIST)

	// binding vertex buffer view and instance buffer view
	vertex_buffers_views := [?]dx.VERTEX_BUFFER_VIEW{dx_context.vertex_buffer_view, dx_context.instance_buffer.vbv}

	c.cmdlist->IASetVertexBuffers(0, len(vertex_buffers_views), &vertex_buffers_views[0])
	c.cmdlist->IASetIndexBuffer(&c.index_buffer_view)

	// rendering each mesh individually
	// going through scene tree

	// drawing scene

	scene_walk(scene, nil, proc(node: Node, scene: Scene, data: rawptr) {
		ct := &dx_context

		if node.mesh == -1 {
			return
		}

		mesh_to_render := meshes[node.mesh]

		if g_mesh_drawn_count < ct.meshes_to_render {
			ct.cmdlist->SetGraphicsRoot32BitConstant(2, u32(g_mesh_drawn_count), 0)
			ct.cmdlist->DrawIndexedInstanced(mesh_to_render.index_count, 1, mesh_to_render.index_offset, 0, 0)
		}

		g_mesh_drawn_count += 1
	})
}

render_lighting_pass :: proc() {

	c := &dx_context

	c.cmdlist->SetPipelineState(c.pipeline_lighting)

	// Transitioning gbuffers from render target to SRVs
	{
		res_barriers: [3]dx.RESOURCE_BARRIER

		// res barrier template

		res_barriers[0] = dx.RESOURCE_BARRIER {
			Type = .TRANSITION,
			Flags = {},
			Transition = {
				pResource = nil,
				StateBefore = {.RENDER_TARGET},
				StateAfter = {.PIXEL_SHADER_RESOURCE},
				Subresource = dx.RESOURCE_BARRIER_ALL_SUBRESOURCES,
			},
		}

		// populating all res barriers with each gbuffer
		res_barriers[0] = res_barriers[0]
		res_barriers[0].Transition.pResource = c.gbuffer.gb_albedo.res

		res_barriers[1] = res_barriers[0]
		res_barriers[1].Transition.pResource = c.gbuffer.gb_normal.res

		res_barriers[2] = res_barriers[0]
		res_barriers[2].Transition.pResource = c.gbuffer.gb_position.res

		c.cmdlist->ResourceBarrier(3, &res_barriers[0])
	}

	// here u have to transition the swapchain buffer so it is a RT
	{
		to_render_target_barrier := dx.RESOURCE_BARRIER {
			Type = .TRANSITION,
			Flags = {},
			Transition = {
				pResource = dx_context.targets[dx_context.frame_index],
				StateBefore = dx.RESOURCE_STATE_PRESENT,
				StateAfter = {.RENDER_TARGET},
				Subresource = dx.RESOURCE_BARRIER_ALL_SUBRESOURCES,
			},
		}

		c.cmdlist->ResourceBarrier(1, &to_render_target_barrier)
	}

	// This state is reset everytime the cmd list is reset, so we need to rebind it
	c.cmdlist->SetGraphicsRootSignature(c.lighting_pass_root_signature)


	// setting the root cbv that we set up in the root signature. root parameter 0
	c.cmdlist->SetGraphicsRootConstantBufferView(0, dx_context.constant_buffer->GetGPUVirtualAddress())

	// setting descriptor tables for our gbuffers
	{

		// setting descriptor heap for our cbv srv uav's
		c.cmdlist->SetDescriptorHeaps(1, &c.gbuffer.srv_heap)

		// setting the graphics root descriptor table
		// in the root signature, so that it points to
		// our SRV descriptor
		c.cmdlist->SetGraphicsRootDescriptorTable(1, get_descriptor_heap_gpu_address(c.gbuffer.srv_heap, 0))

		// c.gbuffer.rtv_heap
	}

	{
		viewport := dx.VIEWPORT {
			Width = f32(wx),
			Height = f32(wy),
			MinDepth = 0,
			MaxDepth = 1,
		}

		scissor_rect := dx.RECT {
			left = 0,
			right = wx,
			top = 0,
			bottom = wy,
		}

		c.cmdlist->RSSetViewports(1, &viewport)
		c.cmdlist->RSSetScissorRects(1, &scissor_rect)
	}

	// Setting render targets. Clearing DSV and RTV.
	{
		rtv_handles := [1]dx.CPU_DESCRIPTOR_HANDLE {
			get_descriptor_heap_cpu_address(c.swapchain_rtv_descriptor_heap, c.frame_index),
		}

		dsv_handle := get_descriptor_heap_cpu_address(dx_context.descriptor_heap_dsv, 0)

		// setting depth buffer
		c.cmdlist->OMSetRenderTargets(1, &rtv_handles[0], false, &dsv_handle)

		// clear backbuffer
		clearcolor := [?]f32{0.05, 0.05, 0.05, 1.0}

		// we should probably clear each gbuffer individually to a sane value...
		c.cmdlist->ClearRenderTargetView(rtv_handles[0], &clearcolor, 0, nil)

		// clearing depth buffer
		c.cmdlist->ClearDepthStencilView(dsv_handle, {.DEPTH, .STENCIL}, 1.0, 0, 0, nil)
	}

	// draw call
	c.cmdlist->IASetPrimitiveTopology(.TRIANGLELIST)

	// 3. Draw 3 vertices (which triggers the VS 3 times)
	c.cmdlist->DrawInstanced(3, 1, 0, 0)

	imgui_update_after()
}

print_ref_count :: proc(obj: ^dx.IUnknown) {
	obj->AddRef()
	count := obj->Release()
	fmt.printfln("count: %v", count)
}

// Prints to windows debug, with a fmt.println() interface
lprintln_donotuse :: proc(args: ..any, sep := " ") {
	str: strings.Builder
	strings.builder_init(&str, context.temp_allocator)
	final_string := fmt.sbprintln(&str, ..args, sep = sep)
	final_string_c, err := strings.to_cstring(&str)

	if err != .None {
		os.exit(1)
	}

	windows.OutputDebugStringA(final_string_c)
}

lprintfln_donotuse :: proc(fmt_s: string, args: ..any) {
	str: strings.Builder
	strings.builder_init(&str, context.temp_allocator)
	final_string := fmt.sbprintf(&str, fmt_s, ..args, newline = true)

	final_string_c, err := strings.to_cstring(&str)

	if err != .None {
		os.exit(1)
	}

	windows.OutputDebugStringA(final_string_c)
}

// Automatic profiling of every procedure:

when PROFILE {

	@(instrumentation_enter)
	spall_enter :: proc "contextless" (
		proc_address, call_site_return_address: rawptr,
		loc: runtime.Source_Code_Location,
	) {
		spall._buffer_begin(&spall_ctx, &spall_buffer, "", "", loc)
	}

	@(instrumentation_exit)
	spall_exit :: proc "contextless" (
		proc_address, call_site_return_address: rawptr,
		loc: runtime.Source_Code_Location,
	) {
		spall._buffer_end(&spall_ctx, &spall_buffer)
	}

}

pso_creation_signature :: proc(root_signature: ^dx.IRootSignature, vs, ps: ^dxc.IBlob) -> ^dx.IPipelineState

// checks if it should rebuild a shader
// if it should then compiles the new shader and makes a new PSO with it
hotswap_watch :: proc(
	hs: ^HotSwapState,
	root_signature: ^dx.IRootSignature,
	shader_name: string,
	pso_creation_proc: pso_creation_signature,
) {

	// watch for shader change
	game_dll_mod, game_dll_mod_err := os.last_write_time_by_name(shader_name)

	reload := false

	if game_dll_mod_err == os.ERROR_NONE && hs.last_write_time != game_dll_mod {
		hs.last_write_time = game_dll_mod
		reload = true
	}

	if reload {
		fmt.println("Recompiling shader...")
		// handle releasing resources
		vs, ps, ok := compile_shader(dx_context.dxc_compiler, shader_name)
		if !ok {
			fmt.println("Could not compile new shader!! check logs")
		} else {
			// create the new PSO to be swapped later
			hs.pso_swap = pso_creation_proc(root_signature, vs, ps)
			vs->Release()
			ps->Release()
		}
	}
}

hotswap_init :: proc(hs: ^HotSwapState, shader_filename: string, index_in_free_queue: int) {
	game_dll_mod, game_dll_mod_err := os.last_write_time_by_name(shader_filename)
	if game_dll_mod_err == os.ERROR_NONE {
		hs.last_write_time = game_dll_mod
	}
	hs.pso_index = index_in_free_queue
}

hotswap_swap :: proc(hs: ^HotSwapState, pso: ^^dx.IPipelineState) {
	if hs.pso_swap != nil {
		pso^->Release()
		pso^ = hs.pso_swap
		// replace pointer from freeing queue
		pso_pointer := sa.get_ptr(&resources_longterm, hs.pso_index)
		pso_pointer^ = pso^
		hs.pso_swap = nil
	}
}
