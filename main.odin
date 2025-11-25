package main

import "core:fmt"
import "core:mem"
import "core:os"
import "core:slice"
import "core:strings"
import "core:sys/windows"
import "core:time"
import dx "vendor:directx/d3d12"
import d3dc "vendor:directx/d3d_compiler"
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
import "core:prof/spall"
import "core:sync"

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

dxm :: matrix[4,4]f32

DXResourcePool :: sa.Small_Array(100, ^dx.IUnknown)

gbuffer_shader_filename :: "shader.hlsl"
lighting_shader_filename :: "lighting.hlsl"

gbuffer_count :: 3

// ---- all state ----

// profiling stuff

when PROFILE {
    spall_ctx: spall.Context
    @(thread_local) spall_buffer: spall.Buffer
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
resources_longterm : DXResourcePool


// constant buffer data
ConstantBufferData :: struct #align (256) {
	view: dxm,
	projection: dxm,
	light_pos: v3,
	light_int: f32,
	view_pos: v3,
	time: f32,
	place_texture: b32
}

// struct that holds instance data, for an instance rendering example
InstanceData :: struct #align (256) {
	world_mat : dxm,
	color: v3,
}

cb_update :: proc () {

	// ticking cbv time value
	thetime := time.diff(start_time, time.now())
	the_time_sec = f32(thetime) / f32(time.Second)
	// if the_time_sec > 1 {
	// 	start_time = time.now()
	// }

	// sending constant buffer data
	cam_pos := get_cam_pos()
	view, projection := get_view_projection(cam_pos)

	cbv_data := ConstantBufferData{
		view = view,
		projection = projection,
		light_pos = light_pos,
		light_int = light_int,
		view_pos = cam_pos,
		time = the_time_sec,
		place_texture = b32(place_texture)
	}

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
	format: dxgi.FORMAT
}

GBuffer :: struct {
	gb_albedo: GBufferUnit,
	gb_normal: GBufferUnit,
	gb_position: GBufferUnit,

	rtv_heap : ^dx.IDescriptorHeap,
	srv_heap : ^dx.IDescriptorHeap,
}

HotSwapState :: struct {
    last_write_time : os.File_Time,
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
	device:              ^dx.IDevice,
	factory:             ^dxgi.IFactory4,
	queue:               ^dx.ICommandQueue,
	swapchain:           ^dxgi.ISwapChain3,
	command_allocator:   ^dx.ICommandAllocator,

	pipeline_gbuffer:            ^dx.IPipelineState,

	cmdlist:             ^dx.IGraphicsCommandList,
	constant_buffer_map: rawptr, //maps to our test constant buffer
	gbuffer_pass_root_signature:      ^dx.IRootSignature,
	constant_buffer:     ^dx.IResource,
	vertex_buffer_view:  dx.VERTEX_BUFFER_VIEW,
	index_buffer_view: dx.INDEX_BUFFER_VIEW,
	// descriptor heap for the render target view
	swapchain_rtv_descriptor_heap: ^dx.IDescriptorHeap, 
	frame_index:         u32,
	targets:             [NUM_RENDERTARGETS]^dx.IResource, // render targets
	gbuffer: GBuffer,

	// lighting pass resources
	pipeline_lighting:            ^dx.IPipelineState,
	lighting_pass_root_signature:      ^dx.IRootSignature,

	// fence stuff (for waiting to render frame)
	fence:               ^dx.IFence,
	fence_value:         u64,
	fence_event:         windows.HANDLE,

	// texture
	texture : ^dx.IResource,

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

	// app data

	cam_angle: f32,
	cam_distance: f32,
	
	// hot swap shader state
	lighting_hotswap : HotSwapState,
	gbuffer_hotswap : HotSwapState, // todo this one (make helper functions for setting initial state and swapping code)
}

// initializes app data in Context struct
context_init :: proc(con: ^Context) {
	con.cam_angle = 0.080
	con.cam_distance = 2.320
	light_pos = v3{4.1,3.5,4.5}
	light_int = 1
	light_speed = 0.002
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
	context_init(&dx_context)

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
			Type           = .RTV,
			Flags          = {},
		}

		hr =
		device->CreateDescriptorHeap(
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
			hr =
			dx_context.swapchain->GetBuffer(
				i,
				dx.IResource_UUID,
				(^rawptr)(&dx_context.targets[i]),
			)
			dx_context.targets[i]->Release()
			check(hr, "Failed getting render target")
			device->CreateRenderTargetView(dx_context.targets[i], nil, rtv_descriptor_handle)
			rtv_descriptor_handle.ptr += uint(rtv_descriptor_size)
		}
	}

	// creating depth buffer

	// The command allocator is used to create the commandlist that is used to tell the GPU what to draw
	hr =
	device->CreateCommandAllocator(
		.DIRECT,
		dx.ICommandAllocator_UUID,
		(^rawptr)(&dx_context.command_allocator),
	)
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

		hr =
		device->CreateCommittedResource(
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
	hr =
	device->CreateCommandList(
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

	// texture test
	create_sample_texture()

	create_descriptor_heap_cbv_srv_uav()

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

		hr =
		device->CreateCommittedResource(
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
			StrideInBytes  = u32(vertex_buffer_size) / dx_context.vertex_count,
			SizeInBytes    = u32(vertex_buffer_size),
		}

		// creating index buffer resource

		index_buffer_size := len(indices) * size_of(indices[0])
		dx_context.index_count = u32(len(indices))

		resource_desc.Width = u64(index_buffer_size)

		hr = device->CreateCommittedResource(
			&heap_props, {}, &resource_desc, dx.RESOURCE_STATE_GENERIC_READ,
			nil, dx.IResource_UUID, (^rawptr)(&index_buffer)
		)
		check(hr, "failed index buffer")
		index_buffer->SetName("lucy's index buffer")
		sa.push(&resources_longterm, index_buffer)

		dx_context.index_buffer_view = dx.INDEX_BUFFER_VIEW {
			BufferLocation = index_buffer->GetGPUVirtualAddress(),
			SizeInBytes = u32(index_buffer_size),
			Format = .R16_UINT
		}

		hr = index_buffer->Map(0, &dx.RANGE{}, &gpu_data)
		check(hr, "failed mapping")

		mem.copy(gpu_data, &indices[0], index_buffer_size)
		index_buffer->Unmap(0, nil)
	}

	// This fence is used to wait for frames to finish
	{
		hr =
		device->CreateFence(
			dx_context.fence_value,
			{},
			dx.IFence_UUID,
			(^rawptr)(&dx_context.fence),
		)
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
	    windows.OutputDebugStringA("=== report start =====\n")
		debug_device : ^dx.IDebugDevice2
		dx_context.device->QueryInterface(dx.IDebugDevice2_UUID,
					(^rawptr)(&debug_device))
		// Finally, release the device (it is not in any pool)
		// The device will be freed after we release the debug device
		dx_context.device->Release()
		debug_device->ReportLiveDeviceObjects({.DETAIL, .IGNORE_INTERNAL})
		debug_device->Release()
		
		// DXGI report
		dxgi_debug : ^dxgi.IDebug1
		dxgi.DXGIGetDebugInterface1(0, dxgi.IDebug1_UUID, (^rawptr)(&dxgi_debug))
		dxgi_debug->ReportLiveObjects(dxgi.DEBUG_ALL, {})
		// TODO: make a function that prints to the debugger but works just like printfln
		windows.OutputDebugStringA("=== report end =====\n")
	}
}

do_main_loop :: proc() {
    main_loop: for {
        when PROFILE do spall.SCOPED_EVENT(&spall_ctx, &spall_buffer, name="main loop")
		for e: sdl.Event; sdl.PollEvent(&e); {
   
			imgui_impl_sdl2.ProcessEvent(&e)
   
			#partial switch e.type {
			case .QUIT:
				break main_loop
			case .WINDOWEVENT:
				if e.window.event == .CLOSE{
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
}

get_projection_matrix :: proc(fov_rad: f32, screenWidth: i32, screenHeight: i32, near: f32, far: f32) -> dxm {
    f := math.tan_f32(fov_rad * 0.5)

    aspect := f32(screenWidth) / f32(screenHeight)

	return dxm {
        aspect / f, 0.0, 0.0, 0.0,
        0.0, 1 / f, 0.0, 0.0,
        0.0, 0.0, far / (far - near), - (near * far) / (far - near),
        0.0, 0.0, 1.0, 0.0,
    }
}

get_cam_pos :: proc() -> v3 {
	cam_pos : v3

	cam_pos.z = -dx_context.cam_distance

	// rotate on Y axis

	rot_mat := linalg.matrix3_rotate_f32(dx_context.cam_angle * TURNS_TO_RAD, {0,1,0})
	return rot_mat * cam_pos
}

get_view_projection :: proc(cam_pos: v3) -> (dxm, dxm) {

	view := linalg.matrix4_look_at_f32(cam_pos, {0,0,0}, {0,1,0}, false)

	fov := linalg.to_radians(f32(90.0))
    aspect := f32(wx) / f32(wy)
	proj := linalg.matrix4_perspective_f32(fov, aspect, 0.1, 100, false)

	// this function is supposedly more correct
	// has correct depth values
	// proj := matrix4_perspective_z0_f32(fov, aspect, 0.1, 100)

	return view, proj
	// return view * proj
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
	
	hotswap_watch(&c.lighting_hotswap, c.lighting_pass_root_signature, lighting_shader_filename, is_lighting_pass = true)
	hotswap_watch(&c.gbuffer_hotswap, c.gbuffer_pass_root_signature, gbuffer_shader_filename, is_lighting_pass = false)
	
	// im.End()
	// 
	
	// imgui stuff

	im.Begin("lucydx12")

	im.SliderFloat("camera angle", &dx_context.cam_angle, 0, 1)
	im.SliderFloat("camera distance", &dx_context.cam_distance, 0.5, 20)

	im.DragFloat3("light pos", &light_pos, 0.1, -5, 5)
	im.DragFloat("light intensity", &light_int, 0.1, 0, 20)
	im.DragFloat("light speed", &light_speed, 0.0001, 0, 20)

	im.Checkbox("place texture", &place_texture)
	if im.Button("Re-roll teapots") {
		reroll_teapots()
	}
}

render :: proc() {

	c := &dx_context

	hr: dx.HRESULT

	cb_update()

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
				pResource   = dx_context.targets[dx_context.frame_index],
				StateBefore = {.RENDER_TARGET},
				StateAfter  = dx.RESOURCE_STATE_PRESENT,
				Subresource = dx.RESOURCE_BARRIER_ALL_SUBRESOURCES,
			}
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
	    when PROFILE do spall.SCOPED_EVENT(&spall_ctx, &spall_buffer, name="Present")
		flags: dxgi.PRESENT
		params: dxgi.PRESENT_PARAMETERS
		hr = dx_context.swapchain->Present1(1, flags, &params)
		check(hr, "Present failed")
	}

	// wait for frame to finish
	{
	    when PROFILE do spall.SCOPED_EVENT(&spall_ctx, &spall_buffer, name="v-sync wait")
					
		current_fence_value := dx_context.fence_value

		hr = dx_context.queue->Signal(dx_context.fence, current_fence_value)
		check(hr, "Failed to signal fence")

		dx_context.fence_value += 1
		completed := dx_context.fence->GetCompletedValue()

		if completed < current_fence_value {
			hr =
			dx_context.fence->SetEventOnCompletion(current_fence_value, dx_context.fence_event)
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
	defer(img.image_free(img_data))

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
	num_rows : u32
	row_size : u64

	dx_context.device->GetCopyableFootprints( &texture_desc, 0, 1, 0, &text_footprint, &num_rows, &row_size, &text_bytes)

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

	hr =
	dx_context.device->CreateCommittedResource(
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
	texture_map_start_mp : [^]u8 = auto_cast texture_map_start

	for row in 0..<texture_height {
		mem.copy(texture_map_start_mp[u32(text_footprint.Footprint.RowPitch) * u32(row):],
				 img_data[texture_width * channels * row:],
				  int(texture_width * channels))
	}

	// here you send the gpu command to copy the data to the texture resource.

	copy_location_src := dx.TEXTURE_COPY_LOCATION {
		pResource       = texture_upload,
		Type            = .PLACED_FOOTPRINT,
		PlacedFootprint = text_footprint,
	}

	copy_location_dst := dx.TEXTURE_COPY_LOCATION {
		pResource        = ct.texture,
		Type             = .SUBRESOURCE_INDEX,
		SubresourceIndex = 0,
	}

	dx_context.cmdlist->Reset(dx_context.command_allocator, ct.pipeline_gbuffer)
	dx_context.cmdlist->CopyTextureRegion(&copy_location_dst, 0, 0, 0, &copy_location_src, nil)


	// TODO: do a fence here, wait for it, then release the upload resource, and change the texture state to generic read

	fence_value: u64
	fence: ^dx.IFence
	hr = dx_context.device->CreateFence(fence_value, {}, dx.IFence_UUID, (^rawptr)(&fence))
	fence_value += 1
	sa.push(&resources_longterm, fence)

	// you need to set a resource barrier here to transition the texture resource to a generic read state
	// read gemini answer: https://gemini.google.com/app/b17b9b13fb300d60

	barrier : dx.RESOURCE_BARRIER = {
		Type = .TRANSITION,
		Flags = {},
		Transition = {
			pResource = ct.texture,
			StateBefore = {.COPY_DEST},
			StateAfter = dx.RESOURCE_STATE_GENERIC_READ,
			Subresource = 0
		}
	}

	// run resource barrier
	dx_context.cmdlist->ResourceBarrier(1, &barrier)


	// close command list and execute
	dx_context.cmdlist->Close()
	cmdlists := [?]^dx.IGraphicsCommandList{dx_context.cmdlist}
	dx_context.queue->ExecuteCommandLists(len(cmdlists), (^^dx.ICommandList)(&cmdlists[0]))

	// we signal only after executing the command list.
	// otherwise we are not sure that the gpu is done with the upload resource.
	hr = dx_context.queue->Signal(fence, fence_value)

	// 4. Wait for the GPU to reach the signal point.
	// First, create an event handle.
	fence_event := windows.CreateEventW(nil, false, false, nil)

	if fence_event == nil {
		fmt.println("Failed to create fence event")
		return
	}

	completed := fence->GetCompletedValue()

	if completed < fence_value {
		// the gpu is not finished yet , so we wait
		fence->SetEventOnCompletion(fence_value, fence_event)
		windows.WaitForSingleObject(fence_event, windows.INFINITE)
	}

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
		rot_vec := v3{rand.float32_range(-rot_fac, rot_fac), 
						rand.float32_range(-rot_fac, rot_fac), rand.float32_range(-rot_fac, rot_fac)}

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
	instance_data := gen_teapot_instance_data()

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
create_descriptor_heap_cbv_srv_uav :: proc() {

	c := &dx_context

	// creating descriptor heap
	cbv_heap_desc := dx.DESCRIPTOR_HEAP_DESC {
		NumDescriptors = 2,
		Type           = .CBV_SRV_UAV,
		Flags          = {.SHADER_VISIBLE},
	}

	hr := c.device->CreateDescriptorHeap(&cbv_heap_desc, dx.IDescriptorHeap_UUID,
		 (^rawptr)(&c.descriptor_heap_cbv_srv_uav))
	check(hr, "failed creating descriptor heap")
	sa.push(&resources_longterm, c.descriptor_heap_cbv_srv_uav)

	c.descriptor_heap_cbv_srv_uav->SetName("lucy's cbv srv uav descriptor heap")

	// creating CBV for my test constant buffer (AT INDEX 0)
	cbv_desc := dx.CONSTANT_BUFFER_VIEW_DESC {
		BufferLocation = dx_context.constant_buffer->GetGPUVirtualAddress(),
		SizeInBytes    = size_of(ConstantBufferData),
	}

	c.device->CreateConstantBufferView(&cbv_desc, 
		get_descriptor_heap_cpu_address(c.descriptor_heap_cbv_srv_uav, 0))

	// creating SRV (my texture) (AT INDEX 1)
	c.device->CreateShaderResourceView(c.texture, nil, 
		get_descriptor_heap_cpu_address(c.descriptor_heap_cbv_srv_uav, 1))
}

create_gbuffer_pass_root_signature :: proc() {

	// We'll define a descriptor range for our texture SRV
	texture_range := dx.DESCRIPTOR_RANGE {
		RangeType = .SRV,
		NumDescriptors = 1, // Only one texture
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
		DescriptorTable = {
			NumDescriptorRanges = 1,
			pDescriptorRanges = &texture_range
		},
		ShaderVisibility = .PIXEL
	}

	// our static sampler

	// We'll define a static sampler description
	sampler_desc := dx.STATIC_SAMPLER_DESC {
		Filter = .MIN_MAG_MIP_LINEAR, // Tri-linear filtering
		AddressU = .CLAMP,           // Repeat the texture in the U direction
		AddressV = .CLAMP,           // Repeat the texture in the V direction
		AddressW = .WRAP,           // Repeat the texture in the W direction
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
			pStaticSamplers = &sampler_desc
		},
	}

	desc.Desc_1_0.Flags = {.ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT}
	serialized_desc: ^dx.IBlob
	hr := dx.SerializeVersionedRootSignature(&desc, &serialized_desc, nil)
	check(hr, "Failed to serialize root signature")
	hr =
	dx_context.device->CreateRootSignature(
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

do_gltf_stuff :: proc() -> (vertices: []VertexData, indices: []u16) {

	model_filepath :: "models/teapot.glb"
	model_filepath_c := strings.clone_to_cstring(model_filepath, context.temp_allocator)

	cgltf_options : cgltf.options

	data, ok := cgltf.parse_file(cgltf_options, model_filepath_c)
	defer cgltf.free(data)


	if ok != .success {
		fmt.eprintln("could not read glb")
	}

	load_buffers_result := cgltf.load_buffers(cgltf_options, data, model_filepath_c)
	if load_buffers_result != .success {
		fmt.eprintln("Error loading buffers from gltf: {} - {}", model_filepath, load_buffers_result)
	}

	// extracting mesh

	mesh := data.nodes[0].mesh
	assert(len(mesh.primitives) == 1)
	primitive := mesh.primitives[0]

	attr_position: cgltf.attribute
	attr_normal: cgltf.attribute
	attr_texcoord: cgltf.attribute
	
	for attribute in primitive.attributes {
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

	assert(attr_position.data.count == attr_normal.data.count)
	assert(attr_position.data.count == attr_texcoord.data.count)

	// if attr_position.data == nil || primitive.indices == nil do return {}, {}

	vertices = make([]VertexData, attr_position.data.count)
	indices = make([]u16, primitive.indices.count)

	// mesh_mat := linalg.matrix4_from_quaternion_f32(rotation)
	// mesh_mat *= linalg.matrix4_rotate_f32(math.to_radians_f32(180), {0, 1, 0})
	// mesh_mat *= linalg.matrix4_translate_f32(translation)

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
		vertices[i] = vertex
	}

	for i in 0 ..< primitive.indices.count {
		indices[i] = u16(cgltf.accessor_read_index(primitive.indices, i))
	}

	return
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
		Flags = {.ALLOW_DEPTH_STENCIL}
	}

	// define a clear value for the depth buffer
	opt_clear := dx.CLEAR_VALUE {
		Format = .D32_FLOAT,
		DepthStencil = {
			Depth = 1.0,
			Stencil = 0
		}
	}

	hr := c.device->CreateCommittedResource(
		&heap_properties,
		dx.HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES,
		&depth_stencil_desc,
		{.DEPTH_WRITE},
		&opt_clear,
		dx.IResource_UUID,
		(^rawptr)(&c.depth_stencil_res)
	)

	check(hr, "failed creating depth resource")
	c.depth_stencil_res->SetName("depth stencil texture")
	sa.push(&resources_longterm, c.depth_stencil_res)

	// depth stencil view descriptor heap

	// creating descriptor heap
	heap_desc := dx.DESCRIPTOR_HEAP_DESC {
		NumDescriptors = 1,
		Type           = .DSV,
		Flags          = {},
	}

	hr = c.device->CreateDescriptorHeap(&heap_desc, 
		dx.IDescriptorHeap_UUID, (^rawptr)(&c.descriptor_heap_dsv))

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
		Type           = .CBV_SRV_UAV,
		Flags          = {.SHADER_VISIBLE},
	}

	hr := c.device->CreateDescriptorHeap(&srv_descriptor_heap_desc,
		 dx.IDescriptorHeap_UUID, (^rawptr)(&dx_context.imgui_descriptor_heap))
	check(hr, "could ont create imgui descriptor heap")
	dx_context.imgui_descriptor_heap->SetName("imgui's cbv srv uav descriptor heap")
	sa.push(&resources_longterm, dx_context.imgui_descriptor_heap)

	dx_context.imgui_allocator = descriptor_heap_allocator_create(dx_context.imgui_descriptor_heap, .CBV_SRV_UAV)

	allocfn := proc "c" (info: ^imgui_impl_dx12.InitInfo, out_cpu_desc_handle: ^dx.CPU_DESCRIPTOR_HANDLE, out_gpu_desc_handle: ^dx.GPU_DESCRIPTOR_HANDLE) {
		context = runtime.default_context()
		cpu, gpu := descriptor_heap_allocator_alloc(&dx_context.imgui_allocator)
		out_cpu_desc_handle.ptr = cpu.ptr
		out_gpu_desc_handle.ptr = gpu.ptr
	}

	freefn := proc "c" (info: ^imgui_impl_dx12.InitInfo, cpu_desc_handle: dx.CPU_DESCRIPTOR_HANDLE, gpu_desc_handle: dx.GPU_DESCRIPTOR_HANDLE) {
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

get_descriptor_heap_gpu_address :: proc(heap: ^dx.IDescriptorHeap, offset: u32 = 0) -> 
		(gpu_descriptor_handle : dx.GPU_DESCRIPTOR_HANDLE){
	heap->GetGPUDescriptorHandleForHeapStart(&gpu_descriptor_handle)
	desc : dx.DESCRIPTOR_HEAP_DESC
	heap->GetDesc(&desc)
	increment := dx_context.device->GetDescriptorHandleIncrementSize(desc.Type)
	gpu_descriptor_handle.ptr += u64(offset * increment)
	return
}


get_descriptor_heap_cpu_address :: proc(heap: ^dx.IDescriptorHeap, offset: u32 = 0) -> 
		(cpu_descriptor_handle : dx.CPU_DESCRIPTOR_HANDLE){
	heap->GetCPUDescriptorHandleForHeapStart(&cpu_descriptor_handle)
	desc : dx.DESCRIPTOR_HEAP_DESC
	heap->GetDesc(&desc)
	increment := dx_context.device->GetDescriptorHandleIncrementSize(desc.Type)
	cpu_descriptor_handle.ptr += uint(offset * increment)
	return
}


// gives you a transformation matrix given a position and scale and rot
get_world_mat :: proc(pos, scale: v3, rot_rads : f32, rot_vec: v3) -> dxm {


	translation_mat := linalg.matrix4_translate_f32(pos)
	scale_mat := linalg.matrix4_scale_f32(scale)

	rot_mat := linalg.matrix4_rotate_f32(rot_rads, rot_vec)

	// TODO: rotation mat.


	return translation_mat * scale_mat * rot_mat
	// return scale_mat * translation_mat
}

// returns a vertex buffer view
create_vertex_buffer :: proc(stride_in_bytes, size_in_bytes: u32, pool: ^DXResourcePool) -> VertexBuffer {

	vb : ^dx.IResource

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

	hr :=
	dx_context.device->CreateCommittedResource(
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
		StrideInBytes  = stride_in_bytes,
		SizeInBytes    = size_in_bytes
	}

	return VertexBuffer {
		buffer = vb,
		vbv = vbv,
		vertex_count = size_in_bytes / stride_in_bytes,
		buffer_size = size_in_bytes,
		buffer_stride = stride_in_bytes
	}
}

// copies data to a dx resource. then unmaps the memory
copy_to_buffer :: proc(buffer: ^dx.IResource, src: rawptr, len:int) {
	gpu_data: rawptr
	hr := buffer->Map(0, &dx.RANGE{}, &gpu_data)
	check(hr, "Failed mapping")
	mem.copy(gpu_data, src, len)
	buffer->Unmap(0, nil)
}

create_gbuffer :: proc() -> GBuffer {
	ct := &dx_context

	// creating rtv heap and srv heaps
	gb_rtv_dh : ^dx.IDescriptorHeap
	gb_srv_dh : ^dx.IDescriptorHeap

	desc := dx.DESCRIPTOR_HEAP_DESC {
		NumDescriptors = gbuffer_count,
		Type           = .RTV,
		Flags          = {},
	}

	hr :=
	ct.device->CreateDescriptorHeap(
		&desc,
		dx.IDescriptorHeap_UUID,
		(^rawptr)(&gb_rtv_dh),
	)
	check(hr, "Failed creating descriptor heap")
	gb_rtv_dh->SetName("lucy's g-buffer RTV descriptor heap")

	rtv_descriptor_size: u32 = ct.device->GetDescriptorHandleIncrementSize(.RTV)
	rtv_descriptor_handle_heap_start: dx.CPU_DESCRIPTOR_HANDLE
	gb_rtv_dh->GetCPUDescriptorHandleForHeapStart(&rtv_descriptor_handle_heap_start)

	// creating SRV descriptor heap

	desc = dx.DESCRIPTOR_HEAP_DESC {
		NumDescriptors = gbuffer_count,
		Type           = .CBV_SRV_UAV,
		Flags          = {.SHADER_VISIBLE},
	}

	hr =
	ct.device->CreateDescriptorHeap(
		&desc,
		dx.IDescriptorHeap_UUID,
		(^rawptr)(&gb_srv_dh),
	)
	check(hr, "Failed creating descriptor heap")
	gb_rtv_dh->SetName("lucy's g-buffer CBV_SRV_UAV descriptor heap")

	// create texture resource and RTV's

	// TODO: look into creating a heap and resources separately.

	gb_albedo_format : dxgi.FORMAT = .R8G8B8A8_UNORM
	gb_normal_format : dxgi.FORMAT = .R10G10B10A2_UNORM
	gb_position_format : dxgi.FORMAT = .R16G16B16A16_FLOAT

	clear_value := dx.CLEAR_VALUE {
		Format = gb_albedo_format,
		Color = {0,0,0,1.0}
	}

	// albedo color and specular
	gb_1_res := create_texture(u64(wx), u32(wy), gb_albedo_format, {.ALLOW_RENDER_TARGET}, 
			initial_state = {.PIXEL_SHADER_RESOURCE},
			pool = &resources_longterm)

	gb_1_res->SetName("gbuffer unit 0: ALBEDO + SPECULAR")

	rtv_descriptor_handle_1 : dx.CPU_DESCRIPTOR_HANDLE = rtv_descriptor_handle_heap_start
	rtv_descriptor_handle_1.ptr += uint(rtv_descriptor_size) * 0
	ct.device->CreateRenderTargetView(gb_1_res, nil, rtv_descriptor_handle_1)
	ct.device->CreateShaderResourceView(gb_1_res, nil, 
		get_descriptor_heap_cpu_address(gb_srv_dh, 0))
	// u gotta release the whole heap

	// world normal data
	gb_2_res := create_texture(u64(wx), u32(wy), gb_normal_format, {.ALLOW_RENDER_TARGET},
		initial_state = {.PIXEL_SHADER_RESOURCE},
		pool = &resources_longterm
	)
	gb_2_res->SetName("gbuffer unit 1: NORMAL")

	rtv_descriptor_handle_2 : dx.CPU_DESCRIPTOR_HANDLE = rtv_descriptor_handle_heap_start
	rtv_descriptor_handle_2.ptr += uint(rtv_descriptor_size) * 1
	ct.device->CreateRenderTargetView(gb_2_res, nil, rtv_descriptor_handle_2)
	ct.device->CreateShaderResourceView(gb_2_res, nil, 
		get_descriptor_heap_cpu_address(gb_srv_dh, 1))

	// world space position
	gb_3_res := create_texture(u64(wx), u32(wy), gb_position_format, {.ALLOW_RENDER_TARGET}, 
		initial_state = {.PIXEL_SHADER_RESOURCE},
		pool = &resources_longterm
	)
	gb_3_res->SetName("gbuffer unit 2: WORLD SPACE POSITION")

	rtv_descriptor_handle_3: dx.CPU_DESCRIPTOR_HANDLE = rtv_descriptor_handle_heap_start
	rtv_descriptor_handle_3.ptr += uint(rtv_descriptor_size) * 2
	ct.device->CreateRenderTargetView(gb_3_res, nil, rtv_descriptor_handle_3)
	ct.device->CreateShaderResourceView(gb_3_res, nil, 
		get_descriptor_heap_cpu_address(gb_srv_dh, 2))

	sa.push(&resources_longterm, gb_rtv_dh)
	sa.push(&resources_longterm, gb_srv_dh)

	return GBuffer {

		gb_albedo = GBufferUnit {
			res = gb_1_res,
			rtv = rtv_descriptor_handle_1,
			format = gb_albedo_format
		},

		gb_normal = GBufferUnit {
			res = gb_2_res,
			rtv = rtv_descriptor_handle_2,
			format = gb_normal_format,
		},

		gb_position = GBufferUnit {
			res = gb_3_res,
			rtv = rtv_descriptor_handle_3,
			format = gb_position_format
		},

		rtv_heap = gb_rtv_dh,
		srv_heap = gb_srv_dh
	}
}

// helper function that creates a texture resource in its own implicit heap
// TODO: look into creating heap separately
create_texture :: proc(width: u64, height: u32, format: dxgi.FORMAT, resource_flags:dx.RESOURCE_FLAGS, 
		initial_state: dx.RESOURCE_STATES,
		opt_clear_value: dx.CLEAR_VALUE = {},
		set_clear_value_to_zero : bool = true,
		pool : ^DXResourcePool
	) ->
		(res: ^dx.IResource){

	ct := &dx_context

	opt_clear_value := opt_clear_value

	heap_properties := dx.HEAP_PROPERTIES {
		Type = .DEFAULT,
	}

	if set_clear_value_to_zero {
		opt_clear_value = dx.CLEAR_VALUE {
			Format = format,
			Color = {0,0,0,1}
		}
	}

	texture_desc := dx.RESOURCE_DESC {
		Width = width,
		Height = height,
		Dimension = .TEXTURE2D,
		Layout = .UNKNOWN,
		Format = format,
		DepthOrArraySize = 1,
		MipLevels = 1,
		SampleDesc = {Count = 1},
		Flags = resource_flags,
	}

	hr := ct.device->CreateCommittedResource(
		&heap_properties,
		dx.HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES,
		&texture_desc,
		initial_state,
		&opt_clear_value,
		dx.IResource_UUID,
		(^rawptr)(&res),
	)
	
	sa.push(pool, res)

	return res
}

create_new_lighting_pso :: proc(root_signature: ^dx.IRootSignature, vs, ps: ^d3dc.ID3D10Blob) -> ^dx.IPipelineState {
    
    c := &dx_context
    
    
    default_blend_state := dx.RENDER_TARGET_BLEND_DESC {
		BlendEnable           = false,
		LogicOpEnable         = false,
		SrcBlend              = .ONE,
		DestBlend             = .ZERO,
		BlendOp               = .ADD,
		SrcBlendAlpha         = .ONE,
		DestBlendAlpha        = .ZERO,
		BlendOpAlpha          = .ADD,
		LogicOp               = .NOOP,
		RenderTargetWriteMask = u8(dx.COLOR_WRITE_ENABLE_ALL),
	}
   
	// the swapchain rtv
	rtv_formats := [8]dxgi.FORMAT {
		0 =.R8G8B8A8_UNORM,
		1..<7 = .UNKNOWN,
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
		DepthStencilState = {
			DepthEnable = true,
			StencilEnable = false,
			DepthWriteMask = .ALL,
			DepthFunc = .LESS,
		},
		// no input layout. we don't need a vertex buffer.
		InputLayout = {
			pInputElementDescs = nil,
			NumElements = 0
		},
		PrimitiveTopologyType = .TRIANGLE,
		NumRenderTargets = 1,
		RTVFormats = {0 = .R8G8B8A8_UNORM, 1 ..< 7 = .UNKNOWN},
		DSVFormat = .D32_FLOAT,
		SampleDesc = {Count = 1, Quality = 0},
	}
	
	pso : ^dx.IPipelineState
   
	hr :=
	c.device->CreateGraphicsPipelineState(
		&pipeline_state_desc,
		dx.IPipelineState_UUID,
		(^rawptr)(&pso),
	)
	check(hr, "Pipeline creation failed")
	pso->SetName("PSO for lighting pass")
	
	return pso
}


create_lighting_pso_initial :: proc() {
	
	c := &dx_context

	vs, ps, ok := compile_shader(lighting_shader_filename)
	
	if !ok {
        lprintfln("could not compile shader!! check logs")
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
		DescriptorTable = {
			NumDescriptorRanges = 1,
			pDescriptorRanges = &texture_range
		},
		ShaderVisibility = .PIXEL
	}

	// our static sampler

	// We'll define a static sampler description
	sampler_desc := dx.STATIC_SAMPLER_DESC {
		Filter = .MIN_MAG_MIP_LINEAR, // Tri-linear filtering
		AddressU = .CLAMP,           // Repeat the texture in the U direction
		AddressV = .CLAMP,           // Repeat the texture in the V direction
		AddressW = .WRAP,           // Repeat the texture in the W direction
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
			pStaticSamplers = &sampler_desc
		},
	}

	// desc.Desc_1_0.Flags = {.ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT}
	desc.Desc_1_0.Flags = {}
	serialized_desc: ^dx.IBlob
	hr := dx.SerializeVersionedRootSignature(&desc, &serialized_desc, nil)
	check(hr, "Failed to serialize root signature")
	hr =
	c.device->CreateRootSignature(
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

// compiles vertex and pixel shader
compile_shader :: proc(shader_filename: string) -> (vs, ps: ^d3dc.ID3D10Blob, ok: bool) {

	c := &dx_context

	data, ok_f := os.read_entire_file(shader_filename)

	if !ok_f {
		lprintfln("could not read file")
		os.exit(1)
	}

	defer(delete(data))
	data_size: uint = len(data)

	compile_flags: u32 = 0
	when ODIN_DEBUG {
		compile_flags |= u32(d3dc.D3DCOMPILE.DEBUG)
		compile_flags |= u32(d3dc.D3DCOMPILE.SKIP_OPTIMIZATION)
	}

	// errors
	vs_res: ^d3dc.ID3DBlob
	ps_res: ^d3dc.ID3DBlob

	hr := d3dc.Compile(
		rawptr(&data[0]), data_size, nil, nil, nil, "VSMain", "vs_4_0",
		compile_flags, 0, &vs, &vs_res,
	)

	if (vs_res != nil) {
		// errors in shader compilation
		a := strings.string_from_ptr(
			(^u8)(vs_res->GetBufferPointer()),
			int(vs_res->GetBufferSize()),
		)
		lprintfln("DXC VS ERRORS in %s: %s", shader_filename, a)
	}

	if (hr < 0) {
	    // vertex shader is worng
	    // something went wrong
	    return vs, ps, false
	}

	hr = d3dc.Compile(
		rawptr(&data[0]), data_size, nil, nil, nil, "PSMain", "ps_4_0",
		compile_flags, 0, &ps, &ps_res
	)

	if (ps_res != nil) {
		// errors in shader compilation
		a := strings.string_from_ptr(
			(^u8)(ps_res->GetBufferPointer()),
			int(ps_res->GetBufferSize()),
		)
		lprintfln("DXC PS ERRORS in %s: %s", shader_filename, a)
	}
	
	if (hr < 0) {
	    // pixel shader is wrong
	    return vs, ps, false
	}

	return vs, ps, true
}

create_new_gbuffer_pso :: proc(root_signature: ^dx.IRootSignature, vs, ps: ^d3dc.ID3D10Blob) -> ^dx.IPipelineState {
    
    c := &dx_context
    
    // This layout matches the vertices data defined further down
	// this has to include the instance data!!
	vertex_format :=  [?]dx.INPUT_ELEMENT_DESC {
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
			InstanceDataStepRate = 1
		},
		{
			SemanticName = "WORLDMATRIX",
			SemanticIndex = 1,
			Format = .R32G32B32A32_FLOAT,
			InputSlot = 1,
			AlignedByteOffset = dx.APPEND_ALIGNED_ELEMENT,
			InputSlotClass = .PER_INSTANCE_DATA,
			InstanceDataStepRate = 1
		},
		{
			SemanticName = "WORLDMATRIX",
			SemanticIndex = 2,
			Format = .R32G32B32A32_FLOAT,
			InputSlot = 1,
			AlignedByteOffset = dx.APPEND_ALIGNED_ELEMENT,
			InputSlotClass = .PER_INSTANCE_DATA,
			InstanceDataStepRate = 1
		},
		{
			SemanticName = "WORLDMATRIX",
			SemanticIndex = 3,
			Format = .R32G32B32A32_FLOAT,
			InputSlot = 1,
			AlignedByteOffset = dx.APPEND_ALIGNED_ELEMENT,
			InputSlotClass = .PER_INSTANCE_DATA,
			InstanceDataStepRate = 1
		},
		{
			SemanticName = "COLOR",
			SemanticIndex = 0,
			Format = .R32G32B32_FLOAT,
			InputSlot = 1,
			AlignedByteOffset = dx.APPEND_ALIGNED_ELEMENT,
			InputSlotClass = .PER_INSTANCE_DATA,
			InstanceDataStepRate = 1
		},
	}
   
	default_blend_state := dx.RENDER_TARGET_BLEND_DESC {
		BlendEnable           = false,
		LogicOpEnable         = false,
		SrcBlend              = .ONE,
		DestBlend             = .ZERO,
		BlendOp               = .ADD,
		SrcBlendAlpha         = .ONE,
		DestBlendAlpha        = .ZERO,
		BlendOpAlpha          = .ADD,
		LogicOp               = .NOOP,
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
			RenderTarget = {0..<gbuffer_count = default_blend_state}
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
		DepthStencilState = {
			DepthEnable = true,
			StencilEnable = false,
			DepthWriteMask = .ALL,
			DepthFunc = .LESS,
		},
		InputLayout = {
			pInputElementDescs = &vertex_format[0],
			NumElements = u32(len(vertex_format)),
		},
		PrimitiveTopologyType = .TRIANGLE,
		NumRenderTargets = gbuffer_count,
		RTVFormats = rtv_formats,
		DSVFormat = .D32_FLOAT,
		SampleDesc = {Count = 1, Quality = 0},
	}
	
	pso : ^dx.IPipelineState
   
	hr :=
	c.device->CreateGraphicsPipelineState(
		&pipeline_state_desc,
		dx.IPipelineState_UUID,
		(^rawptr)(&pso),
	)
	check(hr, "Pipeline creation failed")
	pso->SetName("PSO for gbuffer pass")
	
	return pso
}

// creates PSO for the first drawing pass that populates the gbuffer
create_gbuffer_pso_initial :: proc() {

	c := &dx_context

	vs, ps, ok := compile_shader(gbuffer_shader_filename)
	if !ok {
        lprintfln("could not compile shader!! check logs")
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
	c.cmdlist->SetGraphicsRootConstantBufferView(
		0,
		dx_context.constant_buffer->GetGPUVirtualAddress(),
	)

	// setting descriptor tables for our texture. root parameter 1
	{
		// setting the graphics root descriptor table
		// in the root signature, so that it points to
		// our SRV descriptor
		c.cmdlist->SetGraphicsRootDescriptorTable(1, 
			get_descriptor_heap_gpu_address(dx_context.descriptor_heap_cbv_srv_uav, 1)
		)
	}

	{
		viewport := dx.VIEWPORT {
			Width  = f32(wx),
			Height = f32(wy),
			MinDepth = 0,
			MaxDepth = 1
		}

		scissor_rect := dx.RECT {
			left   = 0,
			right  = wx,
			top    = 0,
			bottom = wy,
		}

		c.cmdlist->RSSetViewports(1, &viewport)
		c.cmdlist->RSSetScissorRects(1, &scissor_rect)
	}

	// TODO: transition the g buffers here instead of the swapchain buffers!!!

	// Transitioning gbuffers from SRVs to render target
	{
		res_barriers : [3]dx.RESOURCE_BARRIER

		// res barrier template

		res_barriers[0] = dx.RESOURCE_BARRIER {
			Type  = .TRANSITION,
			Flags = {},
			Transition = {
					pResource   = nil,
					StateBefore = {.PIXEL_SHADER_RESOURCE},
					StateAfter  = {.RENDER_TARGET},
					Subresource = dx.RESOURCE_BARRIER_ALL_SUBRESOURCES,
			}
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
	vertex_buffers_views := [?]dx.VERTEX_BUFFER_VIEW {
		dx_context.vertex_buffer_view,
		dx_context.instance_buffer.vbv,
	}

	c.cmdlist->IASetVertexBuffers(0, len(vertex_buffers_views), &vertex_buffers_views[0])
	c.cmdlist->IASetIndexBuffer(&c.index_buffer_view)
	c.cmdlist->DrawIndexedInstanced(c.index_count, 
					c.instance_buffer.vertex_count, 0, 0, 0)

}

render_lighting_pass :: proc() {

	c := &dx_context

	c.cmdlist->SetPipelineState(c.pipeline_lighting)

	// Transitioning gbuffers from render target to SRVs
	{
		res_barriers : [3]dx.RESOURCE_BARRIER

		// res barrier template

		res_barriers[0] = dx.RESOURCE_BARRIER {
			Type  = .TRANSITION,
			Flags = {},
			Transition = {
					pResource   = nil,
					StateBefore  = {.RENDER_TARGET},
					StateAfter = {.PIXEL_SHADER_RESOURCE},
					Subresource = dx.RESOURCE_BARRIER_ALL_SUBRESOURCES,
			}
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
			Type  = .TRANSITION,
			Flags = {},
			Transition = {
				pResource   = dx_context.targets[dx_context.frame_index],
				StateBefore = dx.RESOURCE_STATE_PRESENT,
				StateAfter  = {.RENDER_TARGET},
				Subresource = dx.RESOURCE_BARRIER_ALL_SUBRESOURCES,
			}
		}

		c.cmdlist->ResourceBarrier(1, &to_render_target_barrier)
	}

	// TODO: bind and draw here!!!

	// This state is reset everytime the cmd list is reset, so we need to rebind it
	c.cmdlist->SetGraphicsRootSignature(c.lighting_pass_root_signature)


	// setting the root cbv that we set up in the root signature. root parameter 0
	c.cmdlist->SetGraphicsRootConstantBufferView(
		0,
		dx_context.constant_buffer->GetGPUVirtualAddress(),
	)

	// setting descriptor tables for our gbuffers
	{

		// setting descriptor heap for our cbv srv uav's
		c.cmdlist->SetDescriptorHeaps(1, &c.gbuffer.srv_heap)

		// setting the graphics root descriptor table
		// in the root signature, so that it points to
		// our SRV descriptor
		c.cmdlist->SetGraphicsRootDescriptorTable(1, 
			get_descriptor_heap_gpu_address(c.gbuffer.srv_heap, 0)
		)

		// c.gbuffer.rtv_heap
	}

	{
		viewport := dx.VIEWPORT {
			Width  = f32(wx),
			Height = f32(wy),
			MinDepth = 0,
			MaxDepth = 1
		}

		scissor_rect := dx.RECT {
			left   = 0,
			right  = wx,
			top    = 0,
			bottom = wy,
		}

		c.cmdlist->RSSetViewports(1, &viewport)
		c.cmdlist->RSSetScissorRects(1, &scissor_rect)
	}

	// Setting render targets. Clearing DSV and RTV.
	{
		rtv_handles := [1]dx.CPU_DESCRIPTOR_HANDLE {
			get_descriptor_heap_cpu_address(c.swapchain_rtv_descriptor_heap, c.frame_index)
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
	c.cmdlist->DrawInstanced(3,1,0,0)

	imgui_update_after()
}

print_ref_count :: proc(obj: ^dx.IUnknown) {
    obj->AddRef()
    count := obj->Release()
    fmt.printfln("count: %v", count)
}

// Prints to windows debug, with a fmt.println() interface
lprintln :: proc(args: ..any, sep := " ") {
	str: strings.Builder
	strings.builder_init(&str, context.temp_allocator)
	final_string := fmt.sbprintln(&str, ..args, sep=sep)
	final_string_c, err := strings.to_cstring(&str)
	
	if err != .None {
	    os.exit(1)
	}
	
	 windows.OutputDebugStringA(final_string_c)
}

lprintfln :: proc(fmt_s: string, args: ..any) {
    str: strings.Builder
	strings.builder_init(&str, context.temp_allocator)
	final_string := fmt.sbprintf(&str, fmt_s, ..args, newline=true)
	
	final_string_c, err := strings.to_cstring(&str)
	
	if err != .None {
	    os.exit(1)
	}
	
	windows.OutputDebugStringA(final_string_c)
}

// Automatic profiling of every procedure:

when PROFILE {
    
@(instrumentation_enter)
spall_enter :: proc "contextless" (proc_address, call_site_return_address: rawptr, loc: runtime.Source_Code_Location) {
	spall._buffer_begin(&spall_ctx, &spall_buffer, "", "", loc)
}

@(instrumentation_exit)
spall_exit :: proc "contextless" (proc_address, call_site_return_address: rawptr, loc: runtime.Source_Code_Location) {
	spall._buffer_end(&spall_ctx, &spall_buffer)
}

}

// checks if it should rebuild a shader
// if it should then compiles the new shader and makes a new PSO with it
hotswap_watch :: proc(hs: ^HotSwapState, root_signature: ^dx.IRootSignature, shader_name: string, is_lighting_pass: bool) {
    // watch for shader change
	game_dll_mod, game_dll_mod_err := os.last_write_time_by_name(shader_name)
	
	reload := false
	
	if game_dll_mod_err == os.ERROR_NONE && hs.last_write_time != game_dll_mod {
	    hs.last_write_time = game_dll_mod
		reload = true
	}
	
	if reload {
		lprintfln("Recompiling shader...")
		// handle releasing resources
		vs, ps, ok := compile_shader(shader_name)
		if !ok {
               lprintfln("Could not compile new shader!! check logs")
		} else {
			// create the new PSO to be swapped later
			if is_lighting_pass {
    			hs.pso_swap = create_new_lighting_pso(root_signature, vs, ps)
			}
			else {
    			hs.pso_swap = create_new_gbuffer_pso(root_signature, vs, ps)
			}
			
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
