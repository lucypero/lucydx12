// D3D12 single-function triangle sample.
//
// Usage:
// - copy SDL2.dll from Odin/vendor/sdl2 to your project directory
// - odin run .
//
// Contributors:
// - Karl Zylinski <karl@zylinski.se> (version 1, version 3)
// - Jakub Tomšů (version 2)
//
// Based on:
// - https://gist.github.com/karl-zylinski/e1d1d0925ac5db0f12e4837435c5bbfb
// - https://gist.github.com/jakubtomsu/ecd83e61976d974c7730f9d7ad3e1fd0
// - https://github.com/rdunnington/d3d12-hello-triangle/blob/master/main.c

package main

import "core:odin/ast"
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
import "core:container/small_array"
import "base:runtime"

// imgui
import im "../odin-imgui"
// imgui sdl2 implementation
import "../odin-imgui/imgui_impl_sdl2"
// imgui dx12 implementation
import "../odin-imgui/imgui_impl_dx12"

NUM_RENDERTARGETS :: 2

TURNS_TO_RAD :: math.PI * 2

// window dimensions
wx := i32(2000)
wy := i32(1000)

v2 :: linalg.Vector2f32
v3 :: linalg.Vector3f32
v4 :: linalg.Vector4f32

// constant buffer data
ConstantBufferData :: struct #align (256) {
	wvp: dxm,
	time: f32,
}

VertexData :: struct {
	pos: v3,
	normal: v3,
	uv: v2,
}

dxm :: matrix[4,4]f32

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
	pipeline:            ^dx.IPipelineState,
	cmdlist:             ^dx.IGraphicsCommandList,
	map_start:           rawptr, //maps to our test constant buffer
	root_signature:      ^dx.IRootSignature,
	constant_buffer:     ^dx.IResource,
	vertex_buffer_view:  dx.VERTEX_BUFFER_VIEW,
	index_buffer_view: dx.INDEX_BUFFER_VIEW,
	// descriptor heap for the render target view
	rtv_descriptor_heap: ^dx.IDescriptorHeap, 
	frame_index:         u32,
	targets:             [NUM_RENDERTARGETS]^dx.IResource, // render targets

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

	// app data

	cam_angle: f32,
	cam_distance: f32,
}

// initializes app data in Context struct
context_init :: proc(con: ^Context) {
	con.cam_angle = 0.125
	con.cam_distance = 2.4
}

check :: proc(res: dx.HRESULT, message: string) {
	if (res >= 0) {
		return
	}

	fmt.printf("%v. Error code: %0x\n", message, u32(res))
	os.exit(-1)
}

dx_context: Context
start_time: time.Time


main :: proc() {
	// Init SDL and create window
	if err := sdl.Init(sdl.INIT_EVERYTHING); err != 0 {
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
			(^rawptr)(&dx_context.rtv_descriptor_heap),
		)
		check(hr, "Failed creating descriptor heap")
		dx_context.rtv_descriptor_heap->SetName("lucy's RTV descriptor heap")
	}

	// Fetch the two render targets from the swapchain

	{
		rtv_descriptor_size: u32 = device->GetDescriptorHandleIncrementSize(.RTV)

		rtv_descriptor_handle: dx.CPU_DESCRIPTOR_HANDLE
		dx_context.rtv_descriptor_heap->GetCPUDescriptorHandleForHeapStart(&rtv_descriptor_handle)

		for i: u32 = 0; i < NUM_RENDERTARGETS; i += 1 {
			hr =
			dx_context.swapchain->GetBuffer(
				i,
				dx.IResource_UUID,
				(^rawptr)(&dx_context.targets[i]),
			)
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

		// empty range means the cpu won't read from it
		dx_context.constant_buffer->Map(0, &dx.RANGE{}, &dx_context.map_start)

		// // giving the constant buffer some data
		// cbvdata_example := ConstantBufferData{0}
		// mem.copy(dx_context.map_start, (rawptr)(&cbvdata_example), size_of(cbvdata_example))
	}

	create_depth_buffer()


	/* 
	From https://docs.microsoft.com/en-us/windows/win32/direct3d12/root-signatures-overview:
	
		A root signature is configured by the app and links command lists to the resources the shaders require.
		The graphics command list has both a graphics and compute root signature. A compute command list will
		simply have one compute root signature. These root signatures are independent of each other.
	*/

	create_root_signature()


	// The pipeline contains the shaders etc to use

	// SHADERCODE

	{

		data, ok := os.read_entire_file("shader.hlsl")
		if !ok {
			fmt.eprintln("could not read file")
			os.exit(1)
		}
		defer(delete(data))
		data_size: uint = len(data)

		compile_flags: u32 = 0
		when ODIN_DEBUG {
			compile_flags |= u32(d3dc.D3DCOMPILE.DEBUG)
			compile_flags |= u32(d3dc.D3DCOMPILE.SKIP_OPTIMIZATION)
		}

		vs: ^dx.IBlob = nil
		ps: ^dx.IBlob = nil

		// errors
		vs_res: ^d3dc.ID3DBlob
		ps_res: ^d3dc.ID3DBlob

		hr = d3dc.Compile(
			rawptr(&data[0]), data_size, nil, nil, nil, "VSMain", "vs_4_0",
			compile_flags, 0, &vs, &vs_res,
		)

		if (vs_res != nil) {
			// errors in shader compilation
			a := strings.string_from_ptr(
				(^u8)(vs_res->GetBufferPointer()),
				int(vs_res->GetBufferSize()),
			)
			fmt.println("DXC VS ERRORS: ", a)
		}

		check(hr, "Failed to compile vertex shader")


		hr = d3dc.Compile(
			rawptr(&data[0]), data_size, nil, nil, nil, "PSMain", "ps_4_0",
			compile_flags, 0, &ps, &ps_res
		)

		check(hr, "Failed to compile pixel shader")

		if (ps_res != nil) {
			// errors in shader compilation
			a := strings.string_from_ptr(
				(^u8)(ps_res->GetBufferPointer()),
				int(ps_res->GetBufferSize()),
			)
			fmt.println("DXC PS ERRORS: ", a)
		}

		// INPUTLAYOUT

		// This layout matches the vertices data defined further down
		vertex_format: []dx.INPUT_ELEMENT_DESC = {
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

		pipeline_state_desc := dx.GRAPHICS_PIPELINE_STATE_DESC {
			pRootSignature = dx_context.root_signature,
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
			InputLayout = {
				pInputElementDescs = &vertex_format[0],
				NumElements = u32(len(vertex_format)),
			},
			PrimitiveTopologyType = .TRIANGLE,
			NumRenderTargets = 1,
			RTVFormats = {0 = .R8G8B8A8_UNORM, 1 ..< 7 = .UNKNOWN},
			DSVFormat = .D32_FLOAT,
			SampleDesc = {Count = 1, Quality = 0},
		}

		hr =
		device->CreateGraphicsPipelineState(
			&pipeline_state_desc,
			dx.IPipelineState_UUID,
			(^rawptr)(&dx_context.pipeline),
		)
		check(hr, "Pipeline creation failed")

		vs->Release()
		ps->Release()
	}

	// Create the commandlist that is reused further down.
	hr =
	device->CreateCommandList(
		0,
		.DIRECT,
		dx_context.command_allocator,
		dx_context.pipeline,
		dx.ICommandList_UUID,
		(^rawptr)(&dx_context.cmdlist),
	)
	check(hr, "Failed to create command list")
	hr = dx_context.cmdlist->Close()
	check(hr, "Failed to close command list")

	// texture test
	create_texture()

	create_descriptor_heap_cbv_srv_uav()

	vertex_buffer: ^dx.IResource
	index_buffer: ^dx.IResource

	imgui_init()

	{
		// get vertex data from gltf file
		vertices, indices := do_gltf_stuff()
		dx_context.vertex_count = u32(len(vertices))

		// VERTEXDATA
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

	main_loop: for {
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

		update()
		render()
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
	}

	for i: u32 = 0; factory->EnumAdapters1(i, &adapter) != error_not_found; i += 1 {
		desc: dxgi.ADAPTER_DESC1
		adapter->GetDesc1(&desc)
		if .SOFTWARE in desc.Flags {
			continue
		}

		hr = dx.CreateDevice((^dxgi.IUnknown)(adapter), ._12_0, dx.IDevice_UUID, nil)

		if hr >= 0 {
			break
		} else {
			fmt.printfln("Failed to create device, err: %X", hr) // -2147467262
			// E_NOINTERFACE
			// no such interface supported
			return
		}
	}

	if adapter == nil {
		fmt.println("Could not find hardware adapter")
		return
	}

	device: ^dx.IDevice

	// Create D3D12 device that represents the GPU
	hr = dx.CreateDevice((^dxgi.IUnknown)(adapter), ._12_0, dx.IDevice_UUID, (^rawptr)(&device))
	check(hr, "Failed to create device")

	dx_context.device = device
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

get_wvp :: proc() -> dxm {

	cam_pos : v3

	cam_pos.z = -dx_context.cam_distance

	// rotate on Y axis

	rot_mat := linalg.matrix3_rotate_f32(dx_context.cam_angle * TURNS_TO_RAD, {0,1,0})
	cam_pos = rot_mat * cam_pos


	view := linalg.matrix4_look_at_f32(cam_pos, {0,0,0}, {0,1,0}, false)

	fov := linalg.to_radians(f32(90.0))
    aspect := f32(wx) / f32(wy)
	proj := linalg.matrix4_perspective_f32(fov, aspect, 0.1, 100, false)

	// this function is supposedly more correct
	// has correct depth values
	// proj := matrix4_perspective_z0_f32(fov, aspect, 0.1, 100)

	return proj * view
}

update :: proc() {

	sdl.PumpEvents()
	keyboard := sdl.GetKeyboardStateAsSlice()

	// controlling camera
	// cam_speed :: 0.01

	// if keyboard[sdl.Scancode.A] == 1{
	// 	cam_pos.x -= cam_speed
	// }
	// if keyboard[sdl.Scancode.D] == 1{
	// 	cam_pos.x += cam_speed
	// }
	// if keyboard[sdl.Scancode.W] == 1{
	// 	cam_pos.y += cam_speed
	// }
	// if keyboard[sdl.Scancode.S] == 1{
	// 	cam_pos.y -= cam_speed
	// }


	// if keyboard[sdl.Scancode.H] == 1{
	// 	cam_pos.z -= cam_speed
	// }

	// if keyboard[sdl.Scancode.J] == 1{
	// 	cam_pos.z += cam_speed
	// }
}

render :: proc() {

	imgui_update()

	command_allocator := dx_context.command_allocator
	pipeline := dx_context.pipeline
	cmdlist := dx_context.cmdlist


	hr: dx.HRESULT
	// ticking cbv value
	thetime := time.diff(start_time, time.now())
	float_val := f32(thetime) / f32(time.Second)
	if float_val > 1 {
		start_time = time.now()
	}

	// sending constant buffer data

	wvp := get_wvp()


	cbvdata_example := ConstantBufferData{
		wvp = wvp,
		time = float_val
	}

	mem.copy(dx_context.map_start, (rawptr)(&cbvdata_example), size_of(cbvdata_example))
	// case .WINDOWEVENT:
	// This is equivalent to WM_PAINT in win32 API
	// if e.window.event == .EXPOSED {
	hr = command_allocator->Reset()
	check(hr, "Failed resetting command allocator")

	hr = cmdlist->Reset(command_allocator, pipeline)
	check(hr, "Failed to reset command list")

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

	// This state is reset everytime the cmd list is reset, so we need to rebind it
	cmdlist->SetGraphicsRootSignature(dx_context.root_signature)

	// setting descriptor heap for our cbv srv uav's
	cmdlist->SetDescriptorHeaps(1, &dx_context.descriptor_heap_cbv_srv_uav)

	// setting the root cbv that we set up in the root signature. root parameter 0
	cmdlist->SetGraphicsRootConstantBufferView(
		0,
		dx_context.constant_buffer->GetGPUVirtualAddress(),
	)

	// setting descriptor tables for our texture. root parameter 1
	{
		// setting the graphics root descriptor table
		// in the root signature, so that it points to
		// our SRV descriptor
		cmdlist->SetGraphicsRootDescriptorTable(1, 
			get_descriptor_heap_gpu_address(dx_context.descriptor_heap_cbv_srv_uav, 1)
		)
	}

	cmdlist->RSSetViewports(1, &viewport)
	cmdlist->RSSetScissorRects(1, &scissor_rect)

	to_render_target_barrier := dx.RESOURCE_BARRIER {
		Type  = .TRANSITION,
		Flags = {},
	}

	to_render_target_barrier.Transition = {
		pResource   = dx_context.targets[dx_context.frame_index],
		StateBefore = dx.RESOURCE_STATE_PRESENT,
		StateAfter  = {.RENDER_TARGET},
		Subresource = dx.RESOURCE_BARRIER_ALL_SUBRESOURCES,
	}

	cmdlist->ResourceBarrier(1, &to_render_target_barrier)
	// now that the RTVs are set to Render target, you can draw.


	// Setting render targets. Clearing DSV and RTV.
	{
		rtv_handle := get_descriptor_heap_cpu_address(dx_context.rtv_descriptor_heap, dx_context.frame_index)
		dsv_handle := get_descriptor_heap_cpu_address(dx_context.descriptor_heap_dsv, 0)

		// setting depth buffer
		cmdlist->OMSetRenderTargets(1, &rtv_handle, false, &dsv_handle)

		// clear backbuffer
		clearcolor := [?]f32{0.05, 0.05, 0.05, 1.0}
		cmdlist->ClearRenderTargetView(rtv_handle, &clearcolor, 0, nil)

		// clearing depth buffer
		cmdlist->ClearDepthStencilView(dsv_handle, {.DEPTH, .STENCIL}, 1.0, 0, 0, nil)
	}

	// draw call
	cmdlist->IASetPrimitiveTopology(.TRIANGLELIST)
	cmdlist->IASetVertexBuffers(0, 1, &dx_context.vertex_buffer_view)
	cmdlist->IASetIndexBuffer(&dx_context.index_buffer_view)
	cmdlist->DrawIndexedInstanced(dx_context.index_count, 1, 0, 0, 0)

	// add imgui draw commands to cmd list
	imgui_update_after()

	// Cannot draw after this point because we transition the render target to "Present" state

	to_present_barrier := to_render_target_barrier
	to_present_barrier.Transition.StateBefore = {.RENDER_TARGET}
	to_present_barrier.Transition.StateAfter = dx.RESOURCE_STATE_PRESENT

	cmdlist->ResourceBarrier(1, &to_present_barrier)

	hr = cmdlist->Close()
	check(hr, "Failed to close command list")

	// execute
	cmdlists := [?]^dx.IGraphicsCommandList{cmdlist}
	dx_context.queue->ExecuteCommandLists(len(cmdlists), (^^dx.ICommandList)(&cmdlists[0]))

	// present
	{
		flags: dxgi.PRESENT
		params: dxgi.PRESENT_PARAMETERS
		hr = dx_context.swapchain->Present1(1, flags, &params)
		check(hr, "Present failed")
	}

	// wait for frame to finish
	{
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

		dx_context.frame_index = dx_context.swapchain->GetCurrentBackBufferIndex()
	}
}

// creating resource and uploading it to the gpu
create_texture :: proc() {

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

	dx_context.cmdlist->Reset(dx_context.command_allocator, dx_context.pipeline)
	dx_context.cmdlist->CopyTextureRegion(&copy_location_dst, 0, 0, 0, &copy_location_src, nil)


	// TODO: do a fence here, wait for it, then release the upload resource, and change the texture state to generic read

	fence_value: u64
	fence: ^dx.IFence
	hr = dx_context.device->CreateFence(fence_value, {}, dx.IFence_UUID, (^rawptr)(&fence))
	fence_value += 1


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

	// here, the gpu is done! now release upload resource then change texture type to generic read
	texture_upload->Release()


	// creating SRV


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

create_root_signature :: proc() {

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
		(^rawptr)(&dx_context.root_signature),
	)
	check(hr, "Failed creating root signature")
	serialized_desc->Release()
}

do_gltf_stuff :: proc() -> (vertices: []VertexData, indices: []u16) {

	model_filepath :: "models/monke.glb"
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
			fmt.eprintfln("Unkown gltf attribute: {}", attribute)
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


	// depth stencil view descriptor heap

	// creating descriptor heap
	heap_desc := dx.DESCRIPTOR_HEAP_DESC {
		NumDescriptors = 1,
		Type           = .DSV,
		Flags          = {},
	}

	hr = c.device->CreateDescriptorHeap(&heap_desc, 
		dx.IDescriptorHeap_UUID, (^rawptr)(&c.descriptor_heap_dsv))

	dx_context.rtv_descriptor_heap->SetName("lucy's DSV (depth-stencil-view) descriptor heap")

	check(hr, "could not create descriptor heap for DSV")

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

	dx_context.imgui_descriptor_heap->SetName("imgui's cbv srv uav descriptor heap")

	check(hr, "could ont create imgui descriptor heap")

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
	im.DestroyContext()
	imgui_impl_sdl2.Shutdown() // here
	imgui_impl_dx12.Shutdown()
}


imgui_update :: proc() {
	imgui_impl_dx12.NewFrame()
	imgui_impl_sdl2.NewFrame()
	im.NewFrame()

	im.ShowDemoWindow()

	// im.End()

	im.Begin("hello")
	im.Text("hello")
	if im.Button("click me") {
		fmt.println("clicked!")
	}

	im.SliderFloat("camera angle", &dx_context.cam_angle, 0, 1)
	im.SliderFloat("camera distance", &dx_context.cam_distance, 0.5, 20)

	im.End()

	im.Render()
}

imgui_update_after :: proc() {
	// call this right before swapchain present

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
