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

package d3d12_triangle

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

NUM_RENDERTARGETS :: 2

// window dimensions
wx := i32(640)
wy := i32(480)

// constant buffer data
ConstantBufferData :: struct #align (256) {
	time: f32,
}

Context :: struct {
	// core stuff
	device: ^dx.IDevice,
	factory: ^dxgi.IFactory4,
	queue: ^dx.ICommandQueue,
	swapchain: ^dxgi.ISwapChain3,

	command_allocator: ^dx.ICommandAllocator,
	pipeline: ^dx.IPipelineState,
	cmdlist: ^dx.IGraphicsCommandList,

	map_start: rawptr, //maps to our test constant buffer

	root_signature: ^dx.IRootSignature,
	constant_buffer: ^dx.IResource,
	vertex_buffer_view: dx.VERTEX_BUFFER_VIEW,

	rtv_descriptor_heap: ^dx.IDescriptorHeap,

	frame_index : u32,

	targets: [NUM_RENDERTARGETS]^dx.IResource, // render targets

	// fence stuff
	fence: ^dx.IFence,
	fence_value: u64,
	fence_event: windows.HANDLE,
}

check :: proc(res: dx.HRESULT, message: string) {
	if (res >= 0) {
		return
	}

	fmt.printf("%v. Error code: %0x\n", message, u32(res))
	os.exit(-1)
}

dx_context : Context
start_time: time.Time

main :: proc() {
	// Init SDL and create window

	if err := sdl.Init({.VIDEO}); err != 0 {
		fmt.eprintln(err)
		return
	}

	defer sdl.Quit()
	window := sdl.CreateWindow(
		"d3d12 triangle",
		sdl.WINDOWPOS_UNDEFINED,
		sdl.WINDOWPOS_UNDEFINED,
		wx,
		wy,
		{.ALLOW_HIGHDPI, .SHOWN, .RESIZABLE},
	)

	if window == nil {
		fmt.eprintln(sdl.GetError())
		return
	}

	defer sdl.DestroyWindow(window)

	init_dx()

	device := dx_context.device

	hr: dx.HRESULT

	{
		desc := dx.COMMAND_QUEUE_DESC {
			Type = .DIRECT,
		}

		hr = device->CreateCommandQueue(&desc, dx.ICommandQueue_UUID, (^rawptr)(&dx_context.queue))
		check(hr, "Failed creating command queue")
	}

	// Create the swapchain, it's the thing that contains render targets that we draw into. It has 2 render targets (NUM_RENDERTARGETS), giving us double buffering.
	dx_context.swapchain = create_swapchain(dx_context.factory, dx_context.queue, window)

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
	}

	// Fetch the two render targets from the swapchain

	{
		rtv_descriptor_size: u32 = device->GetDescriptorHandleIncrementSize(.RTV)

		rtv_descriptor_handle: dx.CPU_DESCRIPTOR_HANDLE
		dx_context.rtv_descriptor_heap->GetCPUDescriptorHandleForHeapStart(&rtv_descriptor_handle)

		for i: u32 = 0; i < NUM_RENDERTARGETS; i += 1 {
			hr = dx_context.swapchain->GetBuffer(i, dx.IResource_UUID, (^rawptr)(&dx_context.targets[i]))
			check(hr, "Failed getting render target")
			device->CreateRenderTargetView(dx_context.targets[i], nil, rtv_descriptor_handle)
			rtv_descriptor_handle.ptr += uint(rtv_descriptor_size)
		}
	}

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
			Width = 256,
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

		/// creating a cbv (constant buffer view)

		// creating descriptor heap
		cbv_heap_desc := dx.DESCRIPTOR_HEAP_DESC {
			NumDescriptors = 1,
			Type           = .CBV_SRV_UAV,
			Flags          = {.SHADER_VISIBLE},
		}

		cbv_heap: ^dx.IDescriptorHeap
		hr =
		device->CreateDescriptorHeap(
			&cbv_heap_desc,
			dx.IDescriptorHeap_UUID,
			(^rawptr)(&cbv_heap),
		)
		check(hr, "failed creating descriptor heap")

		// creating the cbv

		cbv_desc := dx.CONSTANT_BUFFER_VIEW_DESC {
			BufferLocation = dx_context.constant_buffer->GetGPUVirtualAddress(),
			SizeInBytes    = 256,
		}

		cpu_desc_handle: dx.CPU_DESCRIPTOR_HANDLE
		cbv_heap->GetCPUDescriptorHandleForHeapStart(&cpu_desc_handle)
		device->CreateConstantBufferView(&cbv_desc, cpu_desc_handle)

		// updating the constant buffer

		// example float value

		// ConstantBufferData :: struct #align (256) {
		// 	time: f32,
		// }

		cbvdata_example := ConstantBufferData{0}
		mem.copy(dx_context.map_start, (rawptr)(&cbvdata_example), size_of(cbvdata_example))
	}



	/* 
	From https://docs.microsoft.com/en-us/windows/win32/direct3d12/root-signatures-overview:
	
		A root signature is configured by the app and links command lists to the resources the shaders require.
		The graphics command list has both a graphics and compute root signature. A compute command list will
		simply have one compute root signature. These root signatures are independent of each other.
	*/

	{

		root_parameters: [1]dx.ROOT_PARAMETER
		root_parameters[0] = {
			ParameterType = .CBV,
			Descriptor = {ShaderRegister = 0, RegisterSpace = 0},
			ShaderVisibility = .ALL, // vertex, pixel, or both (all)
		}

		desc := dx.VERSIONED_ROOT_SIGNATURE_DESC {
			Version = ._1_0,
			// defining the cbv here
			Desc_1_0 = {NumParameters = 1, pParameters = &root_parameters[0]},
		}

		desc.Desc_1_0.Flags = {.ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT}
		serialized_desc: ^dx.IBlob
		hr = dx.SerializeVersionedRootSignature(&desc, &serialized_desc, nil)
		check(hr, "Failed to serialize root signature")
		hr =
		device->CreateRootSignature(
			0,
			serialized_desc->GetBufferPointer(),
			serialized_desc->GetBufferSize(),
			dx.IRootSignature_UUID,
			(^rawptr)(&dx_context.root_signature),
		)
		check(hr, "Failed creating root signature")
		serialized_desc->Release()
	}

	// The pipeline contains the shaders etc to use

	{
		// Compile vertex and pixel shaders
		data: cstring = `struct PSInput {
			   float4 position : SV_POSITION;
			   float4 color : COLOR;
			};

cbuffer ConstantBuffer : register(b0) {
    float someValue;
};

			PSInput VSMain(float4 position : POSITION0, float4 color : COLOR0) {
			   PSInput result;
			   result.position = position;
			   result.color = color;
			   result.color.r = someValue;
			   result.color.g = 0;
			   result.color.b = 0;
			   return result;
			}
			float4 PSMain(PSInput input) : SV_TARGET {
			   return input.color;
			};`


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

		hr = d3dc.Compile(
			rawptr(data),
			data_size,
			nil,
			nil,
			nil,
			"VSMain",
			"vs_4_0",
			compile_flags,
			0,
			&vs,
			&vs_res,
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
			rawptr(data),
			data_size,
			nil,
			nil,
			nil,
			"PSMain",
			"ps_4_0",
			compile_flags,
			0,
			&ps,
			nil,
		)
		check(hr, "Failed to compile pixel shader")

		// This layout matches the vertices data defined further down
		vertex_format: []dx.INPUT_ELEMENT_DESC = {
			{
				SemanticName = "POSITION",
				Format = .R32G32B32_FLOAT,
				InputSlotClass = .PER_VERTEX_DATA,
			},
			{
				SemanticName = "COLOR",
				Format = .R32G32B32A32_FLOAT,
				AlignedByteOffset = size_of(f32) * 3,
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
			DepthStencilState = {DepthEnable = false, StencilEnable = false},
			InputLayout = {
				pInputElementDescs = &vertex_format[0],
				NumElements = u32(len(vertex_format)),
			},
			PrimitiveTopologyType = .TRIANGLE,
			NumRenderTargets = 1,
			RTVFormats = {0 = .R8G8B8A8_UNORM, 1 ..< 7 = .UNKNOWN},
			DSVFormat = .UNKNOWN,
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

	vertex_buffer: ^dx.IResource

	{
		// The position and color data for the triangle's vertices go together per-vertex
		vertices := [?]f32 {
			// pos            color
			0.0,
			0.5,
			0.0,
			1,
			0,
			0,
			0,
			0.5,
			-0.5,
			0.0,
			0,
			1,
			0,
			0,
			-0.5,
			-0.5,
			0.0,
			0,
			0,
			1,
			0,
		}

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
		check(hr, "Failed creating verex buffer resource")

		mem.copy(gpu_data, &vertices[0], vertex_buffer_size)
		vertex_buffer->Unmap(0, nil)

		dx_context.vertex_buffer_view = dx.VERTEX_BUFFER_VIEW {
			BufferLocation = vertex_buffer->GetGPUVirtualAddress(),
			StrideInBytes  = u32(vertex_buffer_size / 3),
			SizeInBytes    = u32(vertex_buffer_size),
		}
	}

	// This fence is used to wait for frames to finish
	{
		hr = device->CreateFence(dx_context.fence_value, {}, dx.IFence_UUID, (^rawptr)(&dx_context.fence))
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
			#partial switch e.type {
			case .QUIT:
				break main_loop
			}
		}

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

		if dx.CreateDevice((^dxgi.IUnknown)(adapter), ._12_0, dxgi.IDevice_UUID, nil) >= 0 {
			break
		} else {
			fmt.println("Failed to create device")
		}
	}

	if adapter == nil {
		fmt.println("Could not find hardware adapter")
		return
	}

	device: ^dx.IDevice

	// Create D3D12 device that represents the GPU
	hr = dx.CreateDevice(
		(^dxgi.IUnknown)(adapter),
		._12_0,
		dx.IDevice_UUID,
		(^rawptr)(&device),
	)
	check(hr, "Failed to create device")

	dx_context.device = device
}

render :: proc() {

	command_allocator := dx_context.command_allocator
	pipeline := dx_context.pipeline
	cmdlist := dx_context.cmdlist


	hr : dx.HRESULT
	// ticking cbv value
	thetime := time.diff(start_time, time.now())
	float_val := f32(thetime) / f32(time.Second)
	if float_val > 1 {
		start_time = time.now()
	}

	cbvdata_example := ConstantBufferData{float_val}
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
	}

	scissor_rect := dx.RECT {
		left   = 0,
		right  = wx,
		top    = 0,
		bottom = wy,
	}

	// This state is reset everytime the cmd list is reset, so we need to rebind it
	cmdlist->SetGraphicsRootSignature(dx_context.root_signature)
	cmdlist->SetGraphicsRootConstantBufferView(0, dx_context.constant_buffer->GetGPUVirtualAddress())
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

	rtv_handle: dx.CPU_DESCRIPTOR_HANDLE
	dx_context.rtv_descriptor_heap->GetCPUDescriptorHandleForHeapStart(&rtv_handle)

	if (dx_context.frame_index > 0) {
		s := dx_context.device->GetDescriptorHandleIncrementSize(.RTV)
		rtv_handle.ptr += uint(dx_context.frame_index * s)
	}

	cmdlist->OMSetRenderTargets(1, &rtv_handle, false, nil)

	// clear backbuffer
	clearcolor := [?]f32{0.05, 0.05, 0.05, 1.0}
	cmdlist->ClearRenderTargetView(rtv_handle, &clearcolor, 0, nil)

	// draw call
	cmdlist->IASetPrimitiveTopology(.TRIANGLELIST)
	cmdlist->IASetVertexBuffers(0, 1, &dx_context.vertex_buffer_view)
	cmdlist->DrawInstanced(3, 1, 0, 0)

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
			hr = dx_context.fence->SetEventOnCompletion(current_fence_value, dx_context.fence_event)
			check(hr, "Failed to set event on completion flag")
			windows.WaitForSingleObject(dx_context.fence_event, windows.INFINITE)
		}

		dx_context.frame_index = dx_context.swapchain->GetCurrentBackBufferIndex()
	}
}

// creating resource and uploading it to the gpu
create_texture :: proc() {

	//create texture resource
	texture : ^dx.IResource

	texture_width :: 256

	// default heap (this is where the final texture will reside)

	heap_properties := dx.HEAP_PROPERTIES {
		Type = .DEFAULT,
	}
	texture_desc := dx.RESOURCE_DESC {
		Width = texture_width,
		Dimension = .TEXTURE2D,
		Height = texture_width,
		Layout = .UNKNOWN,
		Format = .R8G8B8A8_UNORM,
		DepthOrArraySize = 1,
		MipLevels = 1,
		SampleDesc = {Count = 1},
	}

	hr :=
	dx_context.device->CreateCommittedResource(
		&heap_properties,
		dx.HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES,
		&texture_desc,
		dx.RESOURCE_STATE_GENERIC_READ,
		nil,
		dx.IResource_UUID,
		(^rawptr)(&texture),
	)

	check(hr, "failed creating texture")

	// getting data from texture that we'll use later
	text_footprint : dx.PLACED_SUBRESOURCE_FOOTPRINT
	text_bytes : u64

	dx_context.device->GetCopyableFootprints(&texture_desc, 0, 1, 0, &text_footprint, nil, nil, &text_bytes)

	// creating upload heap and resource (needed to upload texture data from cpu to the default heap)

	heap_properties = dx.HEAP_PROPERTIES {
		Type = .UPLOAD
	}

	texture_upload : ^dx.IResource
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

	texture_map_start : rawptr
	texture_upload->Map(0, &dx.RANGE{}, &texture_map_start)

	// sending random data for now
	the_texture_data: [25]f32 = 25
	mem.copy(texture_map_start, (rawptr)(&the_texture_data), size_of(the_texture_data))

	// here you send the gpu command to copy the data to the texture resource.

	copy_location_src := dx.TEXTURE_COPY_LOCATION {
		pResource = texture_upload,
		Type = .PLACED_FOOTPRINT,
		PlacedFootprint = text_footprint
	}

	copy_location_dst := dx.TEXTURE_COPY_LOCATION {
		pResource = texture,
		Type = .SUBRESOURCE_INDEX,
		SubresourceIndex = 0
	}

	dx_context.cmdlist->Reset(dx_context.command_allocator, dx_context.pipeline)
	dx_context.cmdlist->CopyTextureRegion(&copy_location_dst, 0, 0, 0, &copy_location_src, nil)
	dx_context.cmdlist->Close()

	// TODO: do a fence here, wait for it, then release the upload resource
}
