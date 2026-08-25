package main

import "core:slice"
import "core:thread"
import dx "vendor:directx/d3d12"
import dxgi "vendor:directx/dxgi"
import sdl "vendor:sdl2"
import "core:strings"
import "core:fmt"
import "core:mem/virtual"

Color :: v4

PSOName :: enum {
	Quad
}

Lucy2DContext :: struct {
	upload_thread : ^thread.Thread,
	resources_resizing : [dynamic]^dx.IUnknown,
	resources_longterm : [dynamic]^dx.IUnknown,
	window : ^sdl.Window,
	window_dimensions: v2i,
	root_signatures: [RootSignatureChoice]^dx.IRootSignature,
	psos: [PSOName]PSO,
	sb_sprites: StructuredBuffer,
	sprites_to_render: [dynamic]Sprite,
	cb_general: ConstantBufferUpload
}

SPRITE_MAX_COUNT :: 100

// TODO: do the reflection thing that copies your structs to hlsl
Sprite :: struct #align (16) {
	pos: v2,
	size: v2
}

GeneralConstants :: struct #align (256) {
	sb_sprites_idx: u32,
	inv_screen: v2,
}

g_lct : Lucy2DContext

// creates the window
// the app HAS to call this before anything else.
window_new :: proc(window_name:string, width, height: int) {

	// set up allocators?
	g_lct.upload_thread = thread.create_and_start(lucy2d_upload_thread_start)

	// setting up resource pool for buffers tied to window size
	g_lct.resources_resizing = make([dynamic]^dx.IUnknown)
	g_lct.resources_longterm = make([dynamic]^dx.IUnknown)

	ct := &g_lct

	ct.window_dimensions = v2i{width, height}

	// Init SDL and create window
	if err := sdl.Init(sdl.InitFlags{.TIMER, .AUDIO, .VIDEO, .EVENTS}); err != 0 {
		fmt.eprintln(err)
		return
	}

	window_name_cstr := strings.clone_to_cstring(window_name, context.temp_allocator)
	ct.window = sdl.CreateWindow(
		window_name_cstr,
		sdl.WINDOWPOS_UNDEFINED,
		sdl.WINDOWPOS_UNDEFINED,
		cast(i32)width,
		cast(i32)height,
		{.ALLOW_HIGHDPI, .SHOWN, .RESIZABLE},
	)

	if ct.window == nil {
		fmt.eprintln(sdl.GetError())
		return
	}

	init_dx(&g_lct.resources_longterm, ct.window)

	// Creating all root signatures
	g_lct.root_signatures = create_root_signatures(&g_lct.resources_longterm)

	// Creating a Constant buffer?? if needed

	// Creating Sprite Structured buffer

	g_lct.cb_general = cb_upload_create(size_of(GeneralConstants), &g_lct.resources_longterm, name = "general constants cbv")

	g_lct.sb_sprites = structured_buffer_create("Sprite buffer", &g_lct.resources_longterm, Sprite, SPRITE_MAX_COUNT, heap_type = .UPLOAD)

	// Creating PSO's
	g_lct.psos[.Quad] = pso_create("src/shaders/quads.hlsl", &ct.root_signatures, &ct.resources_longterm, PSOParameters {
		vertex_input = struct{},
		blend_state = .Normal,
		cull_mode = .None,
		enable_depth = false,
		depth_write = false,
		root_signature = .Standard,
		rtv_count = 1,
		rtv_formats = {0 = .R8G8B8A8_UNORM, 1 ..=7 = .UNKNOWN},
	}, render_proc = pso_quad_render, pso_name = "Quad PSO")
}

create_root_signatures :: proc(pool : ^DXResourcePool) -> (root_signatures : [RootSignatureChoice]^dx.IRootSignature){
	ct := &g_dx_core
	hr : dx.HRESULT

	root_parameters:= [2]dx.ROOT_PARAMETER {
		// This is the index of the CBV on the srv heap
		{
			ParameterType = ._32BIT_CONSTANTS,
			Constants = {ShaderRegister = 0, Num32BitValues = 1},
			ShaderVisibility = .ALL
		},
		// This is the DrawConstants for the mesh drawing
		{
			ParameterType = ._32BIT_CONSTANTS,
			Constants = {ShaderRegister = 1, Num32BitValues = 2},
			ShaderVisibility = .ALL
		}
	}

	sampler_descs := [?]dx.STATIC_SAMPLER_DESC { 
		{
			Filter = .ANISOTROPIC,
			AddressU = .WRAP,
			AddressV = .WRAP,
			AddressW = .WRAP,
			MipLODBias = 0.0,
			MaxAnisotropy = 16,
			ComparisonFunc = .NEVER,
			MinLOD = 0.0,
			MaxLOD = dx.FLOAT32_MAX,
			ShaderRegister = 0,
			RegisterSpace = 0,
			ShaderVisibility = .ALL,
		},
		{
			Filter = .COMPARISON_MIN_MAG_LINEAR_MIP_POINT,
			AddressU = .BORDER,
			AddressV = .BORDER,
			AddressW = .BORDER,
			MipLODBias = 0.0,
			ComparisonFunc = .LESS_EQUAL,
			BorderColor = .OPAQUE_WHITE,
			MinLOD = 0.0,
			MaxLOD = dx.FLOAT32_MAX,
			ShaderRegister = 1,
			RegisterSpace = 0,
			ShaderVisibility = .ALL,
		},
		{
			Filter = .MIN_MAG_MIP_LINEAR,
			AddressU = .CLAMP,
			AddressV = .CLAMP,
			AddressW = .CLAMP,
			MipLODBias = 0.0,
			ComparisonFunc = .NEVER,
			MinLOD = 0.0,
			MaxLOD = dx.FLOAT32_MAX,
			ShaderRegister = 2,
			RegisterSpace = 0,
			ShaderVisibility = .ALL,
		},
	}

	desc := dx.VERSIONED_ROOT_SIGNATURE_DESC {
		Version = ._1_0,
		Desc_1_0 = {
			NumParameters = len(root_parameters),
			pParameters = &root_parameters[0],
			NumStaticSamplers = len(sampler_descs),
			pStaticSamplers = &sampler_descs[0],
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
		(^rawptr)(&root_signatures[.Standard]),
	)
	check(hr, "Failed creating root signature")
	append(pool, root_signatures[.Standard])
	serialized_desc->Release()

	// creating other root signatrues here when needed

	return
}

window_cleanup :: proc() {
	ct := &g_lct
	thread.destroy(g_lct.upload_thread)
	sdl.DestroyWindow(ct.window)
	sdl.Quit()

	resource_pool_release(&g_lct.resources_longterm)
	resource_pool_release(&g_lct.resources_resizing)

	delete(g_lct.resources_resizing)
	delete(g_lct.resources_longterm)

	when ODIN_DEBUG {
	debug_device: ^dx.IDebugDevice2
	g_dx_core.device->QueryInterface(dx.IDebugDevice2_UUID, (^rawptr)(&debug_device))
	// Finally, release the device (it is not in any pool)
	// The device will be freed after we release the debug device
	g_dx_core.device->Release()
	debug_device->ReportLiveDeviceObjects({.DETAIL, .IGNORE_INTERNAL})
	debug_device->Release()

	// DXGI report
	dxgi_debug: ^dxgi.IDebug1
	dxgi.GetDebugInterface1(0, dxgi.IDebug1_UUID, (^rawptr)(&dxgi_debug))
	dxgi_debug->ReportLiveObjects(dxgi.DEBUG_ALL, {})
	}
}

// clears the window with a color
window_clear :: proc(color: Color) {

}

draw_sprite :: proc(pos, size: v2) {

}

// draws all the stuff, and resets frame state
present :: proc() {

	ctd := &g_dx_core
	ct := &g_lct

	// Updating Constant Buffer
	{
		copy_to_buffer_already_mapped_value(ct.cb_general.gpu_pointer, &GeneralConstants {
			sb_sprites_idx = cast(u32)ct.sb_sprites.srv_index,
			inv_screen = 1.0 / v2{cast(f32)ct.window_dimensions.x, cast(f32)ct.window_dimensions.y}
		})
	}

	// Rendering everything
	g_dx_core.cmdlist->Reset(ctd.command_allocator, nil)

	for pso in ct.psos {
		pso.render_proc(pso)
	}

	dx_frame_end()

	// Setting things up for the next frame
	clear(&ct.sprites_to_render)
	sdl.PumpEvents()
}

get_keyboard :: proc() -> []u8 {
	return sdl.GetKeyboardStateAsSlice()
}

lucy2d_upload_thread_start :: proc() {

	// TODO: set the allocator to the tracking allocator

	// context.allocator = mem.tracking_allocator(&g_track)

	// make temp allocator for upload thread
	upload_temp_arena := arena_new()
	upload_temp_allocator := virtual.arena_allocator(&upload_temp_arena)
	context.temp_allocator = upload_temp_allocator

	// ends
	// it does nothing


	// ending...
	arena_destroy(&upload_temp_arena)
}

pso_quad_render :: proc(pso: PSO) {

	ct := &g_lct

	lprintfln("rendering sprites...")

	copy_to_buffer_already_mapped(ct.sb_sprites.gpu_pointer, slice.to_bytes(ct.sprites_to_render[:]))

	// here do the draw call
}
