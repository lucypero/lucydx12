package main

import "core:slice"
import "core:thread"
import dx "vendor:directx/d3d12"
import dxgi "vendor:directx/dxgi"
import sdl "vendor:sdl2"
import "core:strings"
import "core:fmt"
import "core:mem/virtual"

// imgui
import im "../libs/odin-imgui"
// imgui sdl2 implementation
import "../libs/odin-imgui/imgui_impl_sdl2"
// imgui dx12 implementation
import "../libs/odin-imgui/imgui_impl_dx12"

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
	cb_general: ConstantBufferUpload,
	clear_color_issued: Maybe(v4),
	window_should_close: bool,
	loaded_textures: map[int]Texture,

	/// imgui stuff
	imgui_descriptor_heap: ^dx.IDescriptorHeap,
	imgui_allocator: DescriptorHeapAllocator,
}

SPRITE_MAX_COUNT :: 100

// TODO: do the reflection thing that copies your structs to hlsl
Sprite :: struct {
	pos: v2,
	size: v2,
	color: v4,
	tex_idx: i32
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

	// Set up dynamic fields
	g_lct.sprites_to_render = make([dynamic]Sprite, 0, 20)
	g_lct.loaded_textures = make(map[int]Texture)

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

	init_dx(&g_lct.resources_longterm, ct.window, width, height)

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
		cull_mode = .Back,
		enable_depth = false,
		depth_write = false,
		root_signature = .Standard,
		rtv_count = 1,
		rtv_formats = {0 = .R8G8B8A8_UNORM, 1 ..=7 = .UNKNOWN},
	}, render_proc = pso_quad_render, pso_name = "Quad PSO")

	// Leave the cmd list closed until it's time to render
	close_and_execute_cmdlist()
}

create_root_signatures :: proc(pool : ^DXResourcePool) -> (root_signatures : [RootSignatureChoice]^dx.IRootSignature){
	ct := &g_dx_core
	hr : dx.HRESULT

	root_parameters:= [?]dx.ROOT_PARAMETER {
		// This is the index of the CBV on the srv heap
		{
			ParameterType = ._32BIT_CONSTANTS,
			Constants = {ShaderRegister = 0, Num32BitValues = 1},
			ShaderVisibility = .ALL
		},
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
		// Pixel art sampler
		{
			Filter = .MIN_MAG_MIP_POINT,
			AddressU = .CLAMP,
			AddressV = .CLAMP,
			AddressW = .CLAMP,
			MipLODBias = 0.0,
			ComparisonFunc = .NEVER,
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
	g_lct.clear_color_issued = color
}

draw_sprite :: proc(sprite: Sprite) {
	append(&g_lct.sprites_to_render, sprite)
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
	swapchain_transition(dx.RESOURCE_STATE_PRESENT, {.RENDER_TARGET})

	// Clear color
	if clear_color, ok := ct.clear_color_issued.?; ok {
		swapchain_clear(clear_color)
	}

	for pso in ct.psos {
		pso.render_proc(pso)
	}

	dx_frame_end()
	frame_end()
}

// Setting things up for the next frame
frame_end :: proc() {
	ct := &g_lct
	ct.clear_color_issued = nil
	clear(&ct.sprites_to_render)
	sdl.PumpEvents()

	for e: sdl.Event; sdl.PollEvent(&e); {
		#partial switch e.type {
		case .QUIT:
			g_lct.window_should_close = true
		case .WINDOWEVENT:
			#partial switch e.window.event {
			case .CLOSE:
				g_lct.window_should_close = true
				// case .RESIZED:
				// 	g_dx_context.resize_wanted = v2i{cast(int)e.window.data1, cast(int)e.window.data2}
			}
		}
	}
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
	ctd := &g_dx_core
	ct := &g_lct

	copy_to_buffer_already_mapped(ct.sb_sprites.gpu_pointer, slice.to_bytes(ct.sprites_to_render[:]))

	// Common render stuff
	{
		ctd.cmdlist->SetPipelineState(pso.pipeline_state)
		ctd.cmdlist->SetDescriptorHeaps(1, &ctd.heap_cbv_srv_uav.heap)
		ctd.cmdlist->SetGraphicsRootSignature(pso.root_signature)
		ctd.cmdlist->SetGraphicsRoot32BitConstant(0, cast(u32)ct.cb_general.srv_index, 0)
		set_viewport_stuff(ct.window_dimensions[0], ct.window_dimensions[1])
	}

	swapchain_set_as_render_target()

	ctd.cmdlist->IASetPrimitiveTopology(.TRIANGLESTRIP)
	ctd.cmdlist->DrawInstanced(4, cast(u32)len(ct.sprites_to_render), 0, 0)
}

window_should_close :: proc() -> bool {
	return g_lct.window_should_close
}

texture_load :: proc(image_filepath: string) -> int {
	texture_dds_path := texture_cache_query(image_filepath, .BC7_UNORM_SRGB, 1, nil)
	dds_file := parse_dds_file(texture_dds_path)
	texture := texture_create(dds_file.mipmap_data, u64(dds_file.width), dds_file.height,
		dds_file.format, &g_lct.resources_longterm, view_flags = {.SRV}, mip_levels = len(dds_file.mipmap_data), texture_name = string(image_filepath))

	tid := texture.srv_index
	g_lct.loaded_textures[tid] = texture
	return tid
}

// Draws texture at the texture's resolution, scaled up given a `scale`
draw_texture :: proc(tex_id: int, pos: v2, scale := v2{1,1}) {
	tex, found := &g_lct.loaded_textures[tex_id]
	ensure(found)

	append(&g_lct.sprites_to_render, Sprite{
		pos = pos,
		size = v2{cast(f32)tex.width, cast(f32)tex.height} * scale,
		tex_idx = cast(i32)tex.srv_index
	})
}

// Draws a solid color rect sized by `size`
draw_solid_rect :: proc(pos, size: v2, color: Color) {
	append(&g_lct.sprites_to_render, Sprite{
		pos = pos,
		size = size,
		color = color
	})
}

texture_get_size :: proc(tex_id: int) -> v2i {
	tex, found := &g_lct.loaded_textures[tex_id]
	ensure(found)
	return {tex.width, tex.height}
}
