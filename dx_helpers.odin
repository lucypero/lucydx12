package main

import "core:c"
import img "vendor:stb/image"
import "core:math/linalg"
import "core:mem/virtual"
import "core:reflect"
import base64 "core:encoding/base64"
import "core:crypto/hash"
import "core:path/filepath"
import "core:encoding/endian"
import "core:slice"
import "core:mem"
import dx "vendor:directx/d3d12"
import dxgi "vendor:directx/dxgi"
import dxc "vendor:directx/dxc"
import "core:strings"
import "core:os"
import "core:sys/windows"
import "core:fmt"
import "base:runtime"
import "core:math"
import dxma "libs/odin-d3d12ma"

/*
transition_resource_from_copy_to_read :: proc(res: ^dx.IResource, cmd_list: ^dx.IGraphicsCommandList) {
	transition_resource(res, cmd_list, {.COPY_DEST}, dx.RESOURCE_STATE_GENERIC_READ)
}
*/

transition_resource :: proc(res: ^dx.IResource, cmd_list: ^dx.IGraphicsCommandList, state_before, state_after: dx.RESOURCE_STATES) {
	barrier : dx.RESOURCE_BARRIER = {
		Type = .TRANSITION,
		Flags = {},
		Transition = {
			pResource = res,
			StateBefore = state_before,
			StateAfter = state_after,
			Subresource = 0
		}
	}
	
	// run resource barrier
	cmd_list->ResourceBarrier(1, &barrier)
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

	ct := &g_dx_context

	opt_clear_value := opt_clear_value

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
	
	allocation : ^dxma.Allocation
	dxma.Allocator_CreateResource(
		ct.dxma_allocator,
	 	&dxma.ALLOCATION_DESC{HeapType = .DEFAULT},
		&texture_desc,
		initial_state,
	 	&opt_clear_value,
		&allocation,
		nil,
		nil
	)
	
	res = dxma.Allocation_GetResource(allocation)
	
	append(pool, cast(^dxgi.IUnknown)allocation)

	return res
}

include_handler : ^dxc.IIncludeHandler

dxc_init :: proc() -> ^dxc.ICompiler3 {
	// todo here
	utils : ^dxc.IUtils
	compiler : ^dxc.ICompiler3
	
	dxc.CreateInstance(dxc.Utils_CLSID, dxc.IUtils_UUID, (^rawptr)(&utils))
	dxc.CreateInstance(dxc.Compiler_CLSID, dxc.ICompiler3_UUID, (^rawptr)(&compiler))
	
	utils->CreateDefaultIncludeHandler(&include_handler)
	
	return compiler
}

// compiles vertex and pixel shader
compile_shader :: proc(compiler: ^dxc.ICompiler3, shader_filename: string) -> (vs, ps: ^dxc.IBlob, ok: bool) {

	data, ok_f := os.read_entire_file_from_path(shader_filename, context.allocator)
	
	if ok_f != os.General_Error.None {
		lprintfln("could not read file")
		os.exit(1)
	}

	defer(delete(data))
	
	if len(data) == 0 do return vs, ps, false
	
	source_buffer := dxc.Buffer {
		Ptr = &data[0],
		Size = len(data),
		Encoding = dxc.CP_ACP
	}
	
	vs, ok = compile_individual_shader(&source_buffer, compiler, .Vertex)
	
	if !ok do return vs, ps, false
	
	ps, ok = compile_individual_shader(&source_buffer, compiler, .Pixel)
	
	if !ok do return vs, ps, false
	
	return vs, ps, true
}

ShaderKind :: enum {
	Vertex,
	Pixel
}

compile_individual_shader :: proc(source_buffer: ^dxc.Buffer, compiler: ^dxc.ICompiler3, shader_kind: ShaderKind) -> (res:^dxc.IBlob, ok: bool) {
	
	arguments := [?]string {
		"-E", "VSMain", // Entry point
		"-T", "vs_6_6", // target profile (pixel shader 6)
		"-Zi", // enable debug info
		"-O3", // Optimization level 3
		// "-I \".\"", // include paths
	}
	
	if shader_kind == .Pixel {
		arguments[1] = "PSMain"
		arguments[3] = "ps_6_6"
	}
	
	arguments_wide : [len(arguments)]windows.wstring
	
	for arg, i in arguments {
		arguments_wide[i] = windows.utf8_to_wstring_alloc(arg, allocator = context.temp_allocator)
	}
	
	results : ^dxc.IResult
	compiler->Compile(source_buffer, &arguments_wide[0], len(arguments_wide), include_handler, dxc.IOperationResult_UUID, (^rawptr)(&results))
	
	errors : ^dxc.IBlobUtf8
	results->GetOutput(.ERRORS, dxc.IBlobUtf8_UUID, (^rawptr)(&errors), nil)
	if errors != nil && errors->GetStringLength() > 0 {
		error_str := strings.string_from_ptr((^u8)(errors->GetBufferPointer()), int(errors->GetBufferSize()))
		lprintfln("dxc: errors: %v", error_str)
	}
	
	output_blob : ^dxc.IBlob
	
	hr : dxc.HRESULT
	results->GetStatus(&hr)
	if hr < 0 {
		return output_blob, false
	}
	
	results->GetOutput(.OBJECT, dxc.IBlob_UUID, (^rawptr)(&output_blob), nil)
	return output_blob, true
}


DDSFile :: struct {
	width: u32,
	height: u32,
	format: dxgi.FORMAT,
	mipmap_data: [][]byte,
}
	
// creates texture then uploads data to it then transitions to a default heap
// it doesn't execute the command list. u have to do that later.
// release upload resources after executing command list.
create_texture_with_data :: proc(
	image_data: [][]byte, // slice of mipmap data
	width: u64,
	height: u32,
	format: dxgi.FORMAT,
	pool_textures : ^DXResourcePool,
	texture_name := ""
) -> (res: ^dx.IResource) {
	
	ct := &g_dx_context
	
	mip_levels := cast(u16)len(image_data)

	texture_desc := dx.RESOURCE_DESC {
		Width = (u64)(width),
		Height = (u32)(height),
		Dimension = .TEXTURE2D,
		Layout = .UNKNOWN,
		Format = format,
		DepthOrArraySize = 1,
		MipLevels = mip_levels,
		SampleDesc = {Count = 1},
	}
	
	allocation : ^dxma.Allocation
	dxma.Allocator_CreateResource(
		pSelf = ct.dxma_allocator,
		pAllocDesc = &dxma.ALLOCATION_DESC{HeapType = .DEFAULT, ExtraHeapFlags = dx.HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES},
		pResourceDesc = &texture_desc,
		InitialResourceState = dx.RESOURCE_STATE_COMMON, // upload queue promotes the resource implicitly, so we set it here as common
		pOptimizedClearValue = nil,
		ppAllocation = &allocation,
		riidResource = nil,
		ppvResource = nil
	)
	res = dxma.Allocation_GetResource(allocation)
	append(pool_textures, cast(^dxgi.IUnknown)allocation)
	
	if len(texture_name) > 0 {
		texture_name_cstring := windows.utf8_to_wstring_alloc(texture_name, allocator = context.temp_allocator)
		res->SetName(texture_name_cstring)
	}
	
	dx_upload_texture_trigger(&g_upload_service, res, image_data, &texture_desc)
	
	return res
}

// copies data to a dx resource. then unmaps the memory
copy_to_buffer :: proc(buffer: ^dx.IResource, data: []byte) {
	gpu_data: rawptr
	hr := buffer->Map(0, &dx.RANGE{}, &gpu_data)
	check(hr, "Failed mapping")
	mem.copy(gpu_data, raw_data(data), len(data))
	buffer->Unmap(0, nil)
}

// creates a SRV for the resource on the uber SRV heap
create_srv_on_uber_heap :: proc(res : ^dx.IResource, debug_index: bool = false, debug_name: string = "",
		srv_desc : ^dx.SHADER_RESOURCE_VIEW_DESC = nil
	) {
	ct := &g_dx_context
	ct.device->CreateShaderResourceView(res, srv_desc, get_descriptor_heap_cpu_address(ct.cbv_srv_uav_heap, ct.descriptor_count))
	uber_heap_count(debug_index, debug_name)
}

create_cbv_on_uber_heap :: proc(
		cbv_desc : ^dx.CONSTANT_BUFFER_VIEW_DESC, debug_index: bool = false, debug_name: string = "") {
	ct := &g_dx_context
	ct.device->CreateConstantBufferView(cbv_desc, get_descriptor_heap_cpu_address(ct.cbv_srv_uav_heap, ct.descriptor_count))
	uber_heap_count(debug_index, debug_name)
}

uber_heap_count :: proc(debug_index: bool, debug_name: string) {
	ct := &g_dx_context
	if debug_index do lprintfln("creating view on uber heap: name: %v, index: %v", debug_name, ct.descriptor_count)
	ct.descriptor_count += 1
}

close_and_execute_cmdlist :: proc() {
	ct := &g_dx_context
	ct.cmdlist->Close()
	cmdlists := [?]^dx.IGraphicsCommandList{ct.cmdlist}
	g_dx_context.queue->ExecuteCommandLists(len(cmdlists), (^^dx.ICommandList)(&cmdlists[0]))
}

// it's a vertex buffer in the upload heap.
// meant for buffers that are modified often.
create_vertex_buffer_upload :: proc(stride_in_bytes, size_in_bytes: u32, pool: ^DXResourcePool) -> VertexBuffer {

	vb: ^dx.IResource

	// For now we'll just store stuff in an upload heap.
	// it's not optimal for most things but it's more practical for me

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
	
	allocation : ^dxma.Allocation
	hr := dxma.Allocator_CreateResource(
		pSelf = g_dx_context.dxma_allocator,
		pAllocDesc = &dxma.ALLOCATION_DESC{HeapType = .UPLOAD},
		pResourceDesc = &resource_desc,
		InitialResourceState = dx.RESOURCE_STATE_GENERIC_READ,
		pOptimizedClearValue = nil,
		ppAllocation = &allocation,
		riidResource = nil,
		ppvResource = nil
	)
	check(hr, "Failed creating vertex buffer")
	vb = dxma.Allocation_GetResource(allocation)
	append(pool, cast(^dxgi.IUnknown)allocation)

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

generate_uv_sphere :: proc(meridians: u32, parallels: u32, allocator: runtime.Allocator) -> ([]v3, []u32) {
	
	expected_verts := 2 + (parallels - 1) * meridians
	verts := make([dynamic]v3, 0, expected_verts, allocator)
	indices := make([dynamic]u32, 0, expected_verts * 6, allocator)
	
	append(&verts, v3{0.0, 1.0, 0})
	
	for j in 0..<parallels - 1 {
		polar : f32 = math.PI * f32(j+1) / f32(parallels)
		sp : f32 = math.sin(polar)
		cp : f32 = math.cos(polar)
		
		for i in 0..<meridians {
			
			azimuth : f32 = 2.0 * math.PI * f32(i) / f32(meridians)
			sa : f32 = math.sin(azimuth)
			ca : f32 = math.cos(azimuth)
			
			append(&verts, v3{sp * ca, cp, sp * sa})
		}
	}
	
	append(&verts, v3{0, -1, 0})
	
	for i in 0..<meridians {
		a : u32 = i + 1
		b : u32 = (i + 1) % meridians + 1
		// add tiangle? what is that
		// 		mesh.addTriangle(0, b, a);
		append(&indices, 0, b, a)
	}
	
	for j in 0..<parallels - 2 {
		aStart : u32 = j * meridians + 1
		bStart : u32 = (j + 1) * meridians + 1
		
		for i in 0..<meridians {
			a : u32 = aStart + i
			a1 : u32 = aStart + (i + 1) % meridians
			b: u32 = bStart + i
			b1: u32 = bStart + (i + 1) % meridians
			// add quad???? what is this
			// mesh.addQuad(a, a1, b1, b);
			
			append(&indices, a, a1, b1)
			append(&indices, a, b1, b)
		}
		
	}
	
	last_ring_start := (parallels - 2) * meridians + 1
	for i in 0..<meridians {
        a := last_ring_start + i
        b := last_ring_start + (i + 1) % meridians
        append(&indices, u32(len(verts) - 1), a, b)
    }
	
	return verts[:], indices[:]
}

// DDS file format docs: https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-header
parse_dds_file :: proc(dds_filepath: string) -> DDSFile {
	
	file_data, err := os.read_entire_file_from_path(dds_filepath, context.temp_allocator)
	assert(err == os.General_Error.None)
	magic_num, ok := endian.get_u32(file_data, .Little)
	assert(ok)
	assert(magic_num == 0x20534444)
	
	PixelFormatFlagsEnum :: enum {
		ALPHAPIXELS = 0,
		DDPF_ALPHA = 1,
		DDPF_FOURCC = 2,
		DDPF_RGB = 6,
		DDPF_YUV = 9,
		DDPF_LUMINANCE = 17,
	}

	DDSPixelFormat :: struct #packed {
		Size: u32,
		Flags: bit_set[PixelFormatFlagsEnum;u32],
		FourCC: u32,
		RGBBitCount: u32,
		RBitMask: u32,
		GBitMask: u32,
		BBitMask: u32,
		ABitMask: u32,
	}
	
	DDSHeader :: struct #packed {
	  Size : u32,
	  Flags: u32,
	  Height: u32,
	  Width: u32,
	  PitchOrLinearSize: u32,
	  Depth: u32,
	  MipMapCount: u32,
	  Reserved1 : [11]u32,
	  pixel_format : DDSPixelFormat,
	  Caps : u32,
	  Caps2: u32,
	  Caps3: u32,
	  Caps4: u32,
	  Reserved2: u32
	}
	
	DDSHeaderDXT10 :: struct #packed {
		dxgi_format: dxgi.FORMAT,
		resource_dimension: dx.RESOURCE_DIMENSION,
		miscFlag: u32,
		arraySize: u32,
		miscFlags2: u32,
	}
	
	advance :u32
	advance += size_of(magic_num)
	header := (^DDSHeader)(raw_data(file_data[advance:]))
	advance += size_of(DDSHeader)
	
	// assume we have the DX10 header.
	assert(.DDPF_FOURCC in header.pixel_format.Flags && header.pixel_format.FourCC == 0x30315844)
	
	header_dx10 := (^DDSHeaderDXT10)(raw_data(file_data[advance:]))
	advance += size_of(DDSHeaderDXT10)
	
	text_format : dxgi.FORMAT = header_dx10.dxgi_format
	format_num :i32 = cast(i32)header_dx10.dxgi_format 
	
	// It's compressed if it's a BC format
	is_compressed : bool = (format_num >= 70 && format_num <= 84) || (format_num >= 94 && format_num <= 99)
	
	bytes_per_block : u32 = 0
	switch format_num {
	case 70..=72: // BC1
	case 79..=81: // BC4
		bytes_per_block = 8
	case: // Rest of BCs
		bytes_per_block = 16
	}
	
	current_w := header.Width
	current_h := header.Height
	
	// TODO handle freeing
	
	dds_output := DDSFile {
		width = header.Width,
		height = header.Height,
		format = text_format,
		mipmap_data = make([][]byte, header.MipMapCount, context.temp_allocator)
	}
	
	for i in 0..<header.MipMapCount {
	    level_size := get_mip_level_size(current_w, current_h, is_compressed, bytes_per_block)
	    
	    // Slice the specific mip level data
	    mip_data := file_data[advance : advance + level_size]
	    
	    // Move cursor and halve dimensions for next level
	    advance += level_size
	    current_w = max(1, current_w >> 1)
	    current_h = max(1, current_h >> 1)
		dds_output.mipmap_data[i] = slice.clone(mip_data, context.temp_allocator)
	}
	
	return dds_output
}

get_mip_level_size :: proc(width, height: u32, format_is_compressed: bool, bytes_per_block_or_pixel: u32) -> u32 {
    w := max(1, width)
    h := max(1, height)
    
    if format_is_compressed {
        // Round up to the nearest 4-pixel block
        bw := max(1, (w + 3) / 4)
        bh := max(1, (h + 3) / 4)
        return bw * bh * bytes_per_block_or_pixel // 8 or 16
    } else {
        return w * h * bytes_per_block_or_pixel // e.g., 4 for RGBA8
    }
}

texture_cache_query :: proc(model_filepath, image_name: string, format: dxgi.FORMAT) -> (texture_out_path: string) {
	
	// test if this exists already
	
	// lprintln("image texture cache miss. creating texture with mipmaps")
	alloc_err : runtime.Allocator_Error
	
	filepath_hash := hash_thing(model_filepath)
	// image_name_hash := hash_thing(image_name)
	
	cache_dir : string
	cache_dir, alloc_err = filepath.join({"cache", filepath_hash}, context.temp_allocator)
	assert(alloc_err == .None)
	
	image_name_dss := strings.concatenate({filepath.stem(image_name), ".dds"}, context.temp_allocator)
	texture_out_path, alloc_err = filepath.join({cache_dir, image_name_dss}, context.temp_allocator)
	assert(alloc_err == os.ERROR_NONE)
	
	// checking if it exists already
	if os.exists(texture_out_path) {
		return texture_out_path
	}
	
	// create dirs
	dir_err := os.make_directory_all(cache_dir)
	
	assert(dir_err == os.ERROR_NONE)
	
	input_image_dir := filepath.dir(model_filepath, context.temp_allocator)
	input_image_path, alloc_err_2 := filepath.join({input_image_dir, image_name}, context.temp_allocator)
	assert(alloc_err_2 == .None)
	
	state, _, _, err := os.process_exec(os.Process_Desc {
		command = {
			"texconv.exe",
			"-f", reflect.enum_string(format), // select output format
			"-m", "0", // all mip levels
			"-y", // overwrite
			"-dx10", // force adding dx10 header
			"-o", cache_dir,
			"-nologo",
			input_image_path
		}
	}, context.temp_allocator)
	
	// lprintln(string(stdout))
	// lprintln(string(stderr))
	assert(state.exited && state.exit_code == 0 && err == os.General_Error.None)
	
	lprintfln("texture %v converted correctly", image_name)
	
	return texture_out_path
}

hash_thing :: proc(thing: string) -> string {
	ENC_TABLE := [64]byte { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '_', }
	thing_hash_temp := hash.hash_string(.SHA256, thing, context.temp_allocator)
	thing_hash, ok := base64.encode(thing_hash_temp, ENC_TABLE, context.temp_allocator)
	assert(ok == .None)
	return thing_hash
}

check :: proc(res: dx.HRESULT, message: string = "dx call error") {
	if (res >= 0) {
		return
	}

	lprintfln("%v. Error code: %0x", message, u32(res))
	os.exit(-1)
}

arena_temp_end :: proc(arena_temp : virtual.Arena_Temp, loc := #caller_location) {
	virtual.arena_temp_end(arena_temp)
}

// Convenience function for clearing used memory in scope
// NOTE: this assumes temp allocator is a virtual arena
@(deferred_out=virtual.arena_temp_end)
TEMP_GUARD :: #force_inline proc(loc := #caller_location) -> (virtual.Arena_Temp, runtime.Source_Code_Location) {
	arena := (^virtual.Arena)(context.temp_allocator.data)
	return virtual.arena_temp_begin(arena, loc), loc
}

// Prints to windows debug, with a lprintln() interface
lprintln :: proc(args: ..any, sep := " ") {
	str: strings.Builder
	strings.builder_init(&str, context.temp_allocator)
	fmt.sbprintln(&str, ..args, sep = sep)
	final_string_c, err := strings.to_cstring(&str)

	if err != .None {
		os.exit(1)
	}

	fmt.print(final_string_c)
	windows.OutputDebugStringA(final_string_c)
}

lprintfln :: proc(fmt_s: string, args: ..any) {
	str: strings.Builder
	strings.builder_init(&str, context.temp_allocator)
	final_string := fmt.sbprintf(&str, fmt_s, ..args, newline = true)

	final_string_c, err := strings.to_cstring(&str)

	if err != .None {
		os.exit(1)
	}

	fmt.print(final_string)
	windows.OutputDebugStringA(final_string_c)
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
	increment := g_dx_context.device->GetDescriptorHandleIncrementSize(desc.Type)
	cpu_descriptor_handle.ptr += uint(offset * increment)
	return
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

load_white_texture :: proc() {
	ct := g_dx_context
	
	w, h, channels : c.int
	image_data := img.load("white.png", &w, &h, &channels, 4)
	defer img.image_free(image_data)
	assert(image_data != nil)
	
	img_data_mipmaps := make([][]byte, 1, context.temp_allocator)
	img_data_mipmaps[0] = slice.clone(slice.from_ptr(image_data, cast(int)(w * h * channels)), context.temp_allocator)
	
	texture_res := create_texture_with_data(img_data_mipmaps[:], u64(w), u32(h), .R8G8B8A8_UNORM, 
		&g_resources_longterm, "white")
	
	// creating srv on uber heap
	ct.device->CreateShaderResourceView(texture_res, nil, get_descriptor_heap_cpu_address(ct.cbv_srv_uav_heap, TEXTURE_WHITE_INDEX))
}

// TODO: implement global tracking here
arena_new :: proc() -> virtual.Arena {
	arena : virtual.Arena
	alloc_err := virtual.arena_init_growing(&arena, mem.Megabyte)
	assert(alloc_err == .None)
	return arena
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
			w = node_i.transform_r[3],
			x = node_i.transform_r[0],
			y = node_i.transform_r[1],
			z = node_i.transform_r[2],
		)
		rot_mat: dxm = linalg.matrix4_from_quaternion_f32(rot_quat)

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

string_append :: proc(the_strs: ..string, allocator: mem.Allocator = context.allocator) -> string {
	sb: strings.Builder
	
	// calc len
	sb_cap : int
	for str in the_strs {
		sb_cap += len(str)
	}
	
	strings.builder_init_len_cap(&sb, 0, sb_cap, allocator)
	
	for str in the_strs {
		strings.write_string(&sb, str)
	}
	
	return strings.to_string(sb)
}

create_structured_buffer_with_data :: proc(
	buffer_name: string,
	pool_resource : ^DXResourcePool,
	buffer_data : []byte
	) -> (res: ^dx.IResource, fence_value: u64) {
	
	ct := &g_dx_context
	
	buffer_desc := dx.RESOURCE_DESC {
		Width = u64(len(buffer_data)),
		Height = 1,
		Dimension = .BUFFER,
		Layout = .ROW_MAJOR,
		Format = .UNKNOWN,
		DepthOrArraySize = 1,
		MipLevels = 1,
		SampleDesc = {Count = 1},
		Flags = {},
	}

	allocation : ^dxma.Allocation
	dxma.Allocator_CreateResource(
		pSelf = ct.dxma_allocator,
		pAllocDesc = &dxma.ALLOCATION_DESC{HeapType = .DEFAULT, ExtraHeapFlags = dx.HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES},
		pResourceDesc = &buffer_desc,
		InitialResourceState = dx.RESOURCE_STATE_COMMON,
		pOptimizedClearValue = nil,
		ppAllocation = &allocation,
		riidResource = nil,
		ppvResource = nil
	)
	
	default_res := dxma.Allocation_GetResource(allocation)
	// already in upload thread. just do the copy.
	
	// TODO: put data thing back to temp allocator (upload allocator), if the upload thread allocated it.
	fence_value = dx_upload_trigger(&g_upload_service, default_res, buffer_data)
	
	append(pool_resource, cast(^dxgi.IUnknown)allocation)
	
	buffer_name_cstring := windows.utf8_to_wstring_alloc(buffer_name, context.temp_allocator)
	default_res->SetName(buffer_name_cstring)
	
	return default_res, fence_value
}
