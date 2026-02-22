package main

import "core:encoding/endian"
import "core:slice"
import "core:mem"
import dx "vendor:directx/d3d12"
import dxgi "vendor:directx/dxgi"
import dxc "vendor:directx/dxc"
import sa "core:container/small_array"
import "core:strings"
import "core:os"
import "core:sys/windows"
import "core:fmt"
import "base:runtime"
import "core:math"

execute_command_list_and_wait :: proc() {
	
	ct := &dx_context
	
	
	fence_value: u64
	fence: ^dx.IFence
	hr := ct.device->CreateFence(fence_value, {}, dx.IFence_UUID, (^rawptr)(&fence))
	defer fence->Release()
	fence_value += 1
	
	// close command list and execute
	ct.cmdlist->Close()
	cmdlists := [?]^dx.IGraphicsCommandList{ct.cmdlist}
	ct.queue->ExecuteCommandLists(len(cmdlists), (^^dx.ICommandList)(&cmdlists[0]))
	
	// we signal only after executing the command list.
	// otherwise we are not sure that the gpu is done with the upload resource.
	hr = ct.queue->Signal(fence, fence_value)
	
	// 4. Wait for the GPU to reach the signal point.
	// First, create an event handle.
	fence_event := windows.CreateEventW(nil, false, false, nil)
	
	if fence_event == nil {
		fmt.eprintln("Failed to create fence event")
		os.exit(1)
	}
	
	completed := fence->GetCompletedValue()
	
	if completed < fence_value {
		// the gpu is not finished yet , so we wait
		fence->SetEventOnCompletion(fence_value, fence_event)
		windows.WaitForSingleObject(fence_event, windows.INFINITE)
	}
	
}

transition_resource_from_copy_to_read :: proc(res: ^dx.IResource, cmd_list: ^dx.IGraphicsCommandList) {
	transition_resource(res, cmd_list, {.COPY_DEST}, dx.RESOURCE_STATE_GENERIC_READ)
}

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

	ct.device->CreateCommittedResource(
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

create_structured_buffer_with_data :: proc(
	cmdlist : ^dx.IGraphicsCommandList,
	buffer_name: string,
	pool_resource : ^DXResourcePool,
	buffer_data : []byte
	) -> ^dx.IResource {
	
	ct := &dx_context
	
	heap_properties := dx.HEAP_PROPERTIES {
		Type = .DEFAULT,
	}

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

	default_res: ^dx.IResource

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
	
	buffer_name_cstring := windows.utf8_to_wstring_alloc(buffer_name, allocator = context.temp_allocator)
	default_res->SetName(buffer_name_cstring)
	sa.push(pool_resource, default_res)

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
	defer upload_res->Release()

	// Copying data from cpu to upload resource
	copy_to_buffer(upload_res, buffer_data)

	// this might be problematic
	cmdlist->Reset(ct.command_allocator, ct.pipeline_gbuffer)
	
	cmdlist->CopyResource(default_res, upload_res)

	// transition resource to shader readable.
	transition_resource_from_copy_to_read(default_res, cmdlist)

	execute_command_list_and_wait()
	
	return default_res
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
	channels: u32,
	format: dxgi.FORMAT,
	pool_textures : ^DXResourcePool,
	pool_upload_heap : ^DXResourcePoolDynamic,
	texture_name := ""
) -> (res: ^dx.IResource) {
	
	ct := &dx_context
	
	mip_levels := cast(u16)len(image_data)

	// default heap (this is where the final texture will reside)
	heap_properties := dx.HEAP_PROPERTIES {
		Type = .DEFAULT,
	}
	
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

	hr := ct.device->CreateCommittedResource(
		&heap_properties,
		dx.HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES,
		&texture_desc,
		{.COPY_DEST},
		nil,
		dx.IResource_UUID,
		(^rawptr)(&res),
	)

	check(hr, "failed creating texture")
	
	if len(texture_name) > 0 {
		texture_name_cstring := windows.utf8_to_wstring_alloc(texture_name, allocator = context.temp_allocator)
		res->SetName(texture_name_cstring)
	}
	
	sa.push(pool_textures, res)

	// getting data from texture that we'll use later
	text_footprint: dx.PLACED_SUBRESOURCE_FOOTPRINT
	text_bytes: u64
	num_rows: u32
	row_size: u64

	// GetCopyableFootprints:            proc "system" (this: ^IDevice, pResourceDesc: ^RESOURCE_DESC, 
	// FirstSubresource: u32, NumSubresources: u32, BaseOffset: u64, pLayouts: [^]PLACED_SUBRESOURCE_FOOTPRINT,
	// pNumRows: [^]u32, pRowSizeInBytes: [^]u64, pTotalBytes: ^u64),
	
	ct.device->GetCopyableFootprints(&texture_desc, 0, auto_cast mip_levels, 0, &text_footprint, &num_rows, 
		&row_size, &text_bytes)

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

	hr = ct.device->CreateCommittedResource(
		&heap_properties,
		dx.HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES,
		&upload_desc,
		dx.RESOURCE_STATE_GENERIC_READ,
		nil,
		dx.IResource_UUID,
		(^rawptr)(&texture_upload),
	)

	check(hr, "failed creating upload texture")
	append(pool_upload_heap, texture_upload)

	// here you do a Map and you memcpy the data to the upload resource.
	// you'll have to use an image library here to get the pixel data of an image.

	texture_map_start: rawptr
	texture_upload->Map(0, &dx.RANGE{}, &texture_map_start)
	texture_map_start_mp: [^]u8 = auto_cast texture_map_start
	
	offset: int
	
	for mip in 0 ..< mip_levels {
		for row in 0 ..< height {
			source_data : [^]byte = slice.as_ptr(image_data[mip])
			mem.copy(
				texture_map_start_mp[u32(text_footprint.Footprint.RowPitch) * u32(row) + cast(u32)offset:],
				source_data[width * u64(channels) * u64(row):],
				int(width * u64(channels)),
			)
		}
		offset += len(image_data[mip])
	}
	
	// here you send the gpu command to copy the data to the texture resource.
	
	copy_location_src := dx.TEXTURE_COPY_LOCATION {
		pResource = texture_upload,
		Type = .PLACED_FOOTPRINT,
		PlacedFootprint = text_footprint,
	}

	copy_location_dst := dx.TEXTURE_COPY_LOCATION {
		pResource = res,
		Type = .SUBRESOURCE_INDEX,
		SubresourceIndex = 0,
	}
	
	for i in 0..<mip_levels {
		copy_location_dst.SubresourceIndex = auto_cast i
		ct.cmdlist->CopyTextureRegion(&copy_location_dst, 0, 0, 0, &copy_location_src, nil)
	}
	
	transition_resource_from_copy_to_read(res, ct.cmdlist)
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
	ct := &dx_context
	ct.device->CreateShaderResourceView(res, srv_desc, get_descriptor_heap_cpu_address(ct.cbv_srv_uav_heap, ct.descriptor_count))
	uber_heap_count(debug_index, debug_name)
}

create_cbv_on_uber_heap :: proc(
		cbv_desc : ^dx.CONSTANT_BUFFER_VIEW_DESC, debug_index: bool = false, debug_name: string = "") {
	ct := &dx_context
	ct.device->CreateConstantBufferView(cbv_desc, get_descriptor_heap_cpu_address(ct.cbv_srv_uav_heap, ct.descriptor_count))
	uber_heap_count(debug_index, debug_name)
}

uber_heap_count :: proc(debug_index: bool, debug_name: string) {
	ct := &dx_context
	if debug_index do lprintfln("creating view on uber heap: name: %v, index: %v", debug_name, ct.descriptor_count)
	ct.descriptor_count += 1
}

close_and_execute_cmdlist :: proc() {
	ct := &dx_context
	ct.cmdlist->Close()
	cmdlists := [?]^dx.IGraphicsCommandList{ct.cmdlist}
	dx_context.queue->ExecuteCommandLists(len(cmdlists), (^^dx.ICommandList)(&cmdlists[0]))
}


// it's a vertex buffer in the upload heap.
// meant for buffers that are modified often.
create_vertex_buffer_upload :: proc(stride_in_bytes, size_in_bytes: u32, pool: ^DXResourcePool) -> VertexBuffer {

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

generate_uv_sphere :: proc(meridians: u32, parallels: u32, allocator: runtime.Allocator) -> ([]v3, []u32) {
	
	expected_verts := 2 + (parallels - 1) * meridians
	verts := make([dynamic]v3, 0, expected_verts, allocator = allocator)
	indices := make([dynamic]u32, 0, expected_verts * 6, allocator = allocator)
	
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
// TODO: proper memory handling
// TODO: try to not hold so much memory at once without need. use checkpoints
parse_dds_file :: proc(dds_filepath: string) -> DDSFile {
	
	file_data, err := os.read_entire_file_from_path(dds_filepath, context.temp_allocator)
	
	magic_num, ok := endian.get_u32(file_data, .Little)
	
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
	  magic_num: u32,
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
	
	// lprintfln("it's a dds file!")
	
	advance :u32
	
	header := (^DDSHeader)(raw_data(file_data))
	
	advance += size_of(DDSHeader)
	
	// If the DDS_PIXELFORMAT dwFlags is set to DDPF_FOURCC and dwFourCC is set to "DX10" an additional DDS_HEADER_DXT10 structure will be present to accommodate texture arrays or DXGI formats that cannot be expressed as an RGB pixel format such as floating point formats, sRGB formats etc. When the DDS_HEADER_DXT10 structure is present the entire data description will looks like this.
	// flags_u32 := cast(^u32)(&header.pixel_format.Flags)
	
	is_compressed := false
	bytes_per_block : u32 = 2
	text_format: dxgi.FORMAT
	
	if .DDPF_FOURCC in header.pixel_format.Flags && header.pixel_format.FourCC == 0x30315844 { 
		// it has the dx10 header.
		
		header_dx10 := (^DDSHeaderDXT10)(raw_data(file_data[advance:]))
		advance += size_of(DDSHeaderDXT10)
		
		// lprintfln("it has a dx10 header!!!")
		
// 		For an uncompressed texture, use the DDSD_PITCH and DDPF_RGB flags; 
// for a compressed texture, use the DDSD_LINEARSIZE and DDPF_FOURCC flags. For a mipmapped texture, use the DDSD_MIPMAPCOUNT, DDSCAPS_MIPMAP, and DDSCAPS_COMPLEX flags also as well as the mipmap count member. If mipmaps are generated, all levels down to 1-by-1 are usually written.
// For a compressed texture, the size of each mipmap level image is typically one-fourth the size of the previous, with a minimum of 8 (DXT1) or 16 (DXT2-5) bytes (for square textures). Use the following formula to calculate the size of each level for a non-square texture:

		text_format = header_dx10.dxgi_format
		format_num :i32 = cast(i32)header_dx10.dxgi_format 
		
		
		if (format_num >= 70 && format_num <= 84) || (format_num >= 94 && format_num <= 99) {
			// compressed
			is_compressed = true
		} else {
			// uncompressed
			is_compressed = false
		}
		
		switch format_num {
		case 70..=72: // BC1
		case 79..=81: // BC4
			bytes_per_block = 8
		case: // Rest of BCs
			bytes_per_block = 16
		}
		
	} else {
		// TODO: not handled yet. but all textures u use should have the dx10 header. so we're probably good
		assert(false)
	}
	
	current_w := header.Width
	current_h := header.Height
	
	// TODO handle freeing
	
	dds_output := DDSFile {
		width = header.Width,
		height = header.Height,
		format = text_format,
		mipmap_data = make([][]byte, header.MipMapCount)
	}
	
	for i in 0..<header.MipMapCount {
	    level_size := get_mip_level_size(current_w, current_h, is_compressed, bytes_per_block)
	    
	    // Slice the specific mip level data
	    mip_data := file_data[advance : advance + level_size]
	    
	    // Move cursor and halve dimensions for next level
	    advance += level_size
	    current_w = max(1, current_w >> 1)
	    current_h = max(1, current_h >> 1)
					
		// TODO: this memory lives in the allocator used for the file read
		dds_output.mipmap_data[i] = mip_data
	    
	    // lprintfln("Level %v: %v bytes", i, len(mip_data))
	}
	
	// here is the data:
	// file_data[advance:]
	// lprintfln("check the header")
	
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
