package main

import "core:mem"
import dx "vendor:directx/d3d12"
import dxgi "vendor:directx/dxgi"
import dxc "vendor:directx/dxc"
import sa "core:container/small_array"
import "core:strings"
import "core:os"
import "core:sys/windows"
import "core:fmt"

execute_command_list_and_wait :: proc(cmd_list: ^dx.IGraphicsCommandList, queue: ^dx.ICommandQueue) {
	
	fence_value: u64
	fence: ^dx.IFence
	hr := dx_context.device->CreateFence(fence_value, {}, dx.IFence_UUID, (^rawptr)(&fence))
	defer fence->Release()
	fence_value += 1
	
	// close command list and execute
	cmd_list->Close()
	cmdlists := [?]^dx.IGraphicsCommandList{cmd_list}
	queue->ExecuteCommandLists(len(cmdlists), (^^dx.ICommandList)(&cmdlists[0]))
	
	// we signal only after executing the command list.
	// otherwise we are not sure that the gpu is done with the upload resource.
	hr = queue->Signal(fence, fence_value)
	
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

	data, ok_f := os.read_entire_file(shader_filename)
	
	if !ok_f {
		fmt.printfln("could not read file")
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
		fmt.printfln("dxc: errors: %v", error_str)
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

	execute_command_list_and_wait(cmdlist, ct.queue)
	
	return default_res
}

// creates texture then uploads data to it then transitions to a default heap
// it doesn't execute the command list. u have to do that later.
// release upload resources after executing command list.
create_texture_with_data :: proc(
	image_data: [^]byte,
	width: u64,
	height: u32,
	channels: u32,
	format: dxgi.FORMAT,
	pool_textures : ^DXResourcePool,
	pool_upload_heap : ^DXResourcePoolDynamic,
	cmdlist : ^dx.IGraphicsCommandList,
	texture_name := ""
) -> (res: ^dx.IResource) {
	
	ct := &dx_context

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
		MipLevels = 1,
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

	ct.device->GetCopyableFootprints(&texture_desc, 0, 1, 0, &text_footprint, &num_rows, 
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

	for row in 0 ..< height {
		mem.copy(
			texture_map_start_mp[u32(text_footprint.Footprint.RowPitch) * u32(row):],
			image_data[width * u64(channels) * u64(row):],
			int(width * u64(channels)),
		)
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

	cmdlist->CopyTextureRegion(&copy_location_dst, 0, 0, 0, &copy_location_src, nil)
	transition_resource_from_copy_to_read(res, cmdlist)
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
	if debug_index do fmt.printfln("creating view on uber heap: name: %v, index: %v", debug_name, ct.descriptor_count)
	ct.descriptor_count += 1
}
