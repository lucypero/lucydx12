package main

import dx "vendor:directx/d3d12"
import dxgi "vendor:directx/dxgi"
import d3dc "vendor:directx/d3d_compiler"
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

dxc_init :: proc() -> ^dxc.ICompiler3 {
	// todo here
	utils : ^dxc.IUtils
	compiler : ^dxc.ICompiler3
	
	dxc.CreateInstance(dxc.Utils_CLSID, dxc.IUtils_UUID, (^rawptr)(&utils))
	dxc.CreateInstance(dxc.Compiler_CLSID, dxc.ICompiler3_UUID, (^rawptr)(&compiler))
	return compiler
}

// compiles vertex and pixel shader
compile_shader :: proc(compiler: ^dxc.ICompiler3, shader_filename: string) -> (vs, ps: ^dxc.IBlob, ok: bool) {

	c := &dx_context
	
	data, ok_f := os.read_entire_file(shader_filename)
	
	if !ok_f {
		lprintfln("could not read file")
		os.exit(1)
	}

	defer(delete(data))
	
	source_buffer := dxc.Buffer {
		Ptr = &data[0],
		Size = len(data),
		Encoding = dxc.CP_ACP
	}
	
	vs = compile_individual_shader(&source_buffer, compiler, .Vertex)
	ps = compile_individual_shader(&source_buffer, compiler, .Pixel)
	
	return
}

ShaderKind :: enum {
	Vertex,
	Pixel
}

compile_individual_shader :: proc(source_buffer: ^dxc.Buffer, compiler: ^dxc.ICompiler3, shader_kind: ShaderKind) -> ^dxc.IBlob {
	
	arguments := [?]string {
		"-E", "VSMain", // Entry point
		"-T", "vs_6_0", // target profile (pixel shader 6)
		"-Zi", // enable debug info
		"-O3", // Optimization level 3
	}
	
	if shader_kind == .Pixel {
		arguments[1] = "PSMain"
		arguments[3] = "ps_6_0"
	}
	
	arguments_wide : [len(arguments)]windows.wstring
	
	for arg, i in arguments {
		arguments_wide[i] = windows.utf8_to_wstring_alloc(arg, allocator = context.temp_allocator)
	}
	
	results : ^dxc.IResult
	compiler->Compile(source_buffer, &arguments_wide[0], len(arguments_wide), nil, dxc.IOperationResult_UUID, (^rawptr)(&results))
	
	errors : ^dxc.IBlobUtf8
	results->GetOutput(.ERRORS, dxc.IBlobUtf8_UUID, (^rawptr)(&errors), nil)
	if errors != nil && errors->GetStringLength() > 0 {
		error_str := strings.string_from_ptr((^u8)(errors->GetBufferPointer()), int(errors->GetBufferSize()))
		lprintfln("dxc: errors: %v", error_str)
	}
	
	hr : dxc.HRESULT
	results->GetStatus(&hr)
	if hr < 0 {
		os.exit(1)
	}
	
	output_blob : ^dxc.IBlob
	results->GetOutput(.OBJECT, dxc.IBlob_UUID, (^rawptr)(&output_blob), nil)
	return output_blob
}
