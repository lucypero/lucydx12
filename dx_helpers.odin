package main

import dx "vendor:directx/d3d12"
import d3dc "vendor:directx/d3d_compiler"
import dxgi "vendor:directx/dxgi"
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
