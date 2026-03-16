#+private file
package main

import "vendor:portmidi"
import "core:reflect"
import "core:thread"
import "core:sync/chan"
import "core:mem/virtual"

import "core:path/filepath"
import "core:encoding/endian"
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
import "core:slice"
import dxma "libs/odin-d3d12ma"
import "core:sync"

UPLOAD_BUFFER_SIZE :: mem.Gigabyte * 1

g_upload_service : DXUploadService

@(private="package")
g_resource_id: u64

DXUploadService :: struct {
	allocation : ^dxma.Allocation,
	allocation_dest: []byte,
	next_allocation_pt: u64,
	resource: ^dx.IResource,
	fence: ^dx.IFence,
	fence_value: u64, // fence value that was set last on the fence.
	queue_copy: ^dx.ICommandQueue,
	command_allocator_copy: ^dx.ICommandAllocator,
	cmdlist_copy: ^dx.IGraphicsCommandList,
}

BufferInput :: []byte
TextureInput :: struct {
	texture_desc: dx.RESOURCE_DESC,
	data: [][]byte
}

@(private="package")
DXUploadInput :: struct {
	resource_id: u64,
	dest_resource: ^dx.IResource,
	data: union #no_nil {
		BufferInput,
		TextureInput
	}
}

@(private="package")
DXUploadOutput :: struct {
	resource_id: u64,
	fence_value: u64
}

@(private="package")
dx_upload_init :: proc() {

	ct := &g_dx_context

	ct.device->CreateFence(0, {}, dx.IFence_UUID, (^rawptr)(&g_upload_service.fence))
	append(&g_resources_longterm, g_upload_service.fence)
	g_upload_service.fence->SetName("Upload Fence")

	// copy command queue and allocator
	check(ct.device->CreateCommandQueue(&{Type = .COPY}, dx.ICommandQueue_UUID, (^rawptr)(&g_upload_service.queue_copy)))
	append(&g_resources_longterm, g_upload_service.queue_copy)
	g_upload_service.queue_copy->SetName("Upload command queue")

	check(ct.device->CreateCommandAllocator(.COPY, dx.ICommandAllocator_UUID, (^rawptr)(&g_upload_service.command_allocator_copy)))
	append(&g_resources_longterm, g_upload_service.command_allocator_copy)
	g_upload_service.command_allocator_copy->SetName("Upload command allocator")
	

	check(ct.device->CreateCommandList(
			0,
			.COPY,
			g_upload_service.command_allocator_copy,
			nil,
			dx.ICommandList_UUID,
			(^rawptr)(&g_upload_service.cmdlist_copy),
		))
	append(&g_resources_longterm, g_upload_service.cmdlist_copy)
	g_upload_service.cmdlist_copy->SetName("Upload command list")

	g_upload_service.cmdlist_copy->Close()

	check(dxma.Allocator_CreateResource(
			ct.dxma_allocator,
			&{
				HeapType = .UPLOAD,
			},
			&{
				Dimension = .BUFFER,
				Alignment = 0,
				Width = UPLOAD_BUFFER_SIZE,
				Height = 1,
				DepthOrArraySize = 1,
				MipLevels = 1,
				Format = .UNKNOWN,
				SampleDesc = {Count = 1},
				Layout = .ROW_MAJOR
			}, nil, nil, &g_upload_service.allocation, dx.IResource_UUID, nil
		))
	append(&g_resources_longterm, cast(^dx.IUnknown)g_upload_service.allocation)
	g_upload_service.resource = dxma.Allocation_GetResource(g_upload_service.allocation)

	texture_map_start: ^byte
	check(g_upload_service.resource->Map(0, &dx.RANGE{}, cast(^rawptr)&texture_map_start))
	g_upload_service.allocation_dest = slice.from_ptr(texture_map_start, UPLOAD_BUFFER_SIZE)

	// never unmap this. use this resouce to write all data to the GPU.
}

// Order an upload to the gpu. populate resource given
// this can run on a worker thread i think.
dx_upload_trigger :: proc(up_service: ^DXUploadService, resource_dest : ^dx.IResource, data: []byte, temp_allocator: mem.Allocator) -> u64 {
	if up_service.next_allocation_pt + cast(u64)len(data) > cast(u64)len(up_service.allocation_dest) {
		up_service.next_allocation_pt = 0
	}

	copy(up_service.allocation_dest[up_service.next_allocation_pt:], data)
	up_service.cmdlist_copy->Reset(up_service.command_allocator_copy, nil)
	up_service.cmdlist_copy->CopyBufferRegion(resource_dest, 0, up_service.resource,
		up_service.next_allocation_pt, cast(u64)len(data))
	up_service.next_allocation_pt += cast(u64)len(data)
	up_service.cmdlist_copy->Close()
	cmdlists := [?]^dx.IGraphicsCommandList{up_service.cmdlist_copy}
	up_service.queue_copy->ExecuteCommandLists(len(cmdlists), (^^dx.ICommandList)(&cmdlists[0]))
	
	up_service.fence_value += 1
	up_service.queue_copy->Signal(up_service.fence, up_service.fence_value)
	return up_service.fence_value
}

// Order a texture upload to the gpu. populate resource given
// this can run on a worker thread i think.
dx_upload_texture_trigger :: proc(up_service: ^DXUploadService, resource_dest : ^dx.IResource, 
	image_data: [][]byte, // slice of mipmap data
	texture_desc : ^dx.RESOURCE_DESC,
	temp_allocator: mem.Allocator
) -> u64 {
	
	mip_levels := cast(u16)len(image_data)
	
	// getting data from texture that we'll use later
	text_footprint := make([]dx.PLACED_SUBRESOURCE_FOOTPRINT, mip_levels, temp_allocator)
	num_rows := make([]u32, mip_levels, temp_allocator)
	row_size := make([]u64, mip_levels, temp_allocator)
	text_bytes: u64

	g_dx_context.device->GetCopyableFootprints(texture_desc, 0, cast(u32)mip_levels, 0, &text_footprint[0], &num_rows[0], 
		&row_size[0], &text_bytes)

	if up_service.next_allocation_pt + text_bytes > cast(u64)len(up_service.allocation_dest) {
		up_service.next_allocation_pt = 0
	}
	
	up_service.cmdlist_copy->Reset(up_service.command_allocator_copy, nil)

	// start copy
	{
		
		texture_map_start_mp := up_service.allocation_dest[up_service.next_allocation_pt:]

		// copying stuff here.. how do i 
		for mip in 0 ..< mip_levels {
			fp := text_footprint[mip].Footprint
			mip_row_size := row_size[mip]
			for row in 0 ..< num_rows[mip] {
				copy(
					texture_map_start_mp[u64(fp.RowPitch) * u64(row) + text_footprint[mip].Offset:][:mip_row_size],
					image_data[mip][mip_row_size * u64(row):][:mip_row_size],
				)
			}
		}

		copy_location_src := dx.TEXTURE_COPY_LOCATION {
			pResource = up_service.resource,
			Type = .PLACED_FOOTPRINT,
			PlacedFootprint = text_footprint[0],
		}
		
		copy_location_dst := dx.TEXTURE_COPY_LOCATION {
			pResource = resource_dest,
			Type = .SUBRESOURCE_INDEX,
			SubresourceIndex = 0,
		}

		for i in 0..<mip_levels {
			copy_location_src.PlacedFootprint = text_footprint[i]
			copy_location_src.PlacedFootprint.Offset += up_service.next_allocation_pt
			copy_location_dst.SubresourceIndex = auto_cast i
			up_service.cmdlist_copy->CopyTextureRegion(&copy_location_dst, 0, 0, 0, &copy_location_src, nil)
		}
	} // end copy

	up_service.next_allocation_pt += text_bytes
	up_service.cmdlist_copy->Close()
	cmdlists := [?]^dx.IGraphicsCommandList{up_service.cmdlist_copy}
	up_service.queue_copy->ExecuteCommandLists(len(cmdlists), (^^dx.ICommandList)(&cmdlists[0]))
	
	up_service.fence_value += 1
	up_service.queue_copy->Signal(up_service.fence, up_service.fence_value)
	return up_service.fence_value
}

@(private="package")
upload_thread_start :: proc(recv_chan: chan.Chan(DXUploadInput, .Recv), send_chan: chan.Chan(DXUploadOutput, .Send) ) {
	
	context.allocator = mem.tracking_allocator(&g_track)
	
	// make temp allocator for upload thread
	upload_temp_arena := arena_new()
	upload_temp_allocator := virtual.arena_allocator(&upload_temp_arena)
	
	for {
		upload_input, ok := chan.recv(recv_chan)
		if !ok {
			break
		}
		
		fence_val : u64
		
		switch &t in upload_input.data {
		case BufferInput: // buffer
			fence_val = dx_upload_trigger(&g_upload_service, upload_input.dest_resource, t, temp_allocator = upload_temp_allocator)
			// freeing data
			delete(t)
		case TextureInput: // texture
			fence_val = dx_upload_texture_trigger(&g_upload_service, upload_input.dest_resource, t.data, &t.texture_desc, temp_allocator = upload_temp_allocator)
			// freeing data
			for mip in t.data {
				delete(mip)
			}
			delete(t.data)
		}
	
		chan.send(send_chan, DXUploadOutput{
			resource_id = upload_input.resource_id,
			fence_value = fence_val,
		})
		
		virtual.arena_free_all(&upload_temp_arena)
	}
}

// schedule upload
@(private="package")
dx_upload_order_buffer :: proc(dest_res: ^dx.IResource, data: []byte) -> u64 {
	return dx_upload_thing(dest_res, DXUploadInput {
		dest_resource = dest_res,
		data = data,
		resource_id = g_resource_id
	})
}

@(private="package")
dx_upload_order_texture :: proc(dest_res: ^dx.IResource, data: [][]byte, texture_desc: dx.RESOURCE_DESC) -> u64 {
	return dx_upload_thing(dest_res, DXUploadInput {
		data = TextureInput {
			data = data,
			texture_desc = texture_desc
		},
		resource_id = g_resource_id,
		dest_resource = dest_res
	})
}

@(private="package")
queue_wait_on_upload_fence :: proc(queue: ^dx.ICommandQueue, fence_value: u64) {
	queue->Wait(g_upload_service.fence, fence_value)
}

dx_upload_thing :: proc(dest_res: ^dx.IResource, input: DXUploadInput) -> u64 {
	
	for {
		did_send := chan.try_send(g_channel_upload_send, input)
		if !did_send {
			flush_receive_channel()
		}
		else {
			break
		}
	}
	
	g_resource_id += 1
	return g_resource_id - 1
}

@(private="package")
flush_receive_channel :: proc() {
	for {
		if dx_upload_output, ok := chan.try_recv(g_channel_upload_recv); ok {
			append(&g_upload_thread_inbox, dx_upload_output)
			lprintfln("resorce %v is uploaded!", dx_upload_output.resource_id)
		} else {
			break
		}
	}
}
