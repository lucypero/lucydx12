package main

import "vendor:portmidi"
import "core:reflect"

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


UPLOAD_BUFFER_SIZE :: mem.Gigabyte * 1

DXUploadService :: struct {
	allocation : ^dxma.Allocation,
	allocation_dest: []byte,
	next_allocation_pt: int,
	resource: ^dx.IResource,

	queue_copy: ^dx.ICommandQueue,
	command_allocator_copy: ^dx.ICommandAllocator,
	cmdlist_copy: ^dx.IGraphicsCommandList,
}

dx_upload_init :: proc() {

	ct := &dx_context
	up_service := &ct.upload_service

	// copy command queue and allocator
	check(ct.device->CreateCommandQueue(&{Type = .COPY}, dx.ICommandQueue_UUID, (^rawptr)(&up_service.queue_copy)))
	append(&g_resources_longterm, up_service.queue_copy)

	check(ct.device->CreateCommandAllocator(.COPY, dx.ICommandAllocator_UUID, (^rawptr)(&up_service.command_allocator_copy)))
	append(&g_resources_longterm, up_service.command_allocator_copy)

	check(ct.device->CreateCommandList(
		0,
		.COPY,
		up_service.command_allocator_copy,
		nil,
		dx.ICommandList_UUID,
		(^rawptr)(&up_service.cmdlist_copy),
	))
	append(&g_resources_longterm, up_service.cmdlist_copy)

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
		}, nil, nil, &up_service.allocation, dx.IResource_UUID, nil
	))
	append(&g_resources_longterm, cast(^dx.IUnknown)up_service.allocation)
	up_service.resource = dxma.Allocation_GetResource(up_service.allocation)

	texture_map_start: ^byte
	check(up_service.resource->Map(0, &dx.RANGE{}, cast(^rawptr)&texture_map_start))
	up_service.allocation_dest = slice.from_ptr(texture_map_start, UPLOAD_BUFFER_SIZE)

	// never unmap this. use this resouce to write all data to the GPU.
}

// Order an upload to the gpu. populate resource given
dx_upload_trigger :: proc(resource_dest: ^dx.IResource, data: []byte) {

	ct := &dx_context
	up_service := &ct.upload_service

	// use IGraphicsCommandList::CopyBufferRegion method 


	// CopyBufferRegion:                   proc "system" (this: ^IGraphicsCommandList, 
	//pDstBuffer: ^IResource, DstOffset: u64, pSrcBuffer: ^IResource, SrcOffset: u64, NumBytes: u64),

	// memcpy to up_service.resource here

	if up_service.next_allocation_pt + len(data) > len(up_service.allocation_dest) {
		up_service.next_allocation_pt = 0
	}

	copy(up_service.allocation_dest[up_service.next_allocation_pt:], data)

	up_service.cmdlist_copy->CopyBufferRegion(resource_dest, 0, up_service.resource,
		cast(u64)up_service.next_allocation_pt, cast(u64)len(data))

	
	// execute cmd list
	// put a fence here i think

	up_service.next_allocation_pt += len(data)
}
