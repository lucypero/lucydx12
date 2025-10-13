package main

import dx "vendor:directx/d3d12"
import "core:container/small_array"

// Weird thing just to get dear imgui to work

DescriptorHeapAllocator :: struct {
	heap : ^dx.IDescriptorHeap,
	heap_type : dx.DESCRIPTOR_HEAP_TYPE,
	heap_start_cpu: dx.CPU_DESCRIPTOR_HANDLE,
	heap_start_gpu: dx.GPU_DESCRIPTOR_HANDLE,
	heap_handle_increment: u32,
	free_indices: small_array.Small_Array(10, u32),
}

descriptor_heap_allocator_create :: proc(heap: ^dx.IDescriptorHeap,
									 heap_type: dx.DESCRIPTOR_HEAP_TYPE) -> (ha: DescriptorHeapAllocator) {

	ha.heap = heap
	ha.heap_type = heap_type

	ha.heap->GetCPUDescriptorHandleForHeapStart(&ha.heap_start_cpu)
	ha.heap->GetGPUDescriptorHandleForHeapStart(&ha.heap_start_gpu)

	ha.heap_handle_increment = dx_context.device->GetDescriptorHandleIncrementSize(ha.heap_type)

	desc : dx.DESCRIPTOR_HEAP_DESC
	
	ha.heap->GetDesc(&desc)

	for n:= desc.NumDescriptors; n > 0; n-=1 {
		small_array.push_back(&ha.free_indices, n - 1)
	}

	return ha
}

descriptor_heap_allocator_alloc :: proc(ha: ^DescriptorHeapAllocator) -> 
		(cpu_desc_handle: dx.CPU_DESCRIPTOR_HANDLE, gpu_desc_handle: dx.GPU_DESCRIPTOR_HANDLE) {

	n := small_array.pop_back(&ha.free_indices)

	cpu_desc_handle.ptr = ha.heap_start_cpu.ptr + uint(n * ha.heap_handle_increment)
	gpu_desc_handle.ptr = ha.heap_start_gpu.ptr + u64(n * ha.heap_handle_increment)

	return
}

descriptor_heap_allocator_free :: proc(ha: ^DescriptorHeapAllocator, cpu_desc_handle: dx.CPU_DESCRIPTOR_HANDLE, gpu_desc_handle: dx.GPU_DESCRIPTOR_HANDLE) {

	cpu_indx := u32(cpu_desc_handle.ptr - ha.heap_start_cpu.ptr) / ha.heap_handle_increment
	gpu_indx := u32(gpu_desc_handle.ptr - ha.heap_start_gpu.ptr) / ha.heap_handle_increment

	assert(cpu_indx == gpu_indx)

	small_array.push_back(&ha.free_indices, cpu_indx)
}
