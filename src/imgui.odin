package main

import "core:fmt"
import "core:strings"
import "base:runtime"
import dx "vendor:directx/d3d12"
import sdl "vendor:sdl2"

// imgui
import im "../libs/odin-imgui"
// imgui sdl2 implementation
import "../libs/odin-imgui/imgui_impl_sdl2"
// imgui dx12 implementation
import "../libs/odin-imgui/imgui_impl_dx12"

imgui_init :: proc(window: ^sdl.Window, pool: ^DXResourcePool) {

	// initting dear imgui
	im.CHECKVERSION()
	im.CreateContext()
	io := im.GetIO()

	io.ConfigFlags += {.NavEnableKeyboard, .NavEnableGamepad}
	io.ConfigFlags += {.DockingEnable}
	io.ConfigFlags += {.ViewportsEnable}

	style := im.GetStyle()
	style.WindowRounding = 0
	style.Colors[im.Col.WindowBg].w = 1

	im.StyleColorsDark()

	imgui_impl_sdl2.InitForD3D(window)

	// create a shader resource view  heap (srv)
	ctd := &g_dx_core

	allocfn := proc "c" (
		info: ^imgui_impl_dx12.InitInfo,
		out_cpu_desc_handle: ^dx.CPU_DESCRIPTOR_HANDLE,
		out_gpu_desc_handle: ^dx.GPU_DESCRIPTOR_HANDLE,
	) {
		context = runtime.default_context()
		h := &g_dx_core.heap_cbv_srv_uav

		next_i := g_dx_core.heap_cbv_srv_uav.next_descriptor_index
		out_cpu_desc_handle.ptr = h.heap_start_cpu.ptr + cast(uint)(cast(u32)next_i * h.heap_handle_increment)
		out_gpu_desc_handle.ptr = h.heap_start_gpu.ptr + cast(u64)(cast(u32)next_i * h.heap_handle_increment)

		uber_heap_count(h)
	}

	freefn := proc "c" (
		info: ^imgui_impl_dx12.InitInfo,
		cpu_desc_handle: dx.CPU_DESCRIPTOR_HANDLE,
		gpu_desc_handle: dx.GPU_DESCRIPTOR_HANDLE,
	) {
		// NO-OP
		// TODO: implement a free list on uber heap! to handle freeing
		// context = runtime.default_context()
		// descriptor_heap_allocator_free(&g_imgui_context.imgui_allocator, cpu_desc_handle, gpu_desc_handle)
	}

	dx12_init := imgui_impl_dx12.InitInfo {
		Device = ctd.device,
		CommandQueue = ctd.queue,
		// not sure what this is
		NumFramesInFlight = 2,
		RTVFormat = .R8G8B8A8_UNORM,
		DSVFormat = .D32_FLOAT,
		SrvDescriptorHeap = g_dx_core.heap_cbv_srv_uav.heap,
		SrvDescriptorAllocFn = allocfn,
		SrvDescriptorFreeFn = freefn,
	}

	imgui_impl_dx12.Init(&dx12_init)
}

imgui_destroy :: proc() {
	imgui_impl_sdl2.Shutdown() // here
	imgui_impl_dx12.Shutdown()
	im.DestroyContext()
}

// call this right before swapchain present
imgui_end_frame :: proc() {
	im.Render()
	// setting imgui's descriptor heap
	// if i don't do this, it errors out. seems like RenderDrawData doesn't set it
	//  by itself
	g_dx_core.cmdlist->SetDescriptorHeaps(1, &g_dx_core.heap_cbv_srv_uav.heap)
	imgui_impl_dx12.RenderDrawData(im.GetDrawData(), g_dx_core.cmdlist)
	io := im.GetIO()
	if .ViewportsEnable in io.ConfigFlags {
		im.UpdatePlatformWindows()
		im.RenderPlatformWindowsDefault()
	}
}

imgui_start_frame :: proc() {
	imgui_impl_dx12.NewFrame()
	imgui_impl_sdl2.NewFrame()
	im.NewFrame()
}

// helper functions
imgui_do_text :: proc(format:string, args: ..any) {
	sb := strings.builder_make_len_cap(0, 0, context.temp_allocator)
	fmt.sbprintfln(&sb, format, ..args)
	cst, _ := strings.to_cstring(&sb)
	im.Text(cst)
}

imgui_process_sdl_event :: proc(event: ^sdl.Event) {
	imgui_impl_sdl2.ProcessEvent(event)
}
