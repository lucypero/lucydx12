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

ImguiContext :: struct {
	imgui_descriptor_heap: ^dx.IDescriptorHeap,
	imgui_allocator: DescriptorHeapAllocator,
}

g_imgui_context: ImguiContext

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

	// creating descriptor heap

	// if it goes above 3, we are dead
	srv_descriptor_heap_desc := dx.DESCRIPTOR_HEAP_DESC {
		NumDescriptors = 3,
		Type = .CBV_SRV_UAV,
		Flags = {.SHADER_VISIBLE},
	}

	hr := ctd.device->CreateDescriptorHeap(
		&srv_descriptor_heap_desc,
		dx.IDescriptorHeap_UUID,
		(^rawptr)(&g_imgui_context.imgui_descriptor_heap),
	)
	check(hr, "could ont create imgui descriptor heap")
	g_imgui_context.imgui_descriptor_heap->SetName("imgui's cbv srv uav descriptor heap")
	append(pool, g_imgui_context.imgui_descriptor_heap)

	g_imgui_context.imgui_allocator = descriptor_heap_allocator_create(g_imgui_context.imgui_descriptor_heap, .CBV_SRV_UAV)

	allocfn := proc "c" (
		info: ^imgui_impl_dx12.InitInfo,
		out_cpu_desc_handle: ^dx.CPU_DESCRIPTOR_HANDLE,
		out_gpu_desc_handle: ^dx.GPU_DESCRIPTOR_HANDLE,
	) {
		context = runtime.default_context()
		// they want a global here.. what do i do
		cpu, gpu := descriptor_heap_allocator_alloc(&g_imgui_context.imgui_allocator)
		out_cpu_desc_handle.ptr = cpu.ptr
		out_gpu_desc_handle.ptr = gpu.ptr
	}

	freefn := proc "c" (
		info: ^imgui_impl_dx12.InitInfo,
		cpu_desc_handle: dx.CPU_DESCRIPTOR_HANDLE,
		gpu_desc_handle: dx.GPU_DESCRIPTOR_HANDLE,
	) {
		context = runtime.default_context()
		descriptor_heap_allocator_free(&g_imgui_context.imgui_allocator, cpu_desc_handle, gpu_desc_handle)
	}

	dx12_init := imgui_impl_dx12.InitInfo {
		Device = ctd.device,
		CommandQueue = ctd.queue,
		// not sure what this is
		NumFramesInFlight = 2,
		RTVFormat = .R8G8B8A8_UNORM,
		DSVFormat = .D32_FLOAT,
		SrvDescriptorHeap = g_imgui_context.imgui_descriptor_heap,
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
	g_dx_core.cmdlist->SetDescriptorHeaps(1, &g_imgui_context.imgui_descriptor_heap)
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
