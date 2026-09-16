#+build windows
package imgui_impl_dx12

import imgui "../"
import "vendor:directx/d3d12"
import "vendor:directx/dxgi"

when      ODIN_OS == .Windows { foreign import lib "../imgui_windows_x64.lib" } else when ODIN_OS == .Linux   { foreign import lib "../imgui_linux_x64.a" } else when ODIN_OS == .Darwin  {
when ODIN_ARCH == .amd64 { foreign import lib "../imgui_darwin_x64.a" } else { foreign import lib "../imgui_darwin_arm64.a" }
}

// imgui_impl_dx12.h
// Last checked `v1.92.8-docking` (2a1b69f)

// Initialization data, for ImGui_ImplDX12_Init()
InitInfo :: struct {
	Device:            ^d3d12.IDevice,
	CommandQueue:      ^d3d12.ICommandQueue, // Command queue used for queuing texture uploads.
	NumFramesInFlight: i32,
	RTVFormat:         dxgi.FORMAT,          // RenderTarget format.
	DSVFormat:         dxgi.FORMAT,          // DepthStencilView format.
	UserData:          rawptr,

	// Allocating SRV descriptors for textures is up to the application, so we provide callbacks.
	// (current version of the backend will only allocate one descriptor, from 1.92 the backend will need to allocate more)
	SrvDescriptorHeap:    ^d3d12.IDescriptorHeap,
	SrvDescriptorAllocFn: proc "c" (info: ^InitInfo, out_cpu_desc_handle: ^d3d12.CPU_DESCRIPTOR_HANDLE, out_gpu_desc_handle: ^d3d12.GPU_DESCRIPTOR_HANDLE),
	SrvDescriptorFreeFn:  proc "c" (info: ^InitInfo, cpu_desc_handle: d3d12.CPU_DESCRIPTOR_HANDLE, gpu_desc_handle: d3d12.GPU_DESCRIPTOR_HANDLE),
}

@(link_prefix="ImGui_ImplDX12_")
foreign lib {
	Init           :: proc(info: ^InitInfo) -> bool ---
	Shutdown       :: proc() ---
	NewFrame       :: proc() ---
	RenderDrawData :: proc(draw_data: ^imgui.DrawData, graphics_command_list: ^d3d12.IGraphicsCommandList) ---

	// Use if you want to reset your rendering device without losing Dear ImGui state.
	CreateDeviceObjects     :: proc() -> bool ---
	InvalidateDeviceObjects :: proc() ---

	// (Advanced) Use e.g. if you need to precisely control the timing of texture updates (e.g. for staged rendering), by setting ImDrawData::Textures = nullptr to handle this manually.
	UpdateTexture :: proc(tex: ^imgui.TextureData) ---
}

// [BETA] Selected render state data shared with callbacks.
// This is temporarily stored in GetPlatformIO().Renderer_RenderState during the ImGui_ImplDX12_RenderDrawData() call.
// (Please open an issue if you feel you need access to more data)
RenderState :: struct {
	Device:      ^d3d12.IDevice,
	CommandList: ^d3d12.IGraphicsCommandList,
}
