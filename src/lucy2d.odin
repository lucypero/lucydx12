package main

import "core:thread"
import dx "vendor:directx/d3d12"
import "core:debug/trace"
import sdl "vendor:sdl2"
import "core:strings"
import "core:fmt"
import "core:mem/virtual"

Color :: v4

Lucy2DContext :: struct {

	upload_thread : ^thread.Thread,
	resources_resizing : [dynamic]^dx.IUnknown,
	resources_longterm : [dynamic]^dx.IUnknown,
	trace_ctx : trace.Context,
	window : ^sdl.Window
}

g_lct : Lucy2DContext

// creates the window
// the app HAS to call this before anything else.
window_new :: proc(window_name:string, width, height: int) {


	// set up allocators?
	g_lct.upload_thread = thread.create_and_start(lucy2d_upload_thread_start)

	// setting up resource pool for buffers tied to window size
	g_lct.resources_resizing = make([dynamic]^dx.IUnknown)
	g_lct.resources_longterm = make([dynamic]^dx.IUnknown)

	trace.init(&g_lct.trace_ctx)

	ct := &g_lct

	// Init SDL and create window
	if err := sdl.Init(sdl.InitFlags{.TIMER, .AUDIO, .VIDEO, .EVENTS}); err != 0 {
		fmt.eprintln(err)
		return
	}

	window_name_cstr := strings.clone_to_cstring(window_name, context.temp_allocator)
	ct.window = sdl.CreateWindow(
		window_name_cstr,
		sdl.WINDOWPOS_UNDEFINED,
		sdl.WINDOWPOS_UNDEFINED,
		cast(i32)width,
		cast(i32)height,
		{.ALLOW_HIGHDPI, .SHOWN, .RESIZABLE},
	)

	if ct.window == nil {
		fmt.eprintln(sdl.GetError())
		return
	}
}

window_cleanup :: proc() {
	ct := &g_lct
	delete(g_lct.resources_resizing)
	delete(g_lct.resources_longterm)
	trace.destroy(&g_lct.trace_ctx)
	thread.destroy(g_lct.upload_thread)
	sdl.DestroyWindow(ct.window)
	sdl.Quit()
}

// clears the window with a color
window_clear :: proc(color: Color) {

}

draw_sprite :: proc(pos, size: v2) {

}

// draws all the stuff, and resets frame state
present :: proc() {
	sdl.PumpEvents()
}

get_keyboard :: proc() -> []u8 {
	return sdl.GetKeyboardStateAsSlice()
}

lucy2d_upload_thread_start :: proc() {

	// TODO: set the allocator to the tracking allocator

	// context.allocator = mem.tracking_allocator(&g_track)

	// make temp allocator for upload thread
	upload_temp_arena := arena_new()
	upload_temp_allocator := virtual.arena_allocator(&upload_temp_arena)
	context.temp_allocator = upload_temp_allocator

	// ends
	// it does nothing


	// ending...
	arena_destroy(&upload_temp_arena)
}
