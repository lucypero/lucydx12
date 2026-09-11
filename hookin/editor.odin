package hookin

import "core:strings"
import "core:fmt"
import "core:math"
import "core:math/linalg"
import mv "core:mem/virtual"
// import "core:math/rand"
// import "core:math/linalg"
import sdl "vendor:sdl2"
import "audio"
import im "../libs/odin-imgui"
import "core:c"

// Importing rendering engine
import ldx "../src"

editor_init :: proc() {

}

Editor :: struct {
	mouse_coord: Coord
}

g_editor : Editor

editor_update :: #force_inline proc(kb: []u8) -> (_should_quit: bool){

	ldx.window_clear(COLOR_BACKGROUND)

	if kb[sdl.Scancode.TAB] == 1 && !g_was_tab_pressed{
		// switching to play mode
		game_restart()
		fmt.printfln("switching to play mode")
		g_play_mode = .Play
		return false
	}

	if kb[sdl.Scancode.ESCAPE] == 1 do return true

	g_editor.mouse_coord = world_to_coord(g_start_map, g_mouse_world_pos)

	// Left click
	if g_mouse_buttons & 0x01 != 0 && !g_mouse_clicked{
		fmt.println("clicked")
		entity_new(&g_start_map, .Wall, g_editor.mouse_coord)
	}

	// Drawing
	{
		map_draw(g_start_map)

		p_coord_pos := map_coord_to_world_pos(g_start_map, g_editor.mouse_coord)
		ldx.draw_wirebox(p_coord_pos, g_start_map.cell_tex_size, {0,1,0, 0.8}, 5)
	}

	// Imgui stuff
	{
		im.Begin("Level editor")
		defer im.End()

		ldx.imgui_do_text("map pos: %v", g_start_map.pos)

		ldx.imgui_do_text("mouse pos: %v, coord: %v",
			g_mouse_world_pos,
			g_editor.mouse_coord)

		@static tex_offset : int = 0

		gpu_ptr := ldx.get_descriptor_heap_gpu_address(ldx.g_dx_core.heap_cbv_srv_uav, tex_offset)
		tex_id : im.TextureID = gpu_ptr.ptr
		if im.ImageButton("asd", tex_id, {100, 100}) {
			tex_offset += 1
		}
	}

	return false
}
