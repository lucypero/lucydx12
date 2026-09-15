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

lprint :: ldx.lprintfln

EntityBrush :: struct {
	text_id: int,
	et: EntityType,
	tool_type: enum {Entity, SelectTool, DeleteTool}
}

editor_init :: proc() {
	g_editor.entity_brushes = {
		{g_textures.move_hand, .Nothing, .SelectTool},
		{g_textures.trash, .Nothing, .DeleteTool},
		{g_textures.spawn, .PlayerSpawn, .Entity},
		{g_textures.wall, .Wall, .Entity},
		{g_textures.pit, .Pit, .Entity},
		{g_textures.crate_wood, .Crate, .Entity},
		{g_textures.goal, .Goal, .Entity},
	}
}

Editor :: struct {
	mouse_coord: Coord,
	entity_brushes: [7]EntityBrush,
	brush_selected: int,

	coord_selected: Coord,
	entity_in_coord_selected: int,
}

g_editor : Editor

editor_update :: #force_inline proc() -> (_should_quit: bool){

	bs := &g_editor.entity_brushes[g_editor.brush_selected]

	ldx.window_clear(COLOR_BACKGROUND)

	if ldx.key_is_just_pressed(.TAB) {
		// switching to play mode
		game_restart()
		fmt.printfln("switching to play mode")
		g_play_mode = .Play
		return false
	}

	if ldx.key_is_just_pressed(.ESCAPE) do return true

	p_coord_pos := map_coord_to_world_pos(g_start_map, g_editor.mouse_coord)

	g_editor.mouse_coord = world_to_coord(g_start_map, ldx.get_mouse_pos())

	// Left click
	if ldx.mouse_button_is_just_pressed(.Left) {
		fmt.println("clicked")

		switch bs.tool_type {
		case .Entity:
			entity_new(&g_start_map, bs.et, g_editor.mouse_coord)
		case .SelectTool:
			// clicked on a coord with the select tool. select the coord.
			// selected coord change.
			g_editor.coord_selected = g_editor.mouse_coord
			g_editor.entity_in_coord_selected = 0
			lprint("clicked on coord %v. selecting.", g_editor.mouse_coord)
		case .DeleteTool:

			ets := map_tquery(&g_start_map, g_editor.mouse_coord)
			for e in ets {
				entity_delete(&g_start_map, e)
			}
		}
	}

	// Drawing
	{
		map_draw(g_start_map, edit_mode =  true)
		if bs.tool_type == .Entity {
			ldx.draw_texture(bs.text_id, p_coord_pos, tint = {1,1,1,0.5})
		}

		// highlight selected coord
		ldx.draw_wirebox(map_coord_to_world_pos(g_start_map, g_editor.coord_selected), g_start_map.cell_tex_size, {0,1,0,0.5}, 5)

	}

	// Imgui stuff
	{
		im.Begin("Level editor")
		defer im.End()

		brush_selected := &g_editor.entity_brushes[g_editor.brush_selected]

		ldx.imgui_do_text("mouse pos: %v, coord: %v",
			ldx.get_mouse_pos(),
			g_editor.mouse_coord)

		// Errors with level here.
		im.Separator()

		// list of things in the selected coord
		im.Separator()

		ldx.imgui_do_text("Entities At: %v", g_editor.coord_selected)
		entities_cstrings := make([dynamic]cstring, context.temp_allocator)
		entities := map_tquery(&g_start_map, g_editor.coord_selected)
		for e in entities {
			append(&entities_cstrings, fmt.ctprintf("%v", e.et))
		}

		if len(entities_cstrings) > 1 {
			if im.ListBox("Entities At coord:",
				cast(^c.int)&g_editor.entity_in_coord_selected,
				&entities_cstrings[0],
				cast(i32)len(entities_cstrings)
			) {
				fmt.printfln("clicked somewhere on table")
			}
		}

		// Info about selected entity
		if len(entities) > 0 {
			ent_s := entities[g_editor.entity_in_coord_selected]
			ldx.imgui_do_text("Selected Entity: %v", ent_s.et)
			if im.Button("Delete") {
				// Delete this entity
				entity_delete(&g_start_map, ent_s)
				g_editor.entity_in_coord_selected = 0
			}
		}

		// Info about selected brush
		im.Separator()
		ldx.imgui_do_text("Current Brush Selected: %v", brush_selected.et)


		im.Separator()

		for eb, i in g_editor.entity_brushes {
			im.PushID(fmt.ctprintf("%v", i))
			defer im.PopID()

			gpu_ptr := ldx.get_descriptor_heap_gpu_address(ldx.g_dx_core.heap_cbv_srv_uav, eb.text_id)

			if im.ImageButton("asd", gpu_ptr.ptr, {30, 30}) {
				g_editor.brush_selected = i
				fmt.printfln("Selected %v", g_editor.entity_brushes[i].et)
			}
			im.SetItemTooltip(fmt.ctprintf("%v", eb.et))

			// Rows of 4
			if (i % 4 != 3) && i != len(g_editor.entity_brushes) - 1 {
				im.SameLine()
			}
		}

		im.ShowDemoWindow()
	}

	return false
}
