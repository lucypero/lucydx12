package hookin

import "core:container/intrusive/list"
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
import "core:os"

// Importing rendering engine
import ldx "../src"

lprint :: ldx.lprintfln

LEVELS_DIR :: "hookin\\levels"

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

	// Loading levels
	load_levels()

	ls := g_editor.level_files[g_editor.level_selected]
	map_save(&g_start_map, ls, .Load)
}

load_levels :: proc() {
	// deleting previous scan
	for lf in g_editor.level_files do delete(lf)
	delete(g_editor.level_files)

	g_editor.level_files = make([dynamic]string, 0, 20, context.allocator)
	ldx.search_for_files_with_ext(LEVELS_DIR, ".json", &g_editor.level_files, context.allocator)

	if len(g_editor.level_files) == 0 {
		// create empty level
		lprint("no levels found. creating an empty one.")
		map_save(&g_start_map, LEVELS_DIR + "\\new map.json", .Save)
		load_levels()
	}

	g_editor.level_selected = clamp(g_editor.level_selected, 0, len(g_editor.level_files) - 1)
}

Editor :: struct {
	mouse_coord: Coord,
	entity_brushes: [7]EntityBrush,
	brush_selected: int,

	coord_selected: Coord,
	entity_in_coord_selected: int,

	// level_files
	level_files: [dynamic]string,
	level_selected: int,

	rename_field: [256]u8
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

			// Check: Only one solid per coord

			good_to_insert := true

			has_solid := does_coord_have_solid(g_start_map, g_editor.mouse_coord)
			if entity_is_solid(bs.et) && has_solid {
				lprint("This coordinate already has a solid entity.")
				good_to_insert = false
			}

			// Uniqueness check ( delete previous ones)
			if bs.et == .PlayerSpawn do entity_delete_kind(&g_start_map, .PlayerSpawn)
			if bs.et == .Goal do entity_delete_kind(&g_start_map, .Goal)

			if good_to_insert {
				entity_new(&g_start_map, bs.et, g_editor.mouse_coord)
			}

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

		ldx.imgui_do_text("Pointing at coord: %v", g_editor.mouse_coord)

		// Errors with level here.
		im.Separator()

		// list of things in the selected coord
		im.Separator()

		ldx.imgui_do_text("Entities At: %v", g_editor.coord_selected)
		entities_str := make([dynamic]string, context.temp_allocator)
		entities := map_tquery(&g_start_map, g_editor.coord_selected)

		for e in entities {
			append(&entities_str, fmt.tprint(e.et))
		}

		if len(entities_str) > 1 {
			if ldx.imgui_do_listbox("Entities at coord:", &g_editor.entity_in_coord_selected, entities_str[:]) {
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

		ldx.imgui_do_text("Current Brush Selected: %v", brush_to_string(brush_selected^))


		im.Separator()

		for eb, i in g_editor.entity_brushes {
			im.PushID(fmt.ctprintf("%v", i))
			defer im.PopID()

			gpu_ptr := ldx.get_descriptor_heap_gpu_address(ldx.g_dx_core.heap_cbv_srv_uav, eb.text_id)

			if im.ImageButton("asd", gpu_ptr.ptr, {30, 30}) {
				g_editor.brush_selected = i
				fmt.printfln("Selected %v", g_editor.entity_brushes[i].et)
			}
			im.SetItemTooltip(fmt.ctprintf("%v", brush_to_string(eb)))

			// Rows of 4
			if (i % 4 != 3) && i != len(g_editor.entity_brushes) - 1 {
				im.SameLine()
			}
		}

		// Level lister
		im.Separator()

		// list
		if len(g_editor.level_files) > 0 && ldx.imgui_do_listbox("Level List", &g_editor.level_selected, g_editor.level_files[:]) {
			// load clicked level
			ls := g_editor.level_files[g_editor.level_selected]
			map_save(&g_start_map, ls, .Load)
		}

		// Row of buttons "save, load, rename, delete"

		if im.Button("Save") {
			ls := g_editor.level_files[g_editor.level_selected]
			map_save(&g_start_map, ls, .Save)
		}

		im.SameLine()
		if im.Button("New level") {
			new_file_name_buffer : [256]u8
			new_filename_i := len(g_editor.level_files)
			sb := strings.builder_from_bytes(new_file_name_buffer[:])

			for {
				strings.builder_reset(&sb)
				fmt.sbprintf(&sb, "%v\\new_level%v.json", LEVELS_DIR, new_filename_i)
				if !os.exists(strings.to_string(sb)) do break
				new_filename_i += 1
			}

			map_save(&g_start_map, strings.to_string(sb), .Save)
			load_levels()
		}

		im.SameLine()

		// TODO use stacked popups for rename and delete
		// https://codebrowser.dev/imgui/imgui/imgui_demo.cpp.html#5487
		if im.Button("Rename") {
			// set the default rename string
			b := strings.builder_from_bytes(g_editor.rename_field[:])
			old_name := g_editor.level_files[g_editor.level_selected]
			strings.builder_reset(&b)
			strings.write_string(&b, old_name)
			append(&b.buf, 0) // making it a cstring
			im.OpenPopup("Rename Popup")
		}
		im.SameLine()
		if im.Button("Delete##") {
			ls := g_editor.level_files[g_editor.level_selected]
			err := os.remove(ls)
			if err != os.General_Error.None {
				lprint("error deleting level from disk")
			}
			load_levels()
		}

		if im.BeginPopupModal("Rename Popup") {

			ldx.imgui_do_text("Rename Level to:")

			im.InputText("new name", cstring(raw_data(g_editor.rename_field[:])), cast(c.size_t)len(g_editor.rename_field))

			if im.Button("Rename##Rename") {
				new_name := string(cstring(raw_data(g_editor.rename_field[:])))
				old_name := g_editor.level_files[g_editor.level_selected]
				rename_err := os.rename(old_name, new_name)
				if rename_err != os.General_Error.None {
					lprint("error renaming level")
				}
				im.CloseCurrentPopup()
				load_levels()
			}
			im.SameLine()

			if im.Button("Cancel##Rename") {
				im.CloseCurrentPopup()
			}

			im.EndPopup()
		}

		im.ShowDemoWindow()
	}

	return false
}

brush_to_string :: proc(b: EntityBrush) -> string {
	#partial switch b.tool_type {
	case .Entity:
		return fmt.tprint(b.et)
	case:
		return fmt.tprint(b.tool_type)
	}
}
