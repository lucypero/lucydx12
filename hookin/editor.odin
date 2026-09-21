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
import l2d "../lucy2d"
import ldx "../lucydx"

lprint :: ldx.lprintfln
LEVELS_DIR :: "hookin\\levels"


// Global state
g_editor : Editor
g_cam : ^l2d.Camera

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

	g_cam = l2d.get_camera()

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

ToolFilter :: enum i32 { All, Floor, NonFloor}

Editor :: struct {
	mouse_coord: Coord,
	entity_brushes: [7]EntityBrush,
	brush_selected: int,

	coord_selected: Coord,
	entity_in_coord_selected: int,

	// level_files
	level_files: [dynamic]string,
	level_selected: int,

	rename_field: [256]u8,

	filter_selected: ToolFilter
}


editor_update :: #force_inline proc() -> (_should_quit: bool){

	bs := &g_editor.entity_brushes[g_editor.brush_selected]

	l2d.window_clear(COLOR_BACKGROUND)

	if l2d.key_is_just_pressed(.TAB) {
		// switching to play mode
		game_restart()
		g_play_mode = .Play
		return false
	}

	if l2d.key_is_just_pressed(.ESCAPE) do return true

	p_coord_pos := map_coord_to_world_pos(g_start_map, g_editor.mouse_coord)

	mouse_pos_world :v2

	// mouse pos to world pos (considering camera)
	{
		mouse_pos_world = l2d.get_mouse_pos() * g_cam.zoom + g_cam.pos
	}

	g_editor.mouse_coord = world_to_coord(g_start_map, mouse_pos_world)

	// Left click
	if l2d.mouse_button_is_just_pressed(.Left) {
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
		case .DeleteTool:

			ets := map_tquery(&g_start_map, g_editor.mouse_coord)
			for e in ets {
				entity_delete(&g_start_map, e)
			}
		}
	}

	// Drawing Map
	{
		map_draw(g_start_map, edit_mode =  true)
		if bs.tool_type == .Entity {
			l2d.draw_texture(bs.text_id, p_coord_pos, tint = {1,1,1,0.5})
		}

		// highlight selected coord
		l2d.draw_wirebox(map_coord_to_world_pos(g_start_map, g_editor.coord_selected), g_start_map.cell_tex_size, {0,1,0,0.5}, 5)

	}

	do_imgui_ui()


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

do_imgui_ui :: proc() {
	im.Begin("Level editor")
	defer im.End()

	brush_selected := &g_editor.entity_brushes[g_editor.brush_selected]

	// ldx.imgui_do_text("Pointing at coord: %v", g_editor.mouse_coord)

	ent_title := fmt.ctprintf("Entities at Coord: %v",  g_editor.coord_selected)
	im.SeparatorText(ent_title)

	entities_str := make([dynamic]string, context.temp_allocator)
	entities := map_tquery(&g_start_map, g_editor.coord_selected)

	for e in entities {
		append(&entities_str, fmt.tprint(e.et))
	}

	if len(entities_str) > 1 {
		if ldx.imgui_do_listbox("Entities at coord:", &g_editor.entity_in_coord_selected, entities_str[:]) {
			// fmt.printfln("clicked somewhere on table")
		}
	}

	// Info about selected entity
	if len(entities) > 0 {
		ent_s := entities[g_editor.entity_in_coord_selected]
		ldx.imgui_do_text("Selected Entity: %v", ent_s.et)
		if im.Button("Delete Entity") {
			// Delete this entity
			entity_delete(&g_start_map, ent_s)
			g_editor.entity_in_coord_selected = 0
		}
	} else {
		ldx.imgui_do_text("No Entities at selected coord.")
	}

	im.SeparatorText("Tool Selection:")

	ldx.imgui_do_text("Current Brush Selected: %v", brush_to_string(brush_selected^))

	for eb, i in g_editor.entity_brushes {
		im.PushID(fmt.ctprintf("%v", i))
		defer im.PopID()

		// TODO: is it ok for app code to have to access such an implementation detail thing here?
		gpu_ptr := ldx.get_descriptor_heap_gpu_address(ldx.g_dx_core.heap_cbv_srv_uav, eb.text_id)
		texture_ref := im.TextureRef {_TexID = gpu_ptr.ptr}
		if im.ImageButton("asd", texture_ref, {30, 30}) {
			g_editor.brush_selected = i
		}
		im.SetItemTooltip(fmt.ctprintf("%v", brush_to_string(eb)))

		// Rows of 4
		if (i % 4 != 3) && i != len(g_editor.entity_brushes) - 1 {
			im.SameLine()
		}
	}

	// Filter selection
	im.SeparatorText("Tool Filtering:")

	filter_addr := transmute(^i32)&g_editor.filter_selected

	im.RadioButtonIntPtr("Floor", filter_addr, 0)
	im.SameLine()
	im.RadioButtonIntPtr("Non-Floor", filter_addr, 1)
	im.SameLine()
	im.RadioButtonIntPtr("All", filter_addr, 2)

	// Row of buttons "save, load, rename, delete"

	im.SeparatorText("Level Management:")

	// list
	if len(g_editor.level_files) > 0 && ldx.imgui_do_listbox("Level List", &g_editor.level_selected, g_editor.level_files[:]) {
		// load clicked level
		ls := g_editor.level_files[g_editor.level_selected]
		map_save(&g_start_map, ls, .Load)
	}

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


	im.SeparatorText("Camera:")

	im.DragFloat2("camera position", &g_cam.pos)
	im.DragFloat("camera zoom", &g_cam.zoom, 0.01)

	im.ShowDemoWindow()

	// Popup Modals

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
}
