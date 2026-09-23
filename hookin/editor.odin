package hookin

import "core:strings"
import "core:fmt"
import sdl "vendor:sdl2"
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

ToolType :: enum {SelectTool, DeleteTool, PaintTool, RectPaintTool}

ToolButton :: struct {
	text_id: int,
}

EntityButton :: struct {
	text_id: int,
	et: EntityType,
}

ToolFilter :: enum i32 { All, Floor, NonFloor}

Editor :: struct {
	mouse_coord: Coord,
	tool_buttons: [ToolType]ToolButton,
	entity_buttons: [dynamic]EntityButton,

	tool_button_selected: ToolType,
	entity_button_selected: int,

	coord_selected: Coord,
	entity_in_coord_selected: int,

	// level_files
	level_files: [dynamic]string,
	level_selected: int,

	rename_field: [256]u8,

	filter_selected: ToolFilter,

	last_coord_clicked: Maybe(Coord),

	mid_rectpaint: Maybe(Coord)
}

editor_init :: proc() {

	g_editor.tool_buttons =  {
		.SelectTool = {g_textures.move_hand},
		.DeleteTool = {g_textures.trash},
		.PaintTool = {g_textures.paint},
		.RectPaintTool = {g_textures.rect_tool}
	}

	// Registering brushes
	clear(&g_editor.entity_buttons)
	for e in g_entity_defs {
		if !entity_is_editor_brush(e.type) do continue
		append(&g_editor.entity_buttons, EntityButton{e.tex_id, e.type})
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

editor_update :: #force_inline proc() -> (_should_quit: bool){

	l2d.window_clear(COLOR_BACKGROUND)

	// Input

	if l2d.key_is_just_pressed(.TAB) {
		// switching to play mode
		game_restart()
		g_play_mode = .Play
		return false
	}

	if l2d.key_is_just_pressed(.ESCAPE) do return true

	// Camera controls
	{
		cam_vel : v2
		CAM_SPEED :: 4

		if l2d.key_is_down(.W) {
			cam_vel.y += CAM_SPEED
		}

		if l2d.key_is_down(.A) {
			cam_vel.x -= CAM_SPEED
		}

		if l2d.key_is_down(.S) {
			cam_vel.y -= CAM_SPEED
		}

		if l2d.key_is_down(.D) {
			cam_vel.x += CAM_SPEED
		}

		g_cam.pos += cam_vel
	}

	p_coord_pos := map_coord_to_world_pos(g_start_map, g_editor.mouse_coord)

	mouse_pos_world :v2

	// mouse pos to world pos (considering camera)
	{
		mouse_pos_world = l2d.get_mouse_pos() * g_cam.zoom + g_cam.pos
	}

	g_editor.mouse_coord = world_to_coord(g_start_map, mouse_pos_world)

	// Left click
	if l2d.mouse_button_is_down(.Left) {

		if lc, ok := g_editor.last_coord_clicked.?; !ok || (ok && lc != g_editor.mouse_coord) {
			on_mouse_click()
		}

		g_editor.last_coord_clicked = g_editor.mouse_coord

	} else {
		g_editor.last_coord_clicked = nil
	}

	if l2d.mouse_button_is_just_unpressed(.Left) {

		if rect_from, ok := g_editor.mid_rectpaint.? ; ok {
			rect_to := g_editor.mouse_coord

			rf := Coord{min(rect_from.x, rect_to.x), min(rect_from.y, rect_to.y)}
			rt := Coord{max(rect_from.x, rect_to.x), max(rect_from.y, rect_to.y)}

			// get all coords from from to mouse coord
			for x in rf.x..=rt.x {
				for y in rf.y..=rt.y {
					the_coord : Coord = Coord{x,y}
					paint_on_coord(the_coord)
				}
			}
		}

		g_editor.mid_rectpaint = nil
	}

	// Drawing Everything
	{
		map_draw(g_start_map, edit_mode =  true)

		if is_on_paint_tool()  {
			entity_button_selected := g_editor.entity_buttons[g_editor.entity_button_selected]
			l2d.draw_texture(entity_button_selected.text_id, p_coord_pos, tint = {1,1,1,0.5})
		}

		// Drawing paint rect tool

		if rect_from, ok := g_editor.mid_rectpaint.? ; ok {
			rect_to := g_editor.mouse_coord

			rf := Coord{min(rect_from.x, rect_to.x), min(rect_from.y, rect_to.y)}
			rt := Coord{max(rect_from.x, rect_to.x), max(rect_from.y, rect_to.y)}

			// get all coords from from to mouse coord
			for x in rf.x..=rt.x {
				for y in rf.y..=rt.y {
					the_coord : Coord = Coord{x,y}
					entity_button_selected := g_editor.entity_buttons[g_editor.entity_button_selected]
					l2d.draw_texture(entity_button_selected.text_id, map_coord_to_world_pos(g_start_map, the_coord), tint = {1,1,1,0.5})
				}
			}
		}

		// highlight selected coord
		l2d.draw_wirebox(map_coord_to_world_pos(g_start_map, g_editor.coord_selected), g_start_map.cell_tex_size, {0,1,0,0.5}, 5)

	}

	do_imgui_ui()


	return false
}

do_imgui_ui :: proc() {
	im.Begin("Level editor")
	defer im.End()

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

	// ldx.imgui_do_text("Current Brush Selected: %v", brush_to_string(brush_selected^))

	for tool_button, i in g_editor.tool_buttons {
		im.PushID(fmt.ctprintf("%v", i))
		defer im.PopID()

		// TODO: is it ok for app code to have to access such an implementation detail thing here?
		gpu_ptr := ldx.get_descriptor_heap_gpu_address(ldx.g_dx_core.heap_cbv_srv_uav, tool_button.text_id)
		texture_ref := im.TextureRef {_TexID = gpu_ptr.ptr}

		if g_editor.tool_button_selected == i {
			im.PushStyleColorImVec4(.Button, {1,1,1,0.5})
		} else {
			im.PushStyleColorImVec4(.Button, {0,0,0,0.5})
		}

		if im.ImageButton("asd", texture_ref, {30, 30}) {
			g_editor.tool_button_selected = i
		}

		im.PopStyleColor()

		im.SetItemTooltip(fmt.ctprintf("%v", i))

		// Rows of 4
		i_n : int = cast(int)i
		if (i_n % 4 != 3) && i_n != len(g_editor.tool_buttons) - 1 {
			im.SameLine()
		}
	}

	im.SeparatorText("Entity Type Selection")

	for eb, i in g_editor.entity_buttons {
		im.PushID(fmt.ctprintf("%v", i))
		defer im.PopID()

		// TODO: is it ok for app code to have to access such an implementation detail thing here?
		gpu_ptr := ldx.get_descriptor_heap_gpu_address(ldx.g_dx_core.heap_cbv_srv_uav, eb.text_id)
		texture_ref := im.TextureRef {_TexID = gpu_ptr.ptr}

		if g_editor.entity_button_selected == i && is_on_paint_tool() {
			im.PushStyleColorImVec4(.Button, {1,1,1,0.5})
		} else {
			im.PushStyleColorImVec4(.Button, {0,0,0,0.5})
		}

		if im.ImageButton("asd", texture_ref, {30, 30}) {
			g_editor.entity_button_selected = i
			if !is_on_paint_tool() {
				g_editor.tool_button_selected = .PaintTool
			}
		}

		im.PopStyleColor()

		im.SetItemTooltip(fmt.ctprintf("%v", eb.et))

		// Rows of 4
		if (i % 4 != 3) && i != len(g_editor.entity_buttons) - 1 {
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

on_mouse_click :: proc() {
	switch g_editor.tool_button_selected {
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
	case .PaintTool:
		paint_on_coord(g_editor.mouse_coord)
	case.RectPaintTool:

		if _, ok := g_editor.mid_rectpaint.?; !ok {
			g_editor.mid_rectpaint = g_editor.mouse_coord
		}
	}
}

paint_on_coord :: proc(coord: Coord) {
	entity_to_paint_selected := g_editor.entity_buttons[g_editor.entity_button_selected].et

	// Floor Type: replace floor at cord (only one floor entity per coord)
	if entity_is_floor(entity_to_paint_selected) {
		ets_at_coord := map_tquery(&g_start_map, coord)

		for &e in ets_at_coord {
			if entity_is_floor(e.et) {
				// delete
				entity_delete(&g_start_map, e)
			}
		}

		entity_new(&g_start_map, entity_to_paint_selected, coord)
	} else {

		// Check: Only one solid per coord

		good_to_insert := true

		has_solid := does_coord_have_solid(g_start_map, coord)
		if entity_is_solid(entity_to_paint_selected) && has_solid {
			lprint("This coordinate already has a solid entity.")
			good_to_insert = false
		}

		// Uniqueness check ( delete previous ones)
		if entity_to_paint_selected == .PlayerSpawn do entity_delete_kind(&g_start_map, .PlayerSpawn)
		if entity_to_paint_selected == .Goal do entity_delete_kind(&g_start_map, .Goal)

		if good_to_insert {
			entity_new(&g_start_map, entity_to_paint_selected, coord)
		}
	}
}

is_on_paint_tool :: proc() -> bool {
	#partial switch g_editor.tool_button_selected {
	case .PaintTool, .RectPaintTool: return true
	case: return false
	}
}
