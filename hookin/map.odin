package hookin

import "core:mem"
import l2d "../lucy2d"
import "core:encoding/json"
import "core:os"

ENTITY_COUNT_MAX :: 2000
Entities :: [dynamic;ENTITY_COUNT_MAX]Entity
EID :: struct {slot: int, gen: i32}

// Map coordinate. origin at TOP LEFT of the map. Y down. X right
Coord :: v2i

Map :: struct {
	pos, scale: v2,
	size: v2i,
	cell_tex_size: v2,
	entities: Entities,
}

Visualization :: struct {
	offset: v2
}

map_get_player_spawn_coord :: proc(tm: Map) -> Coord {
	for e in tm.entities {
		if e.et == .PlayerSpawn {
			return e.coord
		}
	}
	panic("no player spawn in map!!!")
}

map_get_tile_pos_size :: proc(the_map: Map, coord: Coord) -> (v2, v2) {
	return {
		the_map.pos.x + cast(f32)coord.x * the_map.cell_tex_size.x * the_map.scale.x,
		the_map.pos.y - cast(f32)coord.y * the_map.cell_tex_size.y * the_map.scale.y
	}, the_map.cell_tex_size * the_map.scale
}

// Index into tilemap, to Coord
map_get_coord :: proc(tm: Map, i: int) -> Coord {
	return Coord{i % tm.size.x, i / tm.size.x}
}

// Queries the coord for entities. stores results on temp allocator
map_tquery :: proc(tm: ^Map, c: Coord) -> []^Entity {
	out := make([dynamic]^Entity, 0, 3, context.temp_allocator)

	for &e in tm.entities {
		if e.coord == c && e.et != .Nothing {
			append(&out, &e)
		}
	}

	return out[:]
}

does_coord_have_solid :: proc(tm: Map, c: Coord) -> bool {
	for e in tm.entities {
		if e.coord == c && entity_is_solid(e.et) do return true
	}

	return false
}

entity_new :: proc(tm: ^Map, et: EntityType, c: Coord) -> ^Entity {

	for &e, i in tm.entities {
		if e.et == .Nothing {
			e = Entity{et, e.gen, c, {}}
			return &e
		}
	}

	append(&tm.entities, Entity{et, 0, c, {}})
	return &tm.entities[len(tm.entities) - 1]
}

entity_delete :: proc{entity_delete_eid, entity_delete_ptr, entity_delete_kind}

entity_delete_kind :: proc(tm: ^Map, et: EntityType) {
	for &e in tm.entities {
		if e.et == .Nothing do continue
		if e.et == et do entity_delete_ptr(tm, &e)
	}
}

entity_delete_eid :: proc(tm: ^Map, eid: EID) {
	e := entity_get(tm, eid)
	entity_delete_ptr(tm, e)
}

entity_delete_ptr :: proc(tm: ^Map, e: ^Entity) {
	if e != nil {
		e.et = .Nothing
		e.gen += 1
	} else {
		lprint("tried to delete entity that isn't valid")
	}
}

entity_get :: proc(tm: ^Map, eid: EID) -> ^Entity {
	e := &tm.entities[eid.slot]
	return (e.et != .Nothing && e.gen == eid.gen) ? e : nil
}

// points to top left of coord
map_coord_to_world_pos :: proc(the_map: Map, coord: Coord) -> v2 {

	coord_f := l2d.v2i_to_v2(coord)
	coord_f.y *= -1

	return the_map.pos + the_map.cell_tex_size * the_map.scale * coord_f
}

// points to center of coord
map_coord_to_world_pos_center :: proc(the_map: Map, coord: Coord) -> v2 {

	coord_f := l2d.v2i_to_v2(coord)
	coord_f.y *= -1

	center_offset: v2 = the_map.cell_tex_size / 2
	center_offset.y *= -1

	return the_map.pos + the_map.cell_tex_size * the_map.scale * coord_f + center_offset
}

world_to_coord :: proc(tm: Map, pos: v2) -> Coord {
	pos := pos
	pos.y *= -1

	map_offset := v2{tm.pos.x, -tm.pos.y}
	// where does this point fall in the grid?
	cell_size : v2 = tm.cell_tex_size * tm.scale

	return l2d.v2_to_v2i((pos - map_offset) / cell_size)
}

map_world_box_to_coord :: proc(tm : Map, b: Box) -> Coord {

	// determing middle point of box
	middle_point := b.pos + {b.size.x / 2, -b.size.y / 2}
	// Flipping y (world space is +y up, coord space is +y down)
	middle_point.y *= -1
	map_offset := v2{tm.pos.x, -tm.pos.y}

	// where does this point fall in the grid?
	cell_size : v2 = tm.cell_tex_size * tm.scale

	return l2d.v2_to_v2i((middle_point - map_offset) / cell_size)
}

map_start_default :: proc(tm: ^Map) {

	map_size := v2i{10, 6}

	// initting map
	cell_tex_size := l2d.v2i_to_v2(l2d.texture_get_size(g_entity_defs[EntityType.Wall].tex_id))

	window_res := l2d.get_window_res()

	tm^ = {
		v2{50, cast(f32)window_res.y - 10},
		v2{1,1},
		map_size,
		cell_tex_size,
	{},
	}

	// populating map entities
	entity_new(tm, .PlayerSpawn, {3,3})

	for y in 0..<map_size.y {
		for x in 0..<map_size.x {

			top_bottom_row := x == 0 || x == map_size.x - 1
			left_right_col := y == 0 || y == map_size.y - 1

			if top_bottom_row || left_right_col {
				entity_new(tm, .Wall, {x,y})
			}
		}
	}

	entity_new(tm, .Pit, {6,3})
	entity_new(tm, .Goal, {7,3})
	entity_new(tm, .Crate, {5,2})
}

// Copies state from one map to another. 
map_copy :: proc(md: ^Map, ms: Map) {
	md^ = ms
}

map_save :: proc(m: ^Map, file_name:string, save: enum{Save, Load}) {
	JSON_SPEC :: json.Specification.Bitsquid
	switch save {
	case .Save:
		json_data, err := json.marshal(m^, {
			pretty         = true,
			use_enum_names = true,
			spec = JSON_SPEC
		})
		assert(err == nil)
		werr := os.write_entire_file(file_name, json_data)
		if werr == .Exist {
			lprint("file already exists. overwriting.")
		} else if werr != os.General_Error.None {
			lprint("error while writing file.")
		}
	case .Load:
		data, read_err := os.read_entire_file(file_name, context.temp_allocator)
		if read_err == os.General_Error.Not_Exist {
			lprint("level %v does not exist.", file_name)
			return
		}
		assert(read_err == nil)
		unmarshal_err := json.unmarshal(data, m, JSON_SPEC, context.temp_allocator)
		assert(unmarshal_err == nil)
	}
}

map_draw :: proc(tm: Map, edit_mode: bool) {

	// Draw ground entities first
	for e, i in tm.entities {
		if e.et == .Nothing || !entity_is_floor(e.et) do continue
		if entity_is_invisible(e.et) && !edit_mode do continue

		pos, size := map_get_tile_pos_size(tm, e.coord)
		pos += e.vis.offset

		#partial switch e.et {
		case .Ground:
			l2d.draw_texture(g_entity_defs[EntityType.Ground].tex_id, pos, tm.scale)
		case .Pit: 
			l2d.draw_solid_rect(pos, size, COLOR_BLACK)
			l2d.draw_texture(g_entity_defs[EntityType.Pit].tex_id, pos, tm.scale)
		}
	}

	// Draw non-floor entities
	for e, i in tm.entities {
		if e.et == .Nothing || entity_is_floor(e.et) do continue
		if entity_is_invisible(e.et) && !edit_mode do continue

		pos, size := map_get_tile_pos_size(tm, e.coord)
		pos += e.vis.offset

		if g_entity_defs[e.et].tex_id != 0 {
			l2d.draw_texture(g_entity_defs[e.et].tex_id, pos, tm.scale)
		}
	}
}
