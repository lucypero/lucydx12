package hookin

import "core:mem"
import ldx "../src"

ENTITY_COUNT_MAX :: 200
Entities :: [dynamic;ENTITY_COUNT_MAX]Entity
EID :: int

// Map coordinate. origin at TOP LEFT of the map. Y down. X right
Coord :: v2i

Map :: struct {
	pos, scale: v2,
	size: v2i,
	cell_tex_size: v2,
	entities: Entities,
}

EntityType :: enum { Nothing, Player, Crate, PlayerSpawn, Goal, Wall, Pit }

Entity :: struct {
	et: EntityType,
	coord: Coord,
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
		if e.coord == c {
			append(&out, &e)
		}
	}

	return out[:]
}

entity_is_solid :: proc(e: Entity) -> bool{
	#partial switch e.et {
	case .Crate, .Wall:
		return true
	case: 
		return false
	}
}

does_coord_have_solid :: proc(tm: Map, c: Coord) -> bool {
	for e in tm.entities {
		if e.coord == c && entity_is_solid(e) do return true
	}

	return false
}

entity_new :: proc(tm: ^Map, et: EntityType, c: Coord) -> ^Entity {
	append(&tm.entities, Entity{et, c})
	return &tm.entities[len(tm.entities) - 1]
}

entity_get :: proc(tm: ^Map, eid: EID) -> ^Entity {
	return &tm.entities[eid]
}

map_coord_to_world_pos :: proc(the_map: Map, coord: Coord) -> v2 {

	coord_f := v2i_to_v2(coord)
	coord_f.y *= -1

	return the_map.pos + the_map.cell_tex_size * the_map.scale * coord_f
}

world_to_coord :: proc(tm: Map, pos: v2) -> Coord {
	pos := pos
	pos.y *= -1

	map_offset := v2{tm.pos.x, -tm.pos.y}
	// where does this point fall in the grid?
	cell_size : v2 = tm.cell_tex_size * tm.scale

	return v2_to_v2i((pos - map_offset) / cell_size)
}

map_world_box_to_coord :: proc(tm : Map, b: Box) -> Coord {

	// determing middle point of box
	middle_point := b.pos + {b.size.x / 2, -b.size.y / 2}
	// Flipping y (world space is +y up, coord space is +y down)
	middle_point.y *= -1
	map_offset := v2{tm.pos.x, -tm.pos.y}

	// where does this point fall in the grid?
	cell_size : v2 = tm.cell_tex_size * tm.scale

	return v2_to_v2i((middle_point - map_offset) / cell_size)
}

map_start_default :: proc(tm: ^Map) {

	map_size := v2i{10, 6}

	// initting map
	cell_tex_size := v2i_to_v2(ldx.texture_get_size(g_textures.wall))

	tm^ = {
		v2{50, WINDOW_HEIGHT - 10},
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
