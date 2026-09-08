package hookin

import mv "core:mem/virtual"
import "core:mem"

ENTITY_COUNT_MAX :: 20
Entities :: [dynamic;ENTITY_COUNT_MAX]Entity

Map :: struct {
	arena: mem.Allocator,
	pos, scale: v2,
	size: v2i,
	cell_tex_size: v2,
	tilemap: []Tile, // Tiles are static elements in the map
	entities: Entities,
}

TileType :: enum { Wall, Pit, Ground }

Tile :: struct {
	tt: TileType,// tile typ
	// extra data? idk
}

EntityType :: enum { Player, Crate, PlayerSpawn, Goal }

Entity :: struct {
	et: EntityType, // entity type
	id: int,
	coord: Coord,
}


tile_is_solid :: proc(the_map: Map, coord: Coord) -> bool {
	tile, ok := map_get_tile(the_map, coord)
	if !ok do return true

	switch tile.tt {
	case .Wall:
		return true
	case .Pit, .Ground:
		fallthrough
	case:
		return false
	}
}

map_get_tile_count :: proc(the_map: Map) -> int {
	return len(the_map.entities)
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

@(require_results)
map_get_tile :: proc(tm: Map, coord: Coord) -> (tile: Tile, ok: bool = false) {
	if !(coord.x >= 0 && coord.x < tm.size.x) do return
	if !(coord.y >= 0 && coord.y < tm.size.y) do return
	return tm.tilemap[tm.size.x * coord.y + coord.x], true
}

map_get_tile_ref :: proc(tm: ^Map, coord: Coord) -> (tile: ^Tile) {
	return &tm.tilemap[tm.size.x * coord.y + coord.x]
}

@(require_results)
map_get_tile_unchecked :: proc(tm: Map, coord: Coord) -> (tile: Tile) {
	return tm.tilemap[tm.size.x * coord.y + coord.x]
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
