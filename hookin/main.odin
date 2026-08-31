package hookin

// import "core:math/rand"
// import "core:math/linalg"
import sdl "vendor:sdl2"
import "audio"

// Importing rendering engine
import ldx "../src"

v2i :: ldx.v2i
v2 :: ldx.v2
v4 :: ldx.v4

Coord :: v2i

WINDOW_WIDTH :: 1000
WINDOW_HEIGHT :: 500
COLOR_BACKGROUND :: v4{0.773, 0.686, 0.643,1}
COLOR_CHARACTER :: v4{0.8, 0.494, 0.522, 1}
COLOR_FOOD :: v4{0.639, 0.427, 0.565, 1}
COLOR_BLACK :: v4{0,0,0,1}
CHARACTER_SPEED :: 1
CHARACTER_SIZE :: 64

Tile :: enum {
	Wall,
	Ground,
	Pit,
	CrateWood,
	CrateStone,
	Goal,
	PlayerSpawn
}

Map :: struct {
	pos, scale: v2,
	cell_tex_size: v2,
	player_spawn_coord: Coord,
	tiles: [][]Tile
}

Textures :: struct {
	player, crate_wood, ground, wall, crate_stone, goal, pit: int
}

g_textures: Textures
g_char_pos: v2
g_char_tex_size: v2

main :: proc() {
	ldx.window_new("hookin", WINDOW_WIDTH, WINDOW_HEIGHT)
	g_char_pos = {120, 300}
	char_scale : f32 = 1

	ok := audio.init()
	if !ok {
		// handle error
	}
	midi_track, success := audio.load_midi_file("hookin/audio/ct600ad.mid")
	if !success {
		// handle error
	}

	g_textures.player = ldx.texture_load("hookin_sprites/sokoban-pack/Player/player_01.png")
	g_textures.crate_wood = ldx.texture_load("hookin_sprites/sokoban-pack/Crates/crate_07.png")
	g_textures.ground = ldx.texture_load("hookin_sprites/sokoban-pack/Ground/ground_01.png")
	g_textures.wall = ldx.texture_load("hookin_sprites/sokoban-pack/Blocks/block_01.png")
	g_textures.crate_stone = ldx.texture_load("hookin_sprites/sokoban-pack/Blocks/block_02.png")
	g_textures.goal = ldx.texture_load("hookin_sprites/sokoban-pack/Environment/environment_10.png")
	g_textures.pit = ldx.texture_load("hookin_sprites/sokoban-pack/Environment/environment_06.png")

	char_tex_size_i := ldx.texture_get_size(g_textures.player)
	g_char_tex_size = v2{cast(f32)char_tex_size_i.x, cast(f32)char_tex_size_i.y}

	// constructing map
	the_map: Map
	{
		cell_tex_size := ldx.texture_get_size(g_textures.wall)
		cell_tex_size_f := v2{cast(f32)cell_tex_size.x, cast(f32)cell_tex_size.y}

		map_tiles : [][]Tile = {
			{.Wall, .Wall, .Wall, .Wall, .Wall, .Wall, .Wall},
			{.Wall, .Ground, .Ground, .Pit, .Ground, .CrateStone, .Wall},
			{.Wall, .Ground, .Ground, .Pit, .Goal, .Ground, .Wall},
			{.Wall, .PlayerSpawn, .Ground, .Pit, .Ground, .Ground, .Wall},
			{.Wall, .Wall, .Wall, .Wall, .Wall, .Wall, .Wall}
		}

		player_spawn_coord : Coord

		outer: for row_i, y in map_tiles {
			for col_i, x in row_i {
				if col_i == .PlayerSpawn {
					player_spawn_coord = {x, y}
					break outer
				}
			}
		}

		the_map = {
			v2{50, WINDOW_HEIGHT - 50 - cell_tex_size_f.y},
			v2{1,1},
			cell_tex_size_f, 
			player_spawn_coord,
			map_tiles
		}
	}

	g_char_pos = map_coord_to_world_pos(the_map, the_map.player_spawn_coord)

	audio.play_midi(&midi_track)
	frame := 0
	for !ldx.window_should_close() {
		audio.update()
		kb := ldx.get_keyboard()
		if kb[sdl.Scancode.ESCAPE] == 1 do break
		ldx.window_clear(COLOR_BACKGROUND)

		// Update game logic
		frame += 1
		if frame % 50 == 0 {
			audio.play_note(.C, 2, 0.1, 127, 9)
		}

		if kb[sdl.Scancode.A] == 1 do character_move({-CHARACTER_SPEED, 0}, the_map)
		if kb[sdl.Scancode.D] == 1 do character_move({CHARACTER_SPEED, 0}, the_map)
		if kb[sdl.Scancode.W] == 1 do character_move({0, CHARACTER_SPEED}, the_map)
		if kb[sdl.Scancode.S] == 1 do  character_move({0, -CHARACTER_SPEED}, the_map)

		// Drawing everything
		map_draw(the_map)
		ldx.draw_texture(g_textures.player, g_char_pos, {char_scale, char_scale})
		ldx.present()
	}

	ldx.window_cleanup()
}

character_move :: proc(dir: v2, the_map: Map) {

	pos_future := g_char_pos + dir

	hit_thing := false

	aabb_player := AABB{{0,g_char_tex_size.x}, {0,g_char_tex_size.y}}

	aabb_future := aabb_translate(aabb_player, pos_future)

	for row, row_i in the_map.tiles {
		for cell, column_i in row {
			#partial switch cell {
			case .Wall, .CrateWood,.CrateStone:
				pos, size := map_get_tile_pos_size(the_map, row_i, column_i)
				aabb_tile := aabb_translate(AABB{{0,size.x}, {0,size.y}}, pos)
				if aabb_do_collide(aabb_future, aabb_tile) {
					hit_thing = true
					return
				}
			case:
				continue
			}
		}
	}
	g_char_pos = pos_future
}

map_get_tile_pos_size :: proc(the_map: Map, row_i, column_i: int) -> (v2, v2) {
	return {
		the_map.pos.x + cast(f32)column_i * the_map.cell_tex_size.x * the_map.scale.x,
		the_map.pos.y - cast(f32)row_i * the_map.cell_tex_size.y * the_map.scale.y
	}, the_map.cell_tex_size * the_map.scale
}

map_draw :: proc(the_map: Map) {
	for row, row_i in the_map.tiles {
		for cell, column_i in row {
			the_tex : int

			switch cell {
			case .Wall: the_tex = g_textures.wall
			case .Pit: the_tex = g_textures.pit
			case .Ground, .PlayerSpawn, .Goal: the_tex = g_textures.ground
			case .CrateWood : the_tex = g_textures.crate_wood
			case .CrateStone : the_tex = g_textures.crate_stone
			}

			pos, size := map_get_tile_pos_size(the_map, row_i, column_i)

			if cell == .Pit do  ldx.draw_solid_rect(pos, size, COLOR_BLACK)
			ldx.draw_texture(the_tex, pos, the_map.scale)
			if cell == .Goal do  ldx.draw_texture(g_textures.goal, pos, the_map.scale)
		}
	}
}

Interval :: struct{min,max:f32}

interval_collide :: proc(a,b :Interval) -> bool {
	return a.max > b.min && a.min < b.max
}

interval_add :: proc(interval : Interval, sum: f32) -> Interval {
	return {interval.min + sum, interval.max + sum}
}

AABB :: struct{x, y: Interval}

aabb_do_collide :: proc(a, b: AABB) -> bool{
	return interval_collide(a.x, b.x) && interval_collide(a.y, b.y)
}

aabb_translate :: proc(aabb:AABB, pos: v2) -> AABB {
	return AABB{interval_add(aabb.x, pos.x), interval_add(aabb.y, pos.y)}
}

map_coord_to_world_pos :: proc(the_map: Map, coord: Coord) -> v2 {

	coord_f := v2i_to_v2(coord)
	coord_f.y *= -1

	return the_map.pos + the_map.cell_tex_size * the_map.scale * coord_f
}

v2i_to_v2 :: proc(coord: v2i) -> v2 {
	return {cast(f32)coord.x, cast(f32)coord.y}
}
