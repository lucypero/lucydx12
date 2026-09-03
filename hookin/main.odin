package hookin

import "core:fmt"
import "core:math"
import "core:math/linalg"
// import "core:math/rand"
// import "core:math/linalg"
import sdl "vendor:sdl2"
import "audio"

// Importing rendering engine
import ldx "../src"

v2i :: ldx.v2i
v2 :: ldx.v2
v4 :: ldx.v4

// Map coordinate. origin at TOP LEFT of the map, visually and in data the_map[0][0]
Coord :: v2i

ROW_COUNT :: 7
COLUMN_COUNT :: 7

WINDOW_WIDTH :: 1000
WINDOW_HEIGHT :: 800
COLOR_BACKGROUND :: v4{0.773, 0.686, 0.643,1}
COLOR_CHARACTER :: v4{0.8, 0.494, 0.522, 1}
COLOR_FOOD :: v4{0.639, 0.427, 0.565, 1}
COLOR_BLACK :: v4{0,0,0,1}
CHARACTER_SPEED :: 3
CHARACTER_SIZE :: 64

// Collision Box

Face :: enum {Left, Right, Top, Bottom}
Faces :: bit_set[Face; u8]

Box :: struct {
	pos : v2, // position of top left of the box
	size: v2, // dimensions
	vel: v2,  // velocity
	hittable_faces: Faces
}

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
	tiles: [ROW_COUNT][COLUMN_COUNT]Tile,
}

Textures :: struct {
	player, crate_wood, ground, wall, crate_stone, goal, pit: int
}

Player :: struct {
	using box: Box,// box for collision
	texture_size: v2,
	texture_offset: v2,
}

g_map: Map
g_textures: Textures
g_player : Player
g_player_coord: Coord
g_lives : int
g_times_level_win: int

main :: proc() {
	ldx.window_new("hookin", WINDOW_WIDTH, WINDOW_HEIGHT)
	g_lives = 3

	ok := audio.init()
	if !ok {
		// handle error
	}
	midi_track, success := audio.load_midi_file("hookin/audio/ct600ad.mid")
	if !success {
		// handle error
		fmt.println("could not load midi file.")
	}

	g_textures.player = ldx.texture_load("hookin_sprites/sokoban-pack/Player/player_01.png")
	g_textures.crate_wood = ldx.texture_load("hookin_sprites/sokoban-pack/Crates/crate_07.png")
	g_textures.ground = ldx.texture_load("hookin_sprites/sokoban-pack/Ground/ground_01.png")
	g_textures.wall = ldx.texture_load("hookin_sprites/sokoban-pack/Blocks/block_01.png")
	g_textures.crate_stone = ldx.texture_load("hookin_sprites/sokoban-pack/Blocks/block_02.png")
	g_textures.goal = ldx.texture_load("hookin_sprites/sokoban-pack/Environment/environment_10.png")
	g_textures.pit = ldx.texture_load("hookin_sprites/sokoban-pack/Environment/environment_06.png")

	char_tex_size_i := ldx.texture_get_size(g_textures.player)

	// initializing player
	g_player.texture_size = v2i_to_v2(char_tex_size_i)
	g_player.box.size = g_player.texture_size * 0.7
	g_player.texture_offset = v2{ -10, 10}

	game_restart()

	audio.play_midi(&midi_track)

	for !ldx.window_should_close() {
		audio.update()
		kb := ldx.get_keyboard()
		if kb[sdl.Scancode.ESCAPE] == 1 do break
		ldx.window_clear(COLOR_BACKGROUND)

		// Update game logic

		// Moving player
		{
			vel : v2

			if kb[sdl.Scancode.A] == 1 do vel.x = -1
			if kb[sdl.Scancode.D] == 1 do vel.x = 1
			if kb[sdl.Scancode.W] == 1 do vel.y = 1
			if kb[sdl.Scancode.S] == 1 do vel.y = -1

			if vel != {0,0} do vel = linalg.normalize(vel) * CHARACTER_SPEED
			g_player.vel = vel
			map_boxes, coords := map_generate_collisions(g_map)
			box_i, col_normal, did_hit := move_and_slide(&g_player.box, map_boxes[:])

			@static box_i_last_hit: int
			@static hit_counter: int

			if did_hit {
				tile := map_get_tile_unchecked(g_map, coords[box_i])

				#partial switch tile {
				case .CrateWood:

					hit_counter += 1

					if box_i_last_hit != box_i {
						hit_counter = 0
					}

					box_i_last_hit = box_i


					if hit_counter > 20 {
						move_box(&g_map, coords[box_i], -col_normal)
						hit_counter = 0
					}
				}
			} else {
				hit_counter = 0
			}
		}

		// What tile is the player in?
		coord := map_world_box_to_coord(g_map, g_player.box)

		if coord != g_player_coord {
			// changed cord.
			// TODO: trigger on tile enter event, or whatever.

			tile := map_get_tile_unchecked(g_map, coord)
			#partial switch tile {
			case .Goal: // Goal. you won
				g_times_level_win += 1
				audio.play_note(.F, 2, 0.1, 127, 9)
				// go to the next level i guess?
				game_restart()
			case .Pit: // landed on pit. die
				g_lives -= 1
				note := cast(audio.Notes)coord.y
				audio.play_note(note, 2, 0.1, 127, 9)
				game_restart()
			}

			g_player_coord = coord
		}

		// Drawing everything
		{
			map_draw(g_map)

			// Draw the player
			ldx.draw_texture(g_textures.player, g_player.pos + g_player.texture_offset)

			// Draw player hitbox
			// ldx.draw_solid_rect(g_player.pos, g_player.size, {1,0,0,0.5})

			// draw where player is on the coord screen
			// p_coord_pos := map_coord_to_world_pos(g_map, coord)
			// ldx.draw_solid_rect(p_coord_pos, g_map.cell_tex_size, {1,1,0,0.5})

			// drawing amount of lives
			for i in 0..<g_lives {
				ldx.draw_solid_rect({10 + 55 * cast(f32)i, 5 + 50}, {50, 50}, {1,0,0,1})
			}

			for i in 0..<g_times_level_win {
				ldx.draw_solid_rect({400 + 55 * cast(f32)i, 5 + 50}, {50, 50}, {0,1,0,1})
			}
		}

		ldx.present()
		free_all(context.temp_allocator)
	}

	ldx.window_cleanup()
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
			draw_ground: bool

			switch cell {
			case .Wall: the_tex = g_textures.wall
			case .Pit: the_tex = g_textures.pit
			case .Ground, .PlayerSpawn: the_tex = g_textures.ground
			case .Goal: the_tex = g_textures.goal
			case .CrateWood : the_tex = g_textures.crate_wood
			case .CrateStone : the_tex = g_textures.crate_stone
			}

			pos, size := map_get_tile_pos_size(the_map, row_i, column_i)
			if cell == .Pit do  ldx.draw_solid_rect(pos, size, COLOR_BLACK)

			// Drawing ground  on certain tile types, before the main element
			switch cell {
			case .CrateWood, .CrateStone,.Goal, .Wall:
				ldx.draw_texture(g_textures.ground, pos, the_map.scale)
			case .Ground, .Pit, .PlayerSpawn:
			}

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

// True if there's a collision between the AABB's
aabb_do_collide :: proc(a, b: AABB) -> bool {
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

v2_to_v2i :: proc(a: v2) -> v2i {
	return {cast(int)a.x, cast(int)a.y}
}

box_sweep :: proc(b1, b2 :Box) -> (collision_time: f32, collision_normal: v2) {
	inv_entry : v2
	inv_exit : v2

	// find the distance between the objects on the near and far sides for both x and y 
	if b1.vel.x > 0.0 {
		inv_entry.x = b2.pos.x - (b1.pos.x + b1.size.x)
		inv_exit.x = (b2.pos.x + b2.size.x) - b1.pos.x
	} else {
		inv_entry.x = (b2.pos.x + b2.size.x) - b1.pos.x
		inv_exit.x = b2.pos.x - (b1.pos.x + b1.size.x)
	} 

	if b1.vel.y > 0.0 {
		inv_entry.y = (b2.pos.y - b2.size.y) - b1.pos.y
		inv_exit.y = b2.pos.y - (b1.pos.y - b1.size.y)
	} else {
		inv_entry.y = b2.pos.y - (b1.pos.y - b1.size.y)
		inv_exit.y = (b2.pos.y - b2.size.y) - b1.pos.y
	}

	// find time of collision and time of leaving for each axis (if statement is to prevent divide by zero) 
	entry, exit: v2

	if (b1.vel.x == 0.0) {
		// if there's no overlap in X, there is no colission
		if (b1.pos.x + b1.size.x <= b2.pos.x) || (b1.pos.x >= b2.pos.x + b2.size.x) {
			return 1.0, {0, 0}
		}
		entry.x = math.inf_f32(-1)
		exit.x = math.inf_f32(1)
	} else {
		entry.x = inv_entry.x / b1.vel.x; 
		exit.x = inv_exit.x / b1.vel.x; 
	} 

	if (b1.vel.y == 0.0) {
		// if there's no overlap in Y, there is no colission
		if (b1.pos.y <= b2.pos.y - b2.size.y) || (b1.pos.y - b1.size.y >= b2.pos.y) {
			return 1.0, {0, 0}
		}
		entry.y = math.inf_f32(-1)
		exit.y = math.inf_f32(1)
	} else {
		entry.y = inv_entry.y / b1.vel.y; 
		exit.y = inv_exit.y / b1.vel.y; 
	}

	// find the earliest/latest times of collisionfloat 
	entry_time := max(entry.x, entry.y)
	exit_time := min(exit.x, exit.y)

	// if there was no collision
	if (entry_time >= exit_time) || entry_time < 0 || entry_time > 1 {
		collision_normal = {0, 0}
		collision_time = 1.0
	} else {  // if there was a collision
		// The normal always opposes the velocity on the axis we entered through.
		if (entry.x > entry.y) {
			collision_normal = b1.vel.x > 0 ? {-1,0} : {1, 0}
		} else {
			collision_normal = b1.vel.y > 0 ? {0,-1} : {0,1}
		} 
		collision_time = entry_time
	}

	return
}

// Collision: Testing player against boxes
move_and_slide :: proc(moving_box: ^Box, static_boxes: []Box) -> (_box_i: int, _col_normal: v2, _did_hit: bool){
	outer: for _ in 0..<4 {

		col_normal: v2
		col_time := math.inf_f32(1)

		if moving_box.vel == {0,0} do break

		for static_box, box_i in static_boxes {
			box_broadphase := box_get_broadphase(moving_box^)
			if !box_does_hit_box(box_broadphase, static_box) do continue
			col_time_i, col_normal_i := box_sweep(moving_box^, static_box)

			// Discarding collisions on internal edges
			is_discarded : bool

			if col_normal_i == {1,0} && .Right not_in static_box.hittable_faces do is_discarded = true
			if col_normal_i == {-1,0} && .Left not_in static_box.hittable_faces do is_discarded = true
			if col_normal_i == {0,1} && .Top not_in static_box.hittable_faces do is_discarded = true
			if col_normal_i == {0,-1} && .Bottom not_in static_box.hittable_faces do is_discarded = true

			if is_discarded {
				continue
			}

			if col_time_i < col_time {
				col_normal = col_normal_i
				col_time = col_time_i

				// Saving box we hit for return value
				_box_i = box_i
			}
		}

		// there was collision
		if col_time < 1 {

			// Setting return values
			_did_hit = true
			_col_normal = col_normal

			// moving the box right next to the obstacle
			moving_box.pos += moving_box.vel * col_time

			// Sliding
			remaining_time := 1.0 - col_time
			dotprod := (moving_box.vel.x * col_normal.y + moving_box.vel.y * col_normal.x) * remaining_time
			next_vel := v2{dotprod * col_normal.y, dotprod * col_normal.x}
			// Setting the box's velocity as the slide velocity, and sweeping again before committing to a move.
			moving_box.vel = next_vel
		} else { // no collision. skip all other collision tests
			break outer
		}
	}

	moving_box.pos += moving_box.vel
	return
}

box_get_broadphase :: proc(b: Box) -> (broadphase_box: Box) {
	broadphase_box.pos.x = b.vel.x > 0 ? b.pos.x : b.pos.x + b.vel.x
	broadphase_box.pos.y = b.vel.y > 0 ? b.pos.y + b.vel.y : b.pos.y
	broadphase_box.size.x = b.vel.x > 0 ? b.vel.x + b.size.x : b.size.x - b.vel.x  
	broadphase_box.size.y = b.vel.y > 0 ? b.vel.y + b.size.y : b.size.y - b.vel.y  
	return
}

box_does_hit_box :: proc(b1,b2: Box) -> bool {
	return !((b1.pos.x + b1.size.x < b2.pos.x) ||
		(b1.pos.x > b2.pos.x + b2.size.x) ||
		(b1.pos.y < b2.pos.y - b2.size.y) ||
		(b1.pos.y - b1.size.y > b2.pos.y))
}

map_get_tile_count :: proc(the_map: Map) -> int {
	return len(the_map.tiles) * len(the_map.tiles[0])
}

map_generate_collisions :: proc(the_map: Map) -> ([]Box, []Coord) {

	map_boxes := make([dynamic]Box, 0, map_get_tile_count(the_map), context.temp_allocator)
	coords := make([dynamic]Coord, 0, map_get_tile_count(the_map), context.temp_allocator)

	for row, row_i in the_map.tiles {
		for _, column_i in row {
			if !tile_is_solid(the_map, Coord{column_i, row_i}) do continue

			// Construct Box
			tile_box : Box
			tile_box.pos, tile_box.size = map_get_tile_pos_size(the_map, row_i, column_i)

			if !tile_is_solid(the_map, {column_i+1, row_i}) do tile_box.hittable_faces |= {.Right}
			if !tile_is_solid(the_map, {column_i-1, row_i}) do tile_box.hittable_faces |= {.Left}
			if !tile_is_solid(the_map, {column_i, row_i + 1}) do tile_box.hittable_faces |= {.Bottom}
			if !tile_is_solid(the_map, {column_i, row_i - 1}) do tile_box.hittable_faces |= {.Top}
			append(&map_boxes, tile_box)
			append(&coords, v2i{column_i, row_i})
		}
	}

	return map_boxes[:], coords[:]
}

tile_is_solid :: proc(the_map: Map, coord: Coord) -> bool {
	tile, ok := map_get_tile(the_map, coord)
	if !ok do return true

	switch tile {
	case .Wall, .CrateWood,.CrateStone:
		return true
	case .Ground, .Pit, .Goal, .PlayerSpawn:
		fallthrough
	case:
		return false
	}
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
map_get_tile :: proc(the_map: Map, coord: Coord) -> (tile: Tile, ok: bool = false) {
	row_size := len(the_map.tiles[0])
	if !(coord.x >= 0 && coord.x < row_size) do return
	if !(coord.y >= 0 && coord.y < len(the_map.tiles)) do return
	return the_map.tiles[coord.y][coord.x], true
}

map_get_tile_ref :: proc(the_map: ^Map, coord: Coord) -> (tile: ^Tile) {
	return &the_map.tiles[coord.y][coord.x]
}

@(require_results)
map_get_tile_unchecked :: proc(the_map: Map, coord: Coord) -> (tile: Tile) {
	return the_map.tiles[coord.y][coord.x]
}

game_restart :: proc() {

	// initting map
	map_tiles : [ROW_COUNT][COLUMN_COUNT]Tile = {
		{.Wall, .Wall, .Wall, .Wall, .Wall, .Wall, .Wall},
		{.Wall, .Ground, .Ground, .CrateStone, .Ground, .Pit, .Wall},
		{.Wall, .Ground, .Ground, .Ground, .Ground, .CrateWood, .Wall},
		{.Wall, .Ground, .CrateWood, .Ground, .CrateWood, .Goal, .Wall},
		{.Wall, .PlayerSpawn, .Ground, .CrateWood, .Ground, .Ground, .Wall},
		{.Wall, .Ground, .Ground, .Ground, .Ground, .Ground, .Wall},
		{.Wall, .Wall, .Wall, .Wall, .Wall, .Wall, .Wall}
	}

	cell_tex_size := ldx.texture_get_size(g_textures.wall)
	cell_tex_size_f := v2{cast(f32)cell_tex_size.x, cast(f32)cell_tex_size.y}
	player_spawn_coord : Coord

	outer: for row_i, y in map_tiles {
		for col_i, x in row_i {
			if col_i == .PlayerSpawn {
				player_spawn_coord = {x, y}
				break outer
			}
		}
	}

	g_map = {
		v2{50, WINDOW_HEIGHT - 10},
		v2{1,1},
		cell_tex_size_f, 
		player_spawn_coord,
		map_tiles
	}

	// placing player at spawn position

	g_player.pos = map_coord_to_world_pos(g_map, g_map.player_spawn_coord)
	g_player_coord = g_map.player_spawn_coord
}

move_box :: proc(tm: ^Map, c: Coord, dir: v2) {
	tile_box := map_get_tile_unchecked(tm^, c)

	dir_int := v2_to_v2i(dir)
	dir_int.y *= -1
	next_coord := c + dir_int
	tile_next, ok := map_get_tile(tm^, next_coord)
	if !ok do return

	// We do different things depending on what is the next tile
	switch tile_next {
	case .Wall, .CrateWood, .CrateStone: return
	case .PlayerSpawn, .Ground, .Goal:
		// move the box
		tile_prev := map_get_tile_ref(tm, c)
		tile_prev^ = .Ground

		tile_next := map_get_tile_ref(tm, next_coord)
		tile_next^ = tile_box
	case .Pit:
		// disappear the box
		tile_prev := map_get_tile_ref(tm, c)
		tile_prev^ = .Ground
	}
}
