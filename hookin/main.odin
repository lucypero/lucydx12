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

WINDOW_WIDTH :: 1000
WINDOW_HEIGHT :: 500
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
	tiles: [][]Tile
}

Textures :: struct {
	player, crate_wood, ground, wall, crate_stone, goal, pit: int
}

Player :: struct {
	using box: Box,// box for collision
	texture_size: v2,
}

g_textures: Textures
g_player : Player

main :: proc() {
	ldx.window_new("hookin", WINDOW_WIDTH, WINDOW_HEIGHT)

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
	g_player.texture_size = v2i_to_v2(char_tex_size_i)
	g_player.box.size = g_player.texture_size

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

	// Placing player at position
	g_player.pos = map_coord_to_world_pos(the_map, the_map.player_spawn_coord)

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

		// Moving player
		{
			vel : v2

			if kb[sdl.Scancode.A] == 1 do vel.x = -1
			if kb[sdl.Scancode.D] == 1 do vel.x = 1
			if kb[sdl.Scancode.W] == 1 do vel.y = 1
			if kb[sdl.Scancode.S] == 1 do vel.y = -1

			if vel != {0,0} do vel = linalg.normalize(vel) * CHARACTER_SPEED

			g_player.vel = vel

			map_boxes := map_generate_collisions(the_map)


			move_and_slide(&g_player.box, map_boxes[:])

		}

		// Drawing everything
		{
			map_draw(the_map)

			// Draw the player
			ldx.draw_texture(g_textures.player, g_player.pos)

			// Draw player hitbox
			ldx.draw_solid_rect(g_player.pos, g_player.size, {1,0,0,0.5})
		}

		ldx.present()
		free_all(context.temp_allocator)
	}

	ldx.window_cleanup()
}

// Move and slide
// character_move :: proc(the_map: Map) {

// 	pos_future := g_char_pos + dir

// 	hit_thing := false

// 	aabb_player := AABB{{0,g_char_tex_size.x}, {0,g_char_tex_size.y}}

// 	aabb_future := aabb_translate(aabb_player, pos_future)

// 	for row, row_i in the_map.tiles {
// 		for cell, column_i in row {
// 			#partial switch cell {
// 			case .Wall, .CrateWood,.CrateStone:
// 				pos, size := map_get_tile_pos_size(the_map, row_i, column_i)
// 				aabb_tile := aabb_translate(AABB{{0,size.x}, {0,size.y}}, pos)
// 				if aabb_do_collide(aabb_future, aabb_tile) {
// 					hit_thing = true
// 					return
// 				}
// 			case:
// 				continue
// 			}
// 		}
// 	}
// 	g_char_pos = pos_future
// }

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

// True if there's a collision between the AABB's
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

box_swept :: proc(b1, b2 :Box) -> (collision_time: f32, collision_normal: v2) {
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
move_and_slide :: proc(moving_box: ^Box, static_boxes: []Box) {

	@static counter : int = 0

	// Fix collision on multiple boxes at once by running the code below 4 times on the remaining velocity, or something like that;
	outer: for i in 0..<4 {

		col_normal: v2
		col_time := math.inf_f32(1)
		last_hittable_faces: Faces

		if moving_box.vel == {0,0} do break

		for static_box in static_boxes {
			box_broadphase := box_get_broadphase(moving_box^)
			if !box_does_hit_box(box_broadphase, static_box) do continue
			col_time_i, col_normal_i := box_swept(moving_box^, static_box)

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
				last_hittable_faces = static_box.hittable_faces
			}
		}

		// // there was collision
		if col_time < 1 {

			// moving the box right next to the obstacle
			moving_box.pos += moving_box.vel * col_time

			// Sliding
			remaining_time := 1.0 - col_time
			dotprod := (moving_box.vel.x * col_normal.y + moving_box.vel.y * col_normal.x) * remaining_time

			next_vel :=v2{dotprod * col_normal.y, dotprod * col_normal.x}

			fmt.printfln("i: %v, coltime: %v, normal: %v, vel tested:%v, next: %v", i, col_time, col_normal, moving_box.vel, next_vel)
			moving_box.vel =  next_vel
			counter += 1
			// moving_box.pos += moving_box.vel
		} else {
			break outer
		}
	}

	moving_box.pos += moving_box.vel

	// if !was_collision {
	// 	moving_box.pos += moving_box.vel
	// }
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

map_generate_collisions :: proc(the_map: Map) -> []Box {

	map_boxes := make([dynamic]Box, 0, map_get_tile_count(the_map), context.temp_allocator)

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
		}
	}

	return map_boxes[:]
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

@(require_results)
map_get_tile :: proc(the_map: Map, coord: Coord) -> (tile: Tile, ok: bool = false) {
	row_size := len(the_map.tiles[0])
	if !(coord.x >= 0 && coord.x < row_size) do return
	if !(coord.y >= 0 && coord.y < len(the_map.tiles)) do return
	return the_map.tiles[coord.y][coord.x], true
}

@(require_results)
map_get_tile_unchecked :: proc(the_map: Map, coord: Coord) -> (tile: Tile) {
	return the_map.tiles[coord.y][coord.x]
}
