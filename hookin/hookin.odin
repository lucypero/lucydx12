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

Textures :: struct {
	player, crate_wood, ground, wall, crate_stone, goal, pit: int
}

Player :: struct {
	using box: Box,// box for collision
	texture_size: v2,
	texture_offset: v2,
	current_coord: Coord,
	last_input_vel: v2i // determines where the player is facing
}

GameEvent :: enum{
	JustStarted,
	PlayerDied,
	BeatLevel
}

g_map: Map
g_textures: Textures
g_player : Player
g_lives : int
g_times_level_win: int
g_last_event: GameEvent

// TODO do keyboard system on lucy2d (snapshot of prev frame keys and current frame to see which one started being pressed now)
g_was_space_pressed: bool

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

	g_map.arena = ldx.arena_allocator_new(context.allocator)
	game_restart()

	audio.play_midi(&midi_track)

	for !ldx.window_should_close() {
		if game_update() do break
	}

	ldx.window_cleanup()
}

map_draw :: proc(tm: Map) {

	//first, draw ground
	for y in 0..<tm.size.y {
		for x in 0..<tm.size.x {
			pos, _ := map_get_tile_pos_size(tm, {x,y})
			ldx.draw_texture(g_textures.ground, pos, tm.scale)
		}
	}

	// loop through entities and draw

	for e, i in tm.entities {
		the_tex : int
		draw_ground: bool

		pos, size := map_get_tile_pos_size(tm, e.coord)

		// TODO rest
		#partial switch e.et {
		case .Wall: 
			ldx.draw_texture(g_textures.ground, pos, tm.scale)
			the_tex = g_textures.wall
			ldx.draw_texture(g_textures.wall, pos, tm.scale)
		case .Pit: 
			the_tex = g_textures.pit
			ldx.draw_solid_rect(pos, size, COLOR_BLACK)
			ldx.draw_texture(g_textures.pit, pos, tm.scale)
		}
	}
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

game_restart :: proc() {

	free_all(g_map.arena)

	map_size := v2i{10, 6}

	// initting map

	cell_tex_size := v2i_to_v2(ldx.texture_get_size(g_textures.wall))

	g_map = {
		g_map.arena,
		v2{50, WINDOW_HEIGHT - 10},
		v2{1,1},
		map_size,
		cell_tex_size,
	{},
	}

	// populating map entities
	append(&g_map.entities, Entity{.PlayerSpawn, 0, {3, 3}})

	for y in 0..<map_size.y {
		for x in 0..<map_size.x {

			top_bottom_row := x == 0 || x == map_size.x - 1
			left_right_col := y == 0 || y == map_size.y - 1

			if top_bottom_row || left_right_col {
				append(&g_map.entities, Entity{.Wall, 0, {x,y}})
			}
		}
	}


	psc := map_get_player_spawn_coord(g_map)

	// Placing player at spawn position

	box_place_at_coord(&g_player, psc)
	g_player.current_coord = psc
}

// TODO do this one again
move_box :: proc(tm: ^Map, c_from, c_to: Coord) -> (_box_did_fall: bool) {
	return true
}

player_kill :: proc() {
	g_last_event = .PlayerDied
	g_lives -= 1
	audio.play_note(.A, 2, 0.1, 127, 9)
	game_restart()
}

// teleport = true: teleport player's pos to the middle of coord
// changed cord.
// TODO: trigger on tile enter event, or whatever.
player_coord_changed:: proc(coord: Coord, teleport: bool) {

	if teleport {
		box_place_at_coord(&g_player, coord)
	}

	entities_on_coord := map_tquery(&g_map, coord)

	for e in entities_on_coord {
		#partial switch e.et {
		case .Pit:
			// fall to pit
			player_kill()
			return
		case .Goal:
			// goal. u won
			g_times_level_win += 1
			audio.play_note(.F, 2, 0.1, 127, 9)
			g_last_event = .BeatLevel
			// go to the next level i guess?
			game_restart()
		}
	}

	g_player.current_coord = coord
}

// u gotta do this one again
try_move_box :: proc(bcr: BoxCollisionRecord) {
	// @static box_i_last_hit: int
	// @static hit_counter: int

	// if bcr.did_hit {
	// 	tile := map_get_tile_unchecked(g_map, bcr.coords[bcr.box_i])
	// 	#partial switch tile {
	// 	case .CrateWood:

	// 		hit_counter += 1
	// 		if box_i_last_hit != bcr.box_i {
	// 			hit_counter = 0
	// 		}
	// 		box_i_last_hit = bcr.box_i
	// 		if hit_counter > 20 {

	// 			dir_int := v2_to_v2i(-bcr.col_normal)
	// 			dir_int.y *= -1
	// 			coord_from := bcr.coords[bcr.box_i]
	// 			coord_to := coord_from + dir_int
	// 			tile_next, ok := map_get_tile(g_map, coord_to)

	// 			if ok {
	// 				move_box(&g_map, coord_from, coord_to)
	// 				hit_counter = 0
	// 			}
	// 		}
	// 	}
	// } else {
	// 	hit_counter = 0
	// }
}

game_update :: #force_inline proc() -> (_should_quit: bool) {
	ldx.frame_start()
	defer {
		ldx.frame_end()
		free_all(context.temp_allocator)
	}
	audio.update()
	kb := ldx.get_keyboard()
	if kb[sdl.Scancode.ESCAPE] == 1 do return true
	ldx.window_clear(COLOR_BACKGROUND)

	// Update game logic
	if kb[sdl.Scancode.R] == 1 do game_restart()

	// What tile is the player in?
	coord := map_world_box_to_coord(g_map, g_player.box)

	if coord != g_player.current_coord {
		player_coord_changed(coord, false)
	}

	// Player update Logic
	{
		vel : v2

		if kb[sdl.Scancode.A] == 1 {
			vel.x = -1 
			g_player.last_input_vel = {-1, 0}
		}
		if kb[sdl.Scancode.D] == 1 {
			vel.x = 1
			g_player.last_input_vel = {1, 0}
		}
		if kb[sdl.Scancode.W] == 1 {
			vel.y = 1
			g_player.last_input_vel = {0, -1}
		} 
		if kb[sdl.Scancode.S] == 1 {
			vel.y = -1
			g_player.last_input_vel = {0, 1}
		}

		if vel != {0,0} {
			vel = linalg.normalize(vel) * CHARACTER_SPEED
			g_player.vel = vel
			map_boxes, coords := map_generate_collisions(g_map)
			box_i, col_normal, did_hit := move_and_slide(&g_player.box, map_boxes[:])
			bcr := BoxCollisionRecord { map_boxes, coords, box_i, col_normal, did_hit }
			try_move_box(bcr)
		} else {
			g_player.vel = {}
		}
	}

	// Hook mechanic
	{
		if kb[sdl.Scancode.SPACE] == 1 && !g_was_space_pressed {
			try_do_hook()
		}

		// TODO: super crude temporary code. do input system on lucy2d now!
		if kb[sdl.Scancode.SPACE] == 1 {
			g_was_space_pressed = true
		} else {
			g_was_space_pressed = false
		}
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

	// Do imgui UI
	{
		im.Begin("hookin")
		defer im.End()
		im.Text("Hi hookin!")
		ldx.imgui_do_text("Player Coord: %v", g_player.current_coord)
		ldx.imgui_do_text("Player looking at: %v", g_player.last_input_vel)
		ldx.imgui_do_text("Last Event: %v", g_last_event)
	}

	return false
}

// TODO:  u gonna have to loop through tiles and also entities now...
try_do_hook :: proc() {
	// // get current coord and look up coords along last input vel
	// // loop tiles along that vel

	// last_vel := g_player.last_input_vel

	// fmt.printfln("last vel %v", last_vel)

	// lookup_coord := g_player.current_coord + last_vel

	// distance : int = 1
	// hit_wooden_box : bool


	// outer: for {
	// 	tile, ok := map_get_tile(g_map, lookup_coord)
	// 	if !ok do break outer

	// 	#partial switch tile.tt {
	// 	case .Wall:
	// 		break outer
	// 	}

	// 	lookup_coord += last_vel
	// 	distance += 1
	// }

	// // Bring crate over if it's distance > 1
	// tile, ok := map_get_tile(g_map, lookup_coord)
	// if !ok do return

	// if distance > 1 && tile == .CrateWood {
	// 	move_box(&g_map, lookup_coord, g_player.current_coord + g_player.last_input_vel)

	// 	// center player in the coord, to avoid bugs
	// 	box_place_at_coord(&g_player, g_player.current_coord)

	// 	// move player (leave this for a special box. not the normal box)
	// 	// coord_player_to := g_player.current_coord - g_player.last_input_vel
	// 	// player_coord_changed(g_player.current_coord - g_player.last_input_vel, true)
	// }
}

box_place_at_coord :: proc(b: ^Box, coord: Coord) {
	coord_pos := map_coord_to_world_pos(g_map, coord)
	tile_offset := g_map.cell_tex_size / 2
	tile_offset.y *= -1

	box_offset := b.size / 2
	box_offset.y *= -1

	b.pos = coord_pos + tile_offset - box_offset
}

// TODO: generate collisions for boxes too?
map_generate_collisions :: proc(the_map: Map) -> ([]Box, []Coord) {

	map_boxes := make([dynamic]Box, 0, len(the_map.entities), context.temp_allocator)
	coords := make([dynamic]Coord, 0, len(the_map.entities), context.temp_allocator)

	for t,i in the_map.entities {

		coord := t.coord

		if !entity_is_solid(t) do continue

		// Construct Box
		tile_box : Box
		tile_box.pos, tile_box.size = map_get_tile_pos_size(the_map, coord)

		if !does_coord_have_solid(the_map, {coord.x+1, coord.y}) do tile_box.hittable_faces |= {.Right}
		if !does_coord_have_solid(the_map, {coord.x-1, coord.y}) do tile_box.hittable_faces |= {.Left}
		if !does_coord_have_solid(the_map, {coord.x, coord.y + 1}) do tile_box.hittable_faces |= {.Bottom}
		if !does_coord_have_solid(the_map, {coord.x, coord.y - 1}) do tile_box.hittable_faces |= {.Top}

		append(&map_boxes, tile_box)
		append(&coords, coord)
	}

	return map_boxes[:], coords[:]
}
