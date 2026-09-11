package hookin

import "core:c"
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


AUDIO_ENABLE :: false

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

g_start_map: Map
g_map: Map
g_textures: Textures
g_player : Player
g_lives : int
g_times_level_win: int
g_last_event: GameEvent

// TODO do keyboard system on lucy2d (snapshot of prev frame keys and current frame to see which one started being pressed now)
g_was_space_pressed: bool
g_was_tab_pressed: bool
g_mouse_clicked: bool

g_play_mode : enum {Play, Editor}

g_mouse_buttons: u32
g_mouse_world_pos: v2

main :: proc() {
	ldx.window_new("hookin", WINDOW_WIDTH, WINDOW_HEIGHT)
	g_lives = 3

	when AUDIO_ENABLE {
	ok := audio.init()
	if !ok {
		// handle error
	}
	midi_track, success := audio.load_midi_file("hookin/audio/ct600ad.mid")
	if !success {
		// handle error
		fmt.println("could not load midi file.")
	}
	audio.play_midi(&midi_track)
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

	map_start_default(&g_start_map)
	game_restart()

	outer: for !ldx.window_should_close() {
		ldx.frame_start()
		kb := ldx.get_keyboard()

		// get mouse pos
		m_x, m_y: c.int
		g_mouse_buttons = sdl.GetMouseState(&m_x, &m_y)
		g_mouse_world_pos = v2{cast(f32)m_x, cast(f32)-m_y + WINDOW_HEIGHT}

		when AUDIO_ENABLE {
		audio.update()
		}

		defer {
			ldx.frame_end()
			free_all(context.temp_allocator)

			// doing kb stuff

			// TODO: super crude temporary code. do input system on lucy2d now!
			g_was_space_pressed = kb[sdl.Scancode.SPACE] == 1
			g_was_tab_pressed = kb[sdl.Scancode.TAB] == 1
			g_mouse_clicked = g_mouse_buttons & 0x01 != 0
		}

		switch g_play_mode {
		case .Play:
			if game_update(kb) do break outer
		case .Editor:
			if editor_update(kb) do break outer
		}
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
		draw_ground: bool

		pos, size := map_get_tile_pos_size(tm, e.coord)

		// TODO rest
		switch e.et {
		case .Wall: 
			ldx.draw_texture(g_textures.wall, pos, tm.scale)
		case .Pit: 
			ldx.draw_solid_rect(pos, size, COLOR_BLACK)
			ldx.draw_texture(g_textures.pit, pos, tm.scale)
		case .Goal:
			ldx.draw_texture(g_textures.goal, pos, tm.scale)
		case .Crate:
			ldx.draw_texture(g_textures.crate_wood, pos, tm.scale)
		case .Player, .PlayerSpawn, .Nothing: // player is drawn separately
		}
	}
}

v2i_to_v2 :: proc(coord: v2i) -> v2 {
	return {cast(f32)coord.x, cast(f32)coord.y}
}

v2_to_v2i :: proc(a: v2) -> v2i {
	return {cast(int)a.x, cast(int)a.y}
}

game_restart :: proc() {
	map_copy(&g_map, g_start_map)

	// Placing player at spawn position
	psc := map_get_player_spawn_coord(g_map)
	box_place_at_coord(&g_player, psc)
	g_player.current_coord = psc
}

player_kill :: proc() {
	g_last_event = .PlayerDied
	g_lives -= 1
	when AUDIO_ENABLE {
	audio.play_note(.A, 2, 0.1, 127, 9)
	}
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
			when AUDIO_ENABLE {
			audio.play_note(.F, 2, 0.1, 127, 9)
			}
			g_last_event = .BeatLevel
			// go to the next level i guess?
			game_restart()
		}
	}

	g_player.current_coord = coord
}

// u gotta do this one again
try_move_box :: proc(bcr: BoxCollisionRecord) {
	@static box_i_last_hit: int
	@static hit_counter: int

	if !bcr.did_hit {
		hit_counter = 0
		return
	}

	e := entity_get(&g_map, bcr.eids[bcr.box_i])

	#partial switch e.et {
	case .Crate:

		hit_counter += 1
		if box_i_last_hit != bcr.box_i {
			hit_counter = 0
		}
		box_i_last_hit = bcr.box_i
		if hit_counter < 20 do break

		dir_int := v2_to_v2i(-bcr.col_normal)
		dir_int.y *= -1
		coord_from := e.coord
		coord_to := coord_from + dir_int
		ents_query := map_tquery(&g_map, coord_to)

		is_solid_on_other_side: bool

		for e_q in ents_query {
			if entity_is_solid(e_q^) {
				// there's a solid on the other side. abort the move
				is_solid_on_other_side = true
				break
			}
		}

		if !is_solid_on_other_side {
			// perform move
			move_box(e, coord_to)
			hit_counter = 0
		}
	}
}

move_box :: proc(e: ^Entity, c: Coord) {
	ent_lookup := map_tquery(&g_map, c)

	for eq in ent_lookup {
		if eq.et == .Pit || entity_is_solid(eq^) {
			e.et = .Nothing
			return
		}
	}

	e.coord = c
}

game_update :: #force_inline proc(kb: []u8) -> (_should_quit: bool) {

	if kb[sdl.Scancode.TAB] == 1 && !g_was_tab_pressed {
		// switching to editor mode
		editor_init()
		fmt.printfln("switching to editor mode")
		g_play_mode = .Editor
		return false
	}

	if kb[sdl.Scancode.ESCAPE] == 1 do return true
	ldx.window_clear(COLOR_BACKGROUND)

	// Update game logic
	if kb[sdl.Scancode.R] == 1 do game_restart()

	// What tile is the player in?
	player_coord := map_world_box_to_coord(g_map, g_player.box)

	if player_coord != g_player.current_coord {
		player_coord_changed(player_coord, false)
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
			map_boxes, ids := generate_collisions(g_map)
			box_i, col_normal, did_hit := move_and_slide(&g_player.box, map_boxes[:])
			bcr := BoxCollisionRecord { map_boxes, ids, box_i, col_normal, did_hit }
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

	}

	// Drawing everything
	{
		map_draw(g_map)

		// Draw the player
		ldx.draw_texture(g_textures.player, g_player.pos + g_player.texture_offset)

		// Draw player hitbox
		// ldx.draw_solid_rect(g_player.pos, g_player.size, {1,0,0,0.5})

		// draw where player is on the coord screen
		p_coord_pos := map_coord_to_world_pos(g_map, player_coord)
		ldx.draw_wirebox(p_coord_pos, g_map.cell_tex_size, {0,1,0, 1.0}, 5)

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
	// get current coord and look up coords along last input vel
	// loop tiles along that vel

	last_vel := g_player.last_input_vel
	lookup_coord := g_player.current_coord + last_vel

	distance : int = 1
	hit_wooden_box : bool

	ent_lookup : []^Entity

	outer: for {
		if distance > 30 do return

		ent_lookup = map_tquery(&g_map, lookup_coord)

		for e in ent_lookup {
			#partial switch e.et {
			case .Wall:
				// hit wall. no crate affected. return
				return
			case .Crate:
				// Hook the crate
				move_box(e, g_player.current_coord + g_player.last_input_vel)

				// center player in the coord, to avoid bugs
				box_place_at_coord(&g_player, g_player.current_coord)
				return
			}
		}

		lookup_coord += last_vel
		distance += 1
	}
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
generate_collisions :: proc(the_map: Map) -> ([]Box, []int) {

	map_boxes := make([dynamic]Box, 0, len(the_map.entities), context.temp_allocator)
	ids := make([dynamic]int, 0, len(the_map.entities), context.temp_allocator)

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
		append(&ids, i)
	}

	return map_boxes[:], ids[:]
}
