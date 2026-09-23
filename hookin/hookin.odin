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
import l2d "../lucy2d"
import ldx "../lucydx"

// Core types
v2 :: ldx.v2
v3 :: ldx.v3
v4 :: ldx.v4
v2i :: ldx.v2i
dxm :: ldx.dxm

AUDIO_ENABLE :: false
START_ON_EDITOR :: true

ROW_COUNT :: 7
COLUMN_COUNT :: 7

WINDOW_WIDTH_START :: 1400
WINDOW_HEIGHT_START :: 800
COLOR_BACKGROUND :: v4{0.773, 0.686, 0.643,1}
COLOR_CHARACTER :: v4{0.8, 0.494, 0.522, 1}
COLOR_FOOD :: v4{0.639, 0.427, 0.565, 1}
COLOR_BLACK :: v4{0,0,0,1}


Textures :: struct {
	player, crate_wood, ground, wall, crate_stone, goal, pit, move_hand, trash, spawn, hook, paint, rect_tool, crate_metal: int
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
g_input_disabled: bool

g_play_mode : enum {Play, Editor}
g_frame_i : int

HookVisual :: struct {
	distance: int,
	from, to: Coord,
	visible: bool
}

g_hook: HookVisual

main :: proc() {
	l2d.window_new("hookin", WINDOW_WIDTH_START, WINDOW_HEIGHT_START)
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

	HOOKIN_ASSETS_DIR :: "hookin/assets"

	g_textures.move_hand = l2d.texture_load(HOOKIN_ASSETS_DIR+"/hand.png")
	g_textures.trash = l2d.texture_load(HOOKIN_ASSETS_DIR+"/trashcanOpen.png")
	g_textures.hook = l2d.texture_load(HOOKIN_ASSETS_DIR+"/arrow_e.png")
	g_textures.paint = l2d.texture_load(HOOKIN_ASSETS_DIR+"/drawing_bucket.png")
	g_textures.rect_tool = l2d.texture_load(HOOKIN_ASSETS_DIR+"/element_red_rectangle.png")

	init_tile_registry()
	player_init(&g_player)


	map_start_default(&g_start_map)
	game_restart()

	outer: for !l2d.window_should_close() {
		l2d.frame_start()

		when AUDIO_ENABLE {
		audio.update()
		}

		tweens_update()
		timers_tick()

		defer {
			l2d.frame_end()
			free_all(context.temp_allocator)

			// doing kb stuff
			g_frame_i += 1
		}

		switch g_play_mode {
		case .Play:
			if game_update() do break outer
		case .Editor:
			if editor_update() do break outer
		}

	}

	l2d.window_cleanup()
}

game_restart :: proc() {
	map_copy(&g_map, g_start_map)

	// Placing player at spawn position
	psc := map_get_player_spawn_coord(g_map)
	box_place_at_coord(&g_player, psc)
	g_player.current_coord = psc
}

player_kill :: proc() {
	disable_input_for(0.5)
	old_pos := g_player.pos
	g_last_event = .PlayerDied
	g_lives -= 1
	when AUDIO_ENABLE {
	audio.play_note(.A, 2, 0.1, 127, 9)
	}
	game_restart()
	tween_v2(&g_player.vis.offset, old_pos - g_player.pos, {}, 0.3, .EaseOutCubic)
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

		dir_int := l2d.v2_to_v2i(-bcr.col_normal)
		dir_int.y *= -1
		coord_from := e.coord
		coord_to := coord_from + dir_int
		ents_query := map_tquery(&g_map, coord_to)

		is_solid_on_other_side: bool

		for e_q in ents_query {
			if entity_is_solid(e_q.et) {
				// there's a solid on the other side. abort the move
				is_solid_on_other_side = true
				break
			}
		}

		if !is_solid_on_other_side {
			// perform move
			disable_input_for(0.3)
			move_box(e, coord_to)
			hit_counter = 0
		}
	}
}

move_box :: proc(e: ^Entity, c: Coord) {
	ent_lookup := map_tquery(&g_map, c)

	for eq in ent_lookup {
		if eq.et == .Pit || entity_is_solid(eq.et) {
			entity_delete(&g_map, e)
			return
		}
	}

	tween_v2(&e.vis.offset, map_coord_to_world_pos(g_map, e.coord) - map_coord_to_world_pos(g_map, c), {}, 0.6, .EaseOutCubic)

	e.coord = c
}

game_update :: #force_inline proc() -> (_should_quit: bool) {

	if l2d.key_is_just_pressed(.TAB) || 
	(START_ON_EDITOR && g_frame_i == 1) {
		// switching to editor mode
		editor_init()
		g_play_mode = .Editor
		return false
	}

	if l2d.key_is_just_pressed(.ESCAPE) do return true
	l2d.window_clear(COLOR_BACKGROUND)

	// Update game logic
	if l2d.key_is_just_pressed(.R) do game_restart()

	// What tile is the player in?
	player_coord := map_world_box_to_coord(g_map, g_player.box)

	if player_coord != g_player.current_coord {
		player_coord_changed(player_coord, false)
	}

	bcr, ok := player_update(&g_player).?
	if ok {
		try_move_box(bcr)
	}



	// Hook mechanic
	{
		if !g_input_disabled && l2d.key_is_just_pressed(.SPACE) {
			try_do_hook()
		}

	}

	// Drawing everything
	{
		map_draw(g_map, edit_mode = false)

		// Draw the player
		l2d.draw_texture(g_entity_defs[EntityType.Player].tex_id, g_player.pos + g_player.texture_offset + g_player.vis.offset)

		// Draw player hitbox
		// l2d.draw_solid_rect(g_player.pos, g_player.size, {1,0,0,0.5})

		// draw where player is on the coord screen
		// p_coord_pos := map_coord_to_world_pos(g_map, player_coord)
		// l2d.draw_wirebox(p_coord_pos, g_map.cell_tex_size, {0,1,0, 1.0}, 5)

		// drawing hook
		if g_hook.visible {
			h := &g_hook
			times_to_render := (h.distance - 1) * 3
			for i in 0..<times_to_render {
				t : f32 = cast(f32)i / cast(f32) times_to_render

				hook_from := map_coord_to_world_pos_center(g_map, h.from)
				hook_to := map_coord_to_world_pos_center(g_map, h.to)

				hook_scale :: 2

				// pivot_offset
				tex_pivot_offset : v2 = l2d.texture_get_size_v2(g_textures.hook) * hook_scale / 2
				tex_pivot_offset.y *= -1

				hook_to -= tex_pivot_offset
				hook_from -= tex_pivot_offset

				// make the center the pivot
				l2d.draw_texture(g_textures.hook, t * hook_from + (1 - t) * hook_to, {hook_scale, hook_scale}, {1 * t,1,1,1})
			}

		}

		// drawing amount of lives
		for i in 0..<g_lives {
			l2d.draw_solid_rect({10 + 55 * cast(f32)i, 5 + 50}, {50, 50}, {1,0,0,1})
		}

		for i in 0..<g_times_level_win {
			l2d.draw_solid_rect({400 + 55 * cast(f32)i, 5 + 50}, {50, 50}, {0,1,0,1})
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
				if distance <= 1 do return


				disable_input_for(0.4)

				// Hook the crate
				move_box(e, g_player.current_coord + g_player.last_input_vel)

				// center player in the coord, to avoid bugs
				old_pos := g_player.pos
				box_place_at_coord(&g_player, g_player.current_coord)
				tween_v2(&g_player.vis.offset, old_pos - g_player.pos, {}, 0.3, .EaseOutCubic)

				// Performing the hook!!!
				g_hook = HookVisual {
					distance, g_player.current_coord, lookup_coord, true
				}
				timer(0.3, proc(_:rawptr) {g_hook.visible = false})

				return
			case .MetalCrate:
				old_pos := g_player.pos
				box_place_at_coord(&g_player, e.coord - g_player.last_input_vel)
				tween_v2(&g_player.vis.offset, old_pos - g_player.pos, {}, 0.3, .EaseOutCubic)

				g_hook = HookVisual {
					distance, g_player.current_coord, lookup_coord, true
				}
				timer(0.3, proc(_:rawptr) {g_hook.visible = false})
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
generate_collisions :: proc(the_map: Map) -> ([]Box, []EID) {

	map_boxes := make([dynamic]Box, 0, len(the_map.entities), context.temp_allocator)
	ids := make([dynamic]EID, 0, len(the_map.entities), context.temp_allocator)

	for t,i in the_map.entities {

		coord := t.coord

		if !entity_is_solid(t.et) do continue

		// Construct Box
		tile_box : Box
		tile_box.pos, tile_box.size = map_get_tile_pos_size(the_map, coord)

		if !does_coord_have_solid(the_map, {coord.x+1, coord.y}) do tile_box.hittable_faces |= {.Right}
		if !does_coord_have_solid(the_map, {coord.x-1, coord.y}) do tile_box.hittable_faces |= {.Left}
		if !does_coord_have_solid(the_map, {coord.x, coord.y + 1}) do tile_box.hittable_faces |= {.Bottom}
		if !does_coord_have_solid(the_map, {coord.x, coord.y - 1}) do tile_box.hittable_faces |= {.Top}

		append(&map_boxes, tile_box)
		append(&ids, EID{i, t.gen})
	}

	return map_boxes[:], ids[:]
}

// tweening

TWEENS_MAX_COUNT :: 20

g_tweens: [TWEENS_MAX_COUNT]Tween

TweenEasing :: enum {Linear, EaseOutCubic, EaseOutCirc}

Tween :: struct {
	target: ^v2,
	from, to: v2,
	t, duration: f32,
	easing: TweenEasing
}

tween_v2 :: proc(target: ^v2, from, to: v2, time: f32, ease: TweenEasing) {
	// getting a tween
	t := tween_get(target)
	target^ = from
	t^ = Tween{target, from, to, 0, time, ease}
}

tween_get :: proc(target: ^v2) -> ^Tween {
	tret : ^Tween
	for &t in g_tweens {
		if target == t.target do return &t
		if t.target == nil && tret == nil do tret = &t
	}
	assert(tret != nil, "tweens are full")
	return tret
}

tweens_update :: proc() {
	for &t in g_tweens {
		if t.target == nil do continue
		// advance t

		t.t += l2d.get_dt_sec() * (1 / t.duration)
		// ease out cubic
		the_t :f32 

		switch t.easing {
		case .Linear:
			the_t = t.t
		case .EaseOutCubic:
			the_t = 1 - linalg.pow(1 - t.t, 3)
		case .EaseOutCirc:
			the_t = linalg.sqrt(1.0 - linalg.pow(t.t - 1.0, 2.0))
		}

		if t.t >= 1 {
			// finish tween
			t.target^ = t.to
			t.target = nil
		} else {
			// set value according to T
			next_val := (1 - the_t) * t.from + the_t * t.to
			t.target^ = next_val
		}
	}
}

// Timers

TIMERS_MAX :: 20

Timer :: struct {
	duration: f32,
	data: rawptr,
	trigger: proc(data: rawptr)
}

g_timers : [TIMERS_MAX]Timer

timer :: proc(dur: f32, trigger: proc(data: rawptr), data: rawptr = nil) {
	// Getting a timer
	tim: ^Timer
	for &t in g_timers {
		if t.duration <= 0 {
			tim = &t
			break
		}
	}

	if tim == nil {
		panic("too many timers")
	}

	tim^ = Timer {
		dur, data, trigger
	}
}

timers_tick :: proc() {
	for &t in g_timers {
		if t.duration <= 0 do continue
		t.duration -= cast(f32)l2d.get_dt_sec()
		if t.duration <= 0 {
			t.trigger(t.data)
		}
	}
}

disable_input_for :: proc(dur: f32) {
	g_input_disabled = true
	timer(dur, proc(d:rawptr) {g_input_disabled = false})
}
