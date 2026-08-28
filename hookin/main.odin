package hookin

// import "core:math/rand"
// import "core:math/linalg"
import sdl "vendor:sdl2"
import "audio"

// Importing rendering engine
import ldx "../src"

v2 :: ldx.v2
v4 :: ldx.v4

WINDOW_WIDTH :: 1000
WINDOW_HEIGHT :: 500
COLOR_BACKGROUND :: v4{0.773, 0.686, 0.643,1}
COLOR_CHARACTER :: v4{0.8, 0.494, 0.522, 1}
COLOR_FOOD :: v4{0.639, 0.427, 0.565, 1}
COLOR_BLACK :: v4{0,0,0,1}
CHARACTER_SPEED :: 2
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
	tiles: [][]Tile
}

Textures :: struct {
	player, crate_wood, ground, wall, crate_stone, goal, pit: int
}

g_textures: Textures

main :: proc() {
	ldx.window_new("hookin", WINDOW_WIDTH, WINDOW_HEIGHT)
	char_pos : v2 = {100, 100}
	char_scale : f32 = 1

	if !audio.init() do return
	audio.set_instrument(0, 30)

	g_textures.player = ldx.texture_load("hookin_sprites/sokoban-pack/Player/player_01.png")
	g_textures.crate_wood = ldx.texture_load("hookin_sprites/sokoban-pack/Crates/crate_07.png")
	g_textures.ground = ldx.texture_load("hookin_sprites/sokoban-pack/Ground/ground_01.png")
	g_textures.wall = ldx.texture_load("hookin_sprites/sokoban-pack/Blocks/block_01.png")
	g_textures.crate_stone = ldx.texture_load("hookin_sprites/sokoban-pack/Blocks/block_02.png")
	g_textures.goal = ldx.texture_load("hookin_sprites/sokoban-pack/Environment/environment_10.png")
	g_textures.pit = ldx.texture_load("hookin_sprites/sokoban-pack/Environment/environment_06.png")

	// constructing map
	cell_tex_size := ldx.texture_get_size(g_textures.wall)
	cell_tex_size_f := v2{cast(f32)cell_tex_size.x, cast(f32)cell_tex_size.y}
	the_map : Map = {
		v2{50, WINDOW_HEIGHT - 50 - cast(f32)cell_tex_size.y},
		v2{1,1},
		cell_tex_size_f,
		{
			{.Wall, .Wall, .Wall, .Wall, .Wall, .Wall, .Wall},
			{.Wall, .Ground, .Ground, .Pit, .Ground, .CrateStone, .Wall},
			{.Wall, .Ground, .Ground, .Pit, .Goal, .Ground, .Wall},
			{.Wall, .PlayerSpawn, .Ground, .Pit, .Ground, .Ground, .Wall},
			{.Wall, .Wall, .Wall, .Wall, .Wall, .Wall, .Wall}
		}
	}

	for !ldx.window_should_close() {
		audio.update()
		kb := ldx.get_keyboard()
		if kb[sdl.Scancode.ESCAPE] == 1 do break
		ldx.window_clear(COLOR_BACKGROUND)
		if kb[sdl.Scancode.A] == 1 do char_pos.x -= CHARACTER_SPEED
		if kb[sdl.Scancode.D] == 1 do char_pos.x += CHARACTER_SPEED
		if kb[sdl.Scancode.W] == 1 do char_pos.y += CHARACTER_SPEED
		if kb[sdl.Scancode.S] == 1 do char_pos.y -= CHARACTER_SPEED

		// Rendering the map
		map_draw(the_map)

		ldx.draw_texture(g_textures.player, char_pos, {char_scale, char_scale})

		ldx.present()
	}

	ldx.window_cleanup()
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

			if cell == .Pit {
				ldx.draw_solid_rect(
					{
						the_map.pos.x + cast(f32)column_i * the_map.cell_tex_size.x * the_map.scale.x,
						the_map.pos.y - cast(f32)row_i * the_map.cell_tex_size.y * the_map.scale.y
					}, the_map.cell_tex_size * the_map.scale, COLOR_BLACK)
			}

			ldx.draw_texture(
				the_tex,
				{the_map.pos.x + cast(f32)column_i * the_map.cell_tex_size.x * the_map.scale.x, the_map.pos.y - cast(f32)row_i * the_map.cell_tex_size.y * the_map.scale.y},
				the_map.scale
			)

			if cell == .Goal {
				ldx.draw_texture(
					g_textures.goal,
					{the_map.pos.x + cast(f32)column_i * the_map.cell_tex_size.x * the_map.scale.x, the_map.pos.y - cast(f32)row_i * the_map.cell_tex_size.y * the_map.scale.y},
					the_map.scale
				)
			}
		}
	}
}
