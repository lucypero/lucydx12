package hookin

import "core:math/rand"
import "core:math/linalg"
import sdl "vendor:sdl2"

// Importing rendering engine
import ldx "../src"

v2 :: ldx.v2
v4 :: ldx.v4

WINDOW_WIDTH :: 400
WINDOW_HEIGHT :: 400
COLOR_BACKGROUND :: v4{0.773, 0.686, 0.643,1}
COLOR_CHARACTER :: v4{0.8, 0.494, 0.522, 1}
COLOR_FOOD :: v4{0.639, 0.427, 0.565, 1}
CHARACTER_SPEED :: 4
CHARACTER_SIZE :: 64
FOOD_SIZE :: 10

main :: proc() {
	ldx.window_new("hookin", WINDOW_WIDTH, WINDOW_HEIGHT)
	char_pos : v2 = {100, 100}
	char_scale : f32 = 0.5
	food_pos : v2 = {100, 200}

	tex_player := ldx.texture_load("hookin_sprites/player_01.png")
	tex_size := ldx.texture_get_size(tex_player)

	for !ldx.window_should_close() {
		kb := ldx.get_keyboard()
		if kb[sdl.Scancode.ESCAPE] == 1 do break
		ldx.window_clear(COLOR_BACKGROUND)
		if kb[sdl.Scancode.A] == 1 do char_pos.x -= CHARACTER_SPEED
		if kb[sdl.Scancode.D] == 1 do char_pos.x += CHARACTER_SPEED
		if kb[sdl.Scancode.W] == 1 do char_pos.y += CHARACTER_SPEED
		if kb[sdl.Scancode.S] == 1 do char_pos.y -= CHARACTER_SPEED

		char_size := v2{char_scale, char_scale} * v2{cast(f32)tex_size.x, cast(f32)tex_size.y}
		if linalg.length((food_pos + FOOD_SIZE / 2) - (char_pos + char_size / 2)) < linalg.length(char_size) / 2 {
			// spawn the food somewhere else
			food_pos = {rand.float32_range(0 + 50, WINDOW_WIDTH - 50), rand.float32_range(0 + 50, WINDOW_HEIGHT - 50)}
			char_scale += 0.1
		}

		ldx.draw_texture(tex_player, char_pos, {char_scale, char_scale})
		ldx.draw_solid_rect(food_pos, {FOOD_SIZE, FOOD_SIZE}, COLOR_FOOD)
		ldx.present()
	}

	ldx.window_cleanup()
}
