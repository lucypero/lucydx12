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
CHARACTER_SIZE :: 25
FOOD_SIZE :: 10

main :: proc() {
	ldx.window_new("hookin", WINDOW_WIDTH, WINDOW_HEIGHT)
	main_character_pos : v2 = {100, 100}
	food_pos : v2 = {100, 200}

	ldx.texture_load("hookin_sprites/player_01.png")

	for !ldx.window_should_close() {
		kb := ldx.get_keyboard()
		if kb[sdl.Scancode.ESCAPE] == 1 do break
		ldx.window_clear(COLOR_BACKGROUND)
		if kb[sdl.Scancode.A] == 1 do main_character_pos.x -= CHARACTER_SPEED
		if kb[sdl.Scancode.D] == 1 do main_character_pos.x += CHARACTER_SPEED
		if kb[sdl.Scancode.W] == 1 do main_character_pos.y += CHARACTER_SPEED
		if kb[sdl.Scancode.S] == 1 do main_character_pos.y -= CHARACTER_SPEED
		if is_close(main_character_pos + CHARACTER_SIZE / 2, food_pos + FOOD_SIZE / 2) {
			// spawn the food somewhere else
			food_pos = {rand.float32_range(0 + 50, WINDOW_WIDTH - 50), rand.float32_range(0 + 50, WINDOW_HEIGHT - 50)}
		}
		ldx.draw_sprite(ldx.Sprite{main_character_pos, {CHARACTER_SIZE, CHARACTER_SIZE}, COLOR_CHARACTER})
		ldx.draw_sprite(ldx.Sprite{food_pos, {FOOD_SIZE, FOOD_SIZE}, COLOR_FOOD})
		ldx.present()
	}

	ldx.window_cleanup()
}

is_close :: proc(a, b: v2) -> bool {
	return linalg.length(b - a) < CHARACTER_SIZE
}
