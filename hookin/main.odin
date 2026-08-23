package hookin

import "core:fmt"
import sdl "vendor:sdl2"

// Importing rendering engine
import ldx "../src"

WINDOW_WIDTH :: 400
WINDOW_HEIGHT :: 400

main :: proc() {
	ldx.window_new("hookin", WINDOW_WIDTH, WINDOW_HEIGHT)

	// Loop
	for {
		kb := ldx.get_keyboard()

		// quitting app
		if kb[sdl.Scancode.ESCAPE] == 1 {
			break
		}

		ldx.window_clear({1,1,1,1})

		// position, scale, texture
		// choose pivot? maybe
		ldx.draw_sprite({2,2}, {100, 100})

		ldx.draw_sprite({2,200}, {100, 100})

		ldx.present()
	}

	fmt.println("quitting...")

	ldx.window_cleanup()
}
