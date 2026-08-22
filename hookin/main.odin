package hookin

import "core:fmt"

// Importing rendering engine
import ldx "../src"

main :: proc() {
	fmt.println("hello hookin")

	// using camera?
	cam : ldx.Camera

	fmt.println(cam)
}
