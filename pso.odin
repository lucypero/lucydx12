package main

import "core:thread"
import "core:sync/chan"
import "core:mem/virtual"
import "core:debug/trace"
import "core:reflect"
import img "vendor:stb/image"
import "core:fmt"
import "core:mem"
import "core:slice"
import "core:os"
import "core:strings"
import "core:sys/windows"
import "core:time"
import dx "vendor:directx/d3d12"
import dxgi "vendor:directx/dxgi"
import sdl "vendor:sdl2"
import "core:c"
import "core:math"
import "core:math/linalg"
import "base:runtime"
import "core:math/rand"
import "core:sync"
import dxc "vendor:directx/dxc"
import "core:prof/spall"
import dxma "libs/odin-d3d12ma"
import sa "core:container/small_array"

import sg "src/sluggish_generator"

// imgui
import im "libs/odin-imgui"
// imgui sdl2 implementation
import "libs/odin-imgui/imgui_impl_sdl2"
// imgui dx12 implementation
import "libs/odin-imgui/imgui_impl_dx12"

// checks if it should rebuild a shader
// if it should then compiles the new shader and makes a new PSO with it
pso_hotswap_watch :: proc(
	pso: ^PSO
) {
	// watch for shader change
	game_dll_mod, game_dll_mod_err := os.last_write_time_by_name(pso.shader_filename)

	reload := false

	if game_dll_mod_err == os.ERROR_NONE && pso.last_write_time != game_dll_mod {
		pso.last_write_time = game_dll_mod
		reload = true
	}

	if reload {
		lprintln("Recompiling shader...")
		// handle releasing resources
		vs, ps, ok := compile_shader(g_dx_context.dxc_compiler, pso.shader_filename)
		if !ok {
			lprintln("Could not compile new shader!! check logs")
		} else {
			// create the new PSO to be swapped later
			pso.pso_swap = pso.pso_creation_proc(pso.root_signature, vs, ps)
			vs->Release()
			ps->Release()
		}
	}
}

pso_hotswap_init :: proc(pso : ^PSO) {
	game_dll_mod, game_dll_mod_err := os.last_write_time_by_name(pso.shader_filename)
	if game_dll_mod_err == os.ERROR_NONE {
		pso.last_write_time = game_dll_mod
	}
}

pso_hotswap_swap :: proc(pso: ^PSO) {
	if pso.pso_swap != nil {
		pso.pipeline_state->Release()
		pso.pipeline_state = pso.pso_swap
		// replace pointer from freeing queue
		pso_pointer := &g_resources_longterm[pso.pso_index]
		pso_pointer^ = pso.pipeline_state
		pso.pso_swap = nil
	}
}
