package main

import "core:math"
import sdl "vendor:sdl2"
import "core:math/linalg"
// modes

// fps mode
// static mode

cur_cam : Camera
cur_cam_mode : CameraMode

CameraMode :: enum {
	Static,
	FPS,
}

Camera :: struct {
	pos : v3,
	pitch: f32, // rad
	yaw: f32 // rad
}

camera_init :: proc() -> Camera {
	// do something here
	
	// set this to true when u are in FPS mode
	
	cur_cam_mode = .FPS
	camera_toggle_mode()
	
	
	return Camera{
		pos = {0, 1, -1},
		yaw = -90
	}
}

camera_toggle_mode :: proc() {
	if cur_cam_mode == .Static {
		cur_cam_mode = .FPS
		sdl.SetRelativeMouseMode(true)
	} else {
		cur_cam_mode = .Static
		sdl.SetRelativeMouseMode(false)
	}
}

camera_static_tick :: proc(buttons: u32, keyboard: []u8) {
	if keyboard[sdl.Scancode.F] == 1 {
		camera_toggle_mode()
		return
	}
}

camera_tick :: proc(keyboard: []u8) {
	// todo call this one step up the call stack
	dx, dy: i32
	buttons : u32 = sdl.GetRelativeMouseState(&dx, &dy)
	switch cur_cam_mode {
	case .Static:
		camera_static_tick(buttons, keyboard)
	case .FPS:
		camera_fps_tick(buttons, dx, dy, keyboard)
	}
}

camera_fps_tick :: proc(buttons: u32, dx, dy: i32, keyboard: []u8) {
	// read input. rotate camera.
	sens :: 0.005
	
	cur_cam.yaw -= f32(dx) * sens
	cur_cam.pitch += f32(dy) * sens
	
	
	pitch_clamp :: 0.24
	cur_cam.pitch = math.clamp(cur_cam.pitch, math.TAU * -pitch_clamp, math.TAU * pitch_clamp)
	
	// keyboard controls (moving the cam position)
	
	cam_speed :: 0.1
	
	cam_dir := cam_get_direction(cur_cam)
	
	// assuming world up is 0 1 0 
	right := linalg.vector_cross3(v3{0,1,0}, cam_dir)
	// this might be "left" because "cam_dir" might be "back"
	right = linalg.normalize(right)
	
	if keyboard[sdl.Scancode.W] == 1 {
		// go forward
		cur_cam.pos += cam_dir * -cam_speed
	}
	
	if keyboard[sdl.Scancode.S] == 1 {
		cur_cam.pos += cam_dir * cam_speed
	}
	
	if keyboard[sdl.Scancode.A] == 1 {
		cur_cam.pos += right * cam_speed
	}
	
	if keyboard[sdl.Scancode.D] == 1 {
		cur_cam.pos += right * -cam_speed
	}
	
	if keyboard[sdl.Scancode.Q] == 1 {
		cur_cam.pos += v3{0, 1, 0} * -cam_speed
	}
	
	if keyboard[sdl.Scancode.E] == 1 {
		cur_cam.pos += v3{0, 1, 0} * cam_speed
	}
	
	// fmt.printfln("%v", cam_dir)
	
	// if click, switch to static camera mode
	if (buttons & sdl.BUTTON_LMASK) == 1 {
		camera_toggle_mode()
		return
	}
}

// get direction vector (front) (where the camera is looking at)
// (this might actually be "back")
cam_get_direction :: proc(cam: Camera) -> v3 {
    // Offset yaw by -90 degrees (math.PI/2) so that 0 yaw faces 'forward' 
    // instead of 'right'
    yaw := cam.yaw
    pitch := cam.pitch

    direction: v3
    direction.x = math.cos(yaw) * math.cos(pitch)
    direction.y = math.sin(pitch)
    direction.z = math.sin(yaw) * math.cos(pitch)
    
    return linalg.normalize(direction)
}

get_view_projection :: proc(cam: Camera) -> (dxm, dxm) {
	
	fov_deg :: 90.0
	
    look_at := cam.pos + cam_get_direction(cam)

	view := linalg.matrix4_look_at_f32(cam.pos, look_at, {0, 1, 0}, true)

	fov := linalg.to_radians(f32(fov_deg))
	aspect := f32(wx) / f32(wy)
	// proj := linalg.matrix4_perspective_f32(fov, aspect, 0.1, 100, true)

	// this function is supposedly more correct
	// has correct depth values
	proj := matrix4_perspective_z0_f32(fov, aspect, 0.1, 100)

	return view, proj
	// return view * proj
}
