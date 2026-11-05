package lucydx

import "core:math"
import "core:math/linalg"

// Left-handed view matrix: `target` ends up on +Z, which is what matrix4_perspective_z0_f32 expects.
// (linalg.matrix4_look_at_f32 always puts the target on -Z.)
@(require_results)
matrix4_look_at_lh_f32 :: proc(eye, target, up: linalg.Vector3f32) -> linalg.Matrix4f32 {
	f := linalg.normalize(target - eye)
	s := linalg.normalize(linalg.vector_cross3(up, f))
	u := linalg.vector_cross3(f, s)

	return linalg.Matrix4f32 {
		s.x, s.y, s.z, -linalg.dot(s, eye),
		u.x, u.y, u.z, -linalg.dot(u, eye),
		f.x, f.y, f.z, -linalg.dot(f, eye),
		0,   0,   0,   1,
	}
}

@(require_results)
matrix4_perspective_z0_f32 :: proc "contextless" (
	fovy, aspect, near, far: f32,
) -> (
	m: linalg.Matrix4f32,
) #no_bounds_check {
	tan_half_fovy := math.tan(0.5 * fovy)
	m[0, 0] = 1 / (aspect * tan_half_fovy)
	m[1, 1] = 1 / (tan_half_fovy)
	m[3, 2] = +1

	m[2, 2] = far / (far - near)
	m[2, 3] = -(far * near) / (far - near)

	return
}

@(require_results)
matrix_ortho3d_z0_f32 :: proc "contextless" (
	left, right, bottom, top, near, far: f32,
) -> (
	m: linalg.Matrix4f32,
) #no_bounds_check {
	m[0, 0] = +2 / (right - left)
	m[1, 1] = +2 / (top - bottom)
	m[0, 3] = -(right + left) / (right - left)
	m[1, 3] = -(top + bottom) / (top - bottom)
	m[3, 3] = 1

	m[2, 2] = +1 / (far - near)
	m[2, 3] = -near / (far - near)

	m[2] = -m[2]

	return
}

// unused procs below
/*

@(require_results)
matrix4_infinite_perspective_z0_f32 :: proc "contextless" (
	fovy, aspect, near: f32,
) -> (
	m: linalg.Matrix4f32,
) #no_bounds_check {
	tan_half_fovy := math.tan(0.5 * fovy)
	m[0, 0] = 1 / (aspect * tan_half_fovy)
	m[1, 1] = 1 / (tan_half_fovy)
	m[2, 2] = +1
	m[3, 2] = +1

	m[2, 3] = -near

	m[2] = -m[2]

	return
}

*/
