package hookin

import "core:math"
// Collision Box

Face :: enum {Left, Right, Top, Bottom}
Faces :: bit_set[Face; u8]

Box :: struct {
	pos : v2, // position of top left of the box
	size: v2, // dimensions
	vel: v2,  // velocity
	hittable_faces: Faces
}


box_sweep :: proc(b1, b2 :Box) -> (collision_time: f32, collision_normal: v2) {
	inv_entry : v2
	inv_exit : v2

	// find the distance between the objects on the near and far sides for both x and y 
	if b1.vel.x > 0.0 {
		inv_entry.x = b2.pos.x - (b1.pos.x + b1.size.x)
		inv_exit.x = (b2.pos.x + b2.size.x) - b1.pos.x
	} else {
		inv_entry.x = (b2.pos.x + b2.size.x) - b1.pos.x
		inv_exit.x = b2.pos.x - (b1.pos.x + b1.size.x)
	} 

	if b1.vel.y > 0.0 {
		inv_entry.y = (b2.pos.y - b2.size.y) - b1.pos.y
		inv_exit.y = b2.pos.y - (b1.pos.y - b1.size.y)
	} else {
		inv_entry.y = b2.pos.y - (b1.pos.y - b1.size.y)
		inv_exit.y = (b2.pos.y - b2.size.y) - b1.pos.y
	}

	// find time of collision and time of leaving for each axis (if statement is to prevent divide by zero) 
	entry, exit: v2

	if (b1.vel.x == 0.0) {
		// if there's no overlap in X, there is no colission
		if (b1.pos.x + b1.size.x <= b2.pos.x) || (b1.pos.x >= b2.pos.x + b2.size.x) {
			return 1.0, {0, 0}
		}
		entry.x = math.inf_f32(-1)
		exit.x = math.inf_f32(1)
	} else {
		entry.x = inv_entry.x / b1.vel.x; 
		exit.x = inv_exit.x / b1.vel.x; 
	} 

	if (b1.vel.y == 0.0) {
		// if there's no overlap in Y, there is no colission
		if (b1.pos.y <= b2.pos.y - b2.size.y) || (b1.pos.y - b1.size.y >= b2.pos.y) {
			return 1.0, {0, 0}
		}
		entry.y = math.inf_f32(-1)
		exit.y = math.inf_f32(1)
	} else {
		entry.y = inv_entry.y / b1.vel.y; 
		exit.y = inv_exit.y / b1.vel.y; 
	}

	// find the earliest/latest times of collisionfloat 
	entry_time := max(entry.x, entry.y)
	exit_time := min(exit.x, exit.y)

	// if there was no collision
	if (entry_time >= exit_time) || entry_time < 0 || entry_time > 1 {
		collision_normal = {0, 0}
		collision_time = 1.0
	} else {  // if there was a collision
		// The normal always opposes the velocity on the axis we entered through.
		if (entry.x > entry.y) {
			collision_normal = b1.vel.x > 0 ? {-1,0} : {1, 0}
		} else {
			collision_normal = b1.vel.y > 0 ? {0,-1} : {0,1}
		} 
		collision_time = entry_time
	}

	return
}

// Collision: Testing player against boxes
move_and_slide :: proc(moving_box: ^Box, static_boxes: []Box) -> (_box_i: int, _col_normal: v2, _did_hit: bool){
	outer: for _ in 0..<4 {

		col_normal: v2
		col_time := math.inf_f32(1)

		if moving_box.vel == {0,0} do break

		for static_box, box_i in static_boxes {
			box_broadphase := box_get_broadphase(moving_box^)
			if !box_does_hit_box(box_broadphase, static_box) do continue
			col_time_i, col_normal_i := box_sweep(moving_box^, static_box)

			// Discarding collisions on internal edges
			is_discarded : bool

			if col_normal_i == {1,0} && .Right not_in static_box.hittable_faces do is_discarded = true
			if col_normal_i == {-1,0} && .Left not_in static_box.hittable_faces do is_discarded = true
			if col_normal_i == {0,1} && .Top not_in static_box.hittable_faces do is_discarded = true
			if col_normal_i == {0,-1} && .Bottom not_in static_box.hittable_faces do is_discarded = true

			if is_discarded {
				continue
			}

			if col_time_i < col_time {
				col_normal = col_normal_i
				col_time = col_time_i

				// Saving box we hit for return value
				_box_i = box_i
			}
		}

		// there was collision
		if col_time < 1 {

			// Setting return values
			_did_hit = true
			_col_normal = col_normal

			// moving the box right next to the obstacle
			moving_box.pos += moving_box.vel * col_time

			// Sliding
			remaining_time := 1.0 - col_time
			dotprod := (moving_box.vel.x * col_normal.y + moving_box.vel.y * col_normal.x) * remaining_time
			next_vel := v2{dotprod * col_normal.y, dotprod * col_normal.x}
			// Setting the box's velocity as the slide velocity, and sweeping again before committing to a move.
			moving_box.vel = next_vel
		} else { // no collision. skip all other collision tests
			break outer
		}
	}

	moving_box.pos += moving_box.vel
	return
}

box_get_broadphase :: proc(b: Box) -> (broadphase_box: Box) {
	broadphase_box.pos.x = b.vel.x > 0 ? b.pos.x : b.pos.x + b.vel.x
	broadphase_box.pos.y = b.vel.y > 0 ? b.pos.y + b.vel.y : b.pos.y
	broadphase_box.size.x = b.vel.x > 0 ? b.vel.x + b.size.x : b.size.x - b.vel.x  
	broadphase_box.size.y = b.vel.y > 0 ? b.vel.y + b.size.y : b.size.y - b.vel.y  
	return
}

box_does_hit_box :: proc(b1,b2: Box) -> bool {
	return !((b1.pos.x + b1.size.x < b2.pos.x) ||
		(b1.pos.x > b2.pos.x + b2.size.x) ||
		(b1.pos.y < b2.pos.y - b2.size.y) ||
		(b1.pos.y - b1.size.y > b2.pos.y))
}

BoxCollisionRecord :: struct {
	map_boxes: []Box,
	coords: []Coord,
	box_i: int, //index into map_boxes and coords of box that you hit,
	col_normal: v2,
	did_hit: bool,
}
