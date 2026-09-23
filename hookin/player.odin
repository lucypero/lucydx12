package hookin

import l2d "../lucy2d"

import linalg "core:math/linalg"

CHARACTER_MAX_SPEED :: 2
CHARACTER_SPEED_RATE :: 0.5
CHARACTER_TURN_RATE :: 1.0
CHARACTER_SIZE :: 64
CHARACTER_DECEL_SPEED :: 0.2

Player :: struct {
	using box: Box,// box for collision
	texture_size: v2,
	texture_offset: v2,
	current_coord: Coord,
	vis: Visualization,
	last_input_vel: v2i, // determines where the player is facing
}

player_init :: proc(player: ^Player) {
	char_tex_size_i := l2d.texture_get_size(g_entity_defs[EntityType.Player].tex_id)
	player.texture_size = l2d.v2i_to_v2(char_tex_size_i)
	player.box.size = player.texture_size * 0.7
	player.texture_offset = v2{-10,10}
}

player_update :: proc(player: ^Player) -> Maybe(BoxCollisionRecord) {
	vel : v2

	if !g_input_disabled && l2d.key_is_down(.A) {
		vel.x = -1 
		player.last_input_vel = {-1, 0}
	}
	if !g_input_disabled && l2d.key_is_down(.D) {
		vel.x = 1
		player.last_input_vel = {1, 0}
	}
	if !g_input_disabled && l2d.key_is_down(.W) {
		vel.y = 1
		player.last_input_vel = {0, -1}
	} 
	if !g_input_disabled && l2d.key_is_down(.S) {
		vel.y = -1
		player.last_input_vel = {0, 1}
	}


	if vel != {0,0} {
		input_dir := linalg.normalize(vel)

		accel := v2{CHARACTER_SPEED_RATE, CHARACTER_SPEED_RATE}

		// apply much faster acceleration rate when moving in opposite
		// direction of current velocity
		if input_dir.x != 0 && (input_dir.x * player.vel.x < 0) {
			accel.x *= CHARACTER_TURN_RATE
		}
		if input_dir.y != 0 && (input_dir.y * player.vel.y < 0) {
			accel.y *= CHARACTER_TURN_RATE
		}

		player.vel += input_dir * accel
		player.vel = cap_velocity(player.vel, CHARACTER_MAX_SPEED)

		map_boxes, ids := generate_collisions(g_map)
		box_i, col_normal, did_hit := move_and_slide(&player.box, map_boxes[:])
		bcr := BoxCollisionRecord { map_boxes, ids, box_i, col_normal, did_hit }
		return bcr
	} else {
		// interactions like box moves should be ignored if a collision happens with
		// excess velocity without continued player input
		player.vel = decelerate(player.vel, CHARACTER_DECEL_SPEED)
		map_boxes, ids := generate_collisions(g_map)
		box_i, col_normal, did_hit := move_and_slide(&player.box, map_boxes[:])
		return nil
	}
}


cap_velocity :: proc(vel: v2, max_speed: f32) -> v2 {
	sq_len := linalg.length2(vel)
	max_sq := max_speed * max_speed

	if sq_len > max_sq {
		return (vel / linalg.sqrt(sq_len)) * max_speed
	}

	return vel
}

decelerate :: proc(vel: v2, decel_rate: f32) -> v2 {
	speed := linalg.length(vel)
	drop := decel_rate

	if speed <= drop || speed == 0 {
		return {0, 0}
	}

	new_speed := speed - drop
	return vel * (new_speed / speed)
}
