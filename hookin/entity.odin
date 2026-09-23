package hookin

import l2d "../lucy2d"

HOOKIN_ASSETS_DIR :: "hookin/assets"

g_entity_defs: [EntityType]EntityDef

EntityType :: enum {
	Nothing, Ground, Player, Crate, MetalCrate, PlayerSpawn, Goal, Wall, Pit, Test
}

EntityFlag :: enum {Solid, Floor, Invisible}
EntityFlag_Set :: bit_set[EntityFlag]

EntityDef :: struct {
	type : EntityType,
	tex_id: int,
	flags: EntityFlag_Set
}

Entity :: struct {
	et: EntityType,
	gen: i32,
	coord: Coord,
	vis: Visualization,
}

register_tile :: proc(type: EntityType, texture_path: string, flags: EntityFlag_Set) {
	g_entity_defs[type] = EntityDef{type, l2d.texture_load(texture_path), flags}
}

init_tile_registry :: proc() {
	// Solids
	register_tile(.Player, HOOKIN_ASSETS_DIR+"/sokoban-pack/Player/player_01.png", {.Solid})
	register_tile(.Crate, HOOKIN_ASSETS_DIR+"/sokoban-pack/Crates/crate_07.png", {.Solid})
	register_tile(.Wall, HOOKIN_ASSETS_DIR+"/sokoban-pack/Blocks/block_01.png", {.Solid})
	register_tile(.MetalCrate, HOOKIN_ASSETS_DIR+"/sokoban-pack/Crates/crate_04.png", {.Solid})

	// TEST EXAMPLE
	register_tile(.Test, HOOKIN_ASSETS_DIR+"/sokoban-pack/Blocks/block_05.png", {.Solid})

	// Floors
	register_tile(.Ground, HOOKIN_ASSETS_DIR+"/sokoban-pack/Ground/ground_01.png", {.Floor})
	register_tile(.Pit, HOOKIN_ASSETS_DIR+"/sokoban-pack/Environment/environment_06.png", {.Floor})
	register_tile(.Goal, HOOKIN_ASSETS_DIR+"/sokoban-pack/Environment/environment_10.png", {})
	register_tile(.PlayerSpawn, HOOKIN_ASSETS_DIR+"/d42.png", {.Invisible})
}

entity_is_solid :: proc(e: EntityType) -> bool {
	return .Solid in g_entity_defs[e].flags
}

entity_is_floor :: proc(e:EntityType) -> bool {
	return .Floor in g_entity_defs[e].flags
}

entity_is_invisible :: proc(e:EntityType) -> bool {
	return .Invisible in g_entity_defs[e].flags
}
