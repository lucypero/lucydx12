package hookin

import l2d "../lucy2d"

HOOKIN_ASSETS_DIR :: "hookin/assets"
IS_FLOOR :: true
IS_SOLID :: true

g_entities: [len(EntityType)]EntityDef

EntityType :: enum
{
    Nothing, Ground, Player, Crate, MetalCrate, PlayerSpawn, Goal, Wall, Pit, Test
}

EntityDef :: struct
{
    type : EntityType,
    tex_id: int,
    is_solid, is_floor: bool,
}

Entity :: struct
{
    et: EntityType,
    gen: i32,
    coord: Coord,
    vis: Visualization,
}

register_tile :: proc(type: EntityType, texture_path: string, is_solid, is_floor: bool)
{
    g_entities[type] = EntityDef{type, l2d.texture_load(texture_path), is_solid, is_floor}
}

init_tile_registry :: proc()
{
    // Solids
    register_tile(.Player, HOOKIN_ASSETS_DIR+"/sokoban-pack/Player/player_01.png", IS_SOLID, !IS_FLOOR)
    register_tile(.Crate, HOOKIN_ASSETS_DIR+"/sokoban-pack/Crates/crate_07.png", IS_SOLID, !IS_FLOOR)
    register_tile(.Wall, HOOKIN_ASSETS_DIR+"/sokoban-pack/Blocks/block_01.png", IS_SOLID, !IS_FLOOR)
    register_tile(.MetalCrate, HOOKIN_ASSETS_DIR+"/sokoban-pack/Crates/crate_04.png", IS_SOLID, !IS_FLOOR)
    
    // TEST EXAMPLE
    register_tile(.Test, HOOKIN_ASSETS_DIR+"/sokoban-pack/Blocks/block_05.png", IS_SOLID, !IS_FLOOR)

    // Floors
    register_tile(.Ground, HOOKIN_ASSETS_DIR+"/sokoban-pack/Ground/ground_01.png", !IS_SOLID, IS_FLOOR)
    register_tile(.Pit, HOOKIN_ASSETS_DIR+"/sokoban-pack/Environment/environment_06.png", !IS_SOLID, IS_FLOOR)
    register_tile(.Goal, HOOKIN_ASSETS_DIR+"/sokoban-pack/Environment/environment_10.png", !IS_SOLID, IS_FLOOR)
    register_tile(.PlayerSpawn, HOOKIN_ASSETS_DIR+"/d42.png", !IS_SOLID, IS_FLOOR)
}