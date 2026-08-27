# LucyDX12

A from-scratch **DirectX 12** codebase written in **Odin**. One repository, three things built on
top of a single shared D3D12 foundation:

| Part | Lives in | What it is |
|---|---|---|
| **3D showcase** | `src/main.odin` (+ `camera.odin`, `gltf_processing.odin`, …) | A deferred PBR renderer / engine demo: glTF scenes, shadows, post-processing, ImGui tooling. This is the original project. |
| **lucy2d** | `src/lucy2d.odin` | A small immediate-mode **2D sprite API** layered on the same DX12 core. Intended to be consumed as a library. |
| **hookin** | `hookin/main.odin` | A handmade 2D puzzle game, the first client of lucy2d. Imports the root package as its renderer. |

---

## 🚫 Consultation only — do not edit

**Do not make any edits to files in this repository until Lucy explicitly tells you to.**
Lucy works with Claude here for consultation: analysis, explanation, design discussion, and
answering questions from the source that already exists. Investigate, reason, and report back.
Wait for an explicit instruction before writing or changing any file (this note being the one
exception that established the rule).

---

## ⚠️ Package layout (read this first)

**Everything under `src/` is a single Odin package named `main`.** That includes
`src/main.odin` (the 3D showcase's `main :: proc`) *and* `src/lucy2d.odin` (the 2D API) *and* all
the DX12 helpers. `hookin` is a separate package that imports it:

```odin
import ldx "../src"   // hookin/main.odin
```

Consequences to keep in mind when editing:

- lucy2d and the 3D showcase **share the global namespace**. `v2`/`v4`/`dxm`, `check`, `Texture`,
  `PSO`, `texture_create`, `structured_buffer_create` etc. are common to both.
- They do **not** share global state: the 3D showcase uses `g_dx_context` (`Context`), lucy2d uses
  `g_lct` (`Lucy2DContext`). Both sit on top of `g_dx_core` (`DX_Core`), which is the shared device,
  queue, command list, swapchain and descriptor heaps.
- Some identifiers are deliberately duplicated per-side (e.g. each side has its own
  `create_root_signatures`, `GeneralConstants` and `PSOName`) — check which one you are looking at.
- Splitting the engine into a real library package, separate from the showcase app, is a known open
  task (see the top of `notes.md`).

---

## 🛠️ Tech Stack

- **Language**: [Odin](https://odin-lang.org/)
- **Graphics API**: Direct3D 12 (`vendor:directx/d3d12`, `vendor:directx/dxgi`, `vendor:directx/dxc`)
- **Memory**: D3D12 Memory Allocator (`libs/odin-d3d12ma`) + virtual arenas (`core:mem/virtual`)
- **Platform / Windowing**: SDL2 (`vendor:sdl2`)
- **UI**: Dear ImGui (`libs/odin-imgui`, SDL2 + DX12 backends) — 3D showcase only
- **Assets**: `cgltf`, `stb/image`, `meshopt` (`third_party/meshopt`), `texconv.exe` for BC7
- **Profiling / debugging**: Spall (`core:prof/spall`, behind `-define:PROFILE=true`), RAD Debugger
- **Shader compiler**: DXC at runtime (`dxcompiler.dll`, `dxil.dll`)

---

## 🚀 Build & Run

```cmd
:: assets for the 3D showcase (Sponza, teapot, …)
dl_scenes.bat

:: 3D showcase
odin run src
odin build src -debug -out:lucydx12.exe -linker:radlink -vet-packages:main,sluggish_generator -vet-unused-variables -vet-shadowing -vet-using-stmt -vet-cast
odin build src -o:speed -out:lucydx12.exe -linker:radlink -vet-shadowing

:: the game (builds src as a library package)
odin build hookin -debug -out:hookin.exe -linker:radlink -keep-executable -vet

:: font data generator (separate package, dynamic lib)
odin build src/sluggish_generator -debug -build-mode:dynamic -linker:radlink
```

The canonical task list lives in `.zed/tasks.json` (build + `raddbg` launch combos). Note both
executables and the shaders are loaded with **paths relative to the repo root**, so always run from
the repo root, never from inside `src/` or `hookin/`.

---

## 📁 Repository Structure

```
.
├── src/                                  # package `main` — DX12 core + 3D showcase + lucy2d
│   ├── main.odin                         # 3D showcase: entry point, frame flow, passes, ImGui UI
│   ├── lucy2d.odin                       # 2D API: window/sprite/texture calls used by games
│   ├── dx_helpers.odin                   # the bulk of the engine: DX_Core, uber heaps, PSOs, DXC,
│   │                                     #   textures, buffers, DDS, texture cache, odin→HLSL gen
│   ├── dx_upload.odin                    # async upload thread + copy queue
│   ├── gltf_processing.odin              # glTF/GLB parsing, materials, meshopt optimization
│   ├── camera.odin                       # FPS / fly / orbit cameras
│   ├── dx_matrix_math.odin               # reverse-Z / z0 projection matrices
│   ├── descriptor_heap_allocator.odin    # free-list allocator (used by the ImGui backend)
│   ├── sluggish_generator/               # separate package: SDF-ish font data generator (DLL)
│   └── shaders/
│       ├── geometry.hlsl                 # GBuffer pass
│       ├── lighting.hlsl                 # deferred PBR + shadow lookup
│       ├── shadowmap.hlsl                # directional depth pass
│       ├── post_process.hlsl             # compute post-process / tonemap
│       ├── FXAA3_11.hlsl                 # FXAA 3.11
│       ├── ui.hlsl                       # gizmos / font / UI
│       ├── quads.hlsl                    # lucy2d sprite pass
│       ├── shader_common.hlsl            # shared bindings/helpers
│       └── gen/structs.gen.hlsl          # GENERATED from Odin at startup — never edit by hand
├── hookin/main.odin                      # the game
├── hookin_sprites/                       # game art (PNG)
├── libs/                                 # odin-imgui, odin-d3d12ma
├── models/                               # glTF assets (gitignored, fetched by dl_scenes.bat)
├── cache/                                # BC7 .dds texture cache, keyed by hashed source path
├── config.json                           # persisted 3D showcase settings (camera, lights, AA)
├── notes.md                              # roadmap, active tasks, refactor plans
└── .zed/tasks.json                       # build/run/debug tasks
```

---

## 🏗️ Shared DX12 Foundation (`dx_helpers.odin`)

Both the 3D showcase and lucy2d are built from these pieces.

- **`DX_Core` / `g_dx_core`** — device, direct queue, command allocator + list, swapchain, fences,
  D3D12MA allocator, and the uber descriptor heaps. Created by `init_dx(pool, window, width, height)`.
- **Uber descriptor heaps** — one big shader-visible `CBV_SRV_UAV` heap (up to ~1,000,000
  descriptors), plus RTV and DSV heaps. `create_srv` / `create_uav` / `create_cbv` / `create_rtv` /
  `create_dsv` bump-allocate a slot and return its **index**.
- **Bindless** — root signatures set `.CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED`, and shaders fetch
  resources with `ResourceDescriptorHeap[index]` (SM 6.6). The only root parameter is a single
  32-bit root constant carrying the index of a "general constants" CBV; every other index is read
  out of that constant buffer. Adding a resource to a shader usually means adding an index field to
  a constants struct, not touching the root signature.
- **`PSO` + `pso_create` / `pso_compute_create`** — takes a `.hlsl` path and a `PSOParameters`
  (blend, cull, depth, fill mode, RTV formats, vertex input `typeid`) and builds the pipeline state.
  Vertex input layouts are derived from Odin structs via RTTI (`get_dx_vertex_input`). Each PSO owns
  a `render_proc` that records that pass.
- **Shader hot-reload** — `pso_hotswap_watch` polls the `.hlsl` file's mtime, recompiles with DXC in
  process, and swaps the pipeline state without restarting.
- **`DXResourcePool`** — a `[dynamic]^dx.IUnknown` used as an explicit cleanup list. Two standing
  pools: *longterm* and *resizing* (freed and rebuilt on window resize).
- **Upload thread** (`dx_upload.odin`) — `dx_upload_trigger` / `dx_upload_texture_trigger` memcpy
  into mapped staging memory and issue copies on a dedicated copy queue, synchronized with a fence.
- **Texture pipeline** — `texture_cache_query` shells out to `texconv.exe` to transcode a source
  image into a mipmapped **BC7 .dds** under `cache/` (filename = hash of the source path), then
  `parse_dds_file` + `texture_create` upload it and allocate an SRV index.

---

## 🎬 The 3D Showcase (`src/main.odin`)

Deferred PBR renderer. Per frame, `render()` runs the PSOs in order:

1. **Shadowmap** — 2048² D32 directional depth map.
2. **GBuffer** — `geometry.hlsl` writes Albedo (RGBA8), Normal (packed), AO/Roughness/Metallic, and
   depth. Meshes are drawn from one big vertex/index buffer per scene; per-draw `mesh_index` and
   `material_index` come through `DrawConstants`.
3. **Lighting** — full-screen pass reading the GBuffer bindlessly, PBR with directional + point
   lights from a `Light` structured buffer, plus shadow lookup.
4. **Post-process** — compute pass (tonemap, FXAA options in `AAOptions`).
5. **ImGui** — settings, light editing, scene switching; state persists to `config.json`.

Scenes are loaded from glTF/GLB (`scene_from_gltf`), optimized with meshopt, and uploaded
asynchronously; `g_scenes` holds a small fixed set with a `SceneStatus` state machine.

---

## 🟦 lucy2d (`src/lucy2d.odin`)

A deliberately tiny immediate-mode 2D API. The whole public surface:

```odin
window_new(name: string, width, height: int)
window_should_close() -> bool
window_clear(color: Color)
texture_load(path: string) -> Texture
draw_sprite(sprite: Sprite)
present()
get_keyboard() -> []u8
window_cleanup()
```

How it works:

- `window_new` creates the SDL window, calls `init_dx`, builds the one standard root signature, the
  `GeneralConstants` CBV, the sprite structured buffer, and the single `.Quad` PSO
  (`src/shaders/quads.hlsl`).
- `draw_sprite` just appends to `g_lct.sprites_to_render` (max `SPRITE_MAX_COUNT`).
- `present` clears, runs every PSO's `render_proc`, then `frame_end` resets frame state and pumps
  SDL events.
- The quad pass uploads the sprite slice into an UPLOAD-heap structured buffer and issues **one**
  `DrawInstanced(4, sprite_count)` with a triangle strip; the vertex shader builds each corner from
  `SV_VertexID` and reads the sprite from the buffer with `SV_InstanceID`.
- **Coordinate space is pixels, origin bottom-left, +Y up.** The VS maps to NDC with
  `pos * inv_screen * 2 - 1`.
- Three static samplers are always available: `s0` anisotropic, `s1` point/nearest (for pixel art),
  `s2` linear. The quad shader uses `s1`.
- `Sprite.tex_idx` is a bindless SRV index; `0` means "untextured, use `color`".

---

## 🎮 hookin (`hookin/main.odin`)

The game, and the reason lucy2d exists. A ~50 line loop: WASD moves the player sprite, colliding
with the food respawns it. It is the reference for what lucy2d's API should feel like — if
something is awkward to write in `hookin/main.odin`, that is a bug in lucy2d, not in the game.

---

## 🔗 CPU ↔ GPU struct layout (important, and currently only half-solved)

Odin structs that get uploaded to the GPU must match their HLSL counterparts **exactly**.

**The generator.** `convert_struct_odin_to_hlsl` (`dx_helpers.odin:1840`) walks Odin RTTI and emits
HLSL structs/enums. `init_dx_user` runs it over `TYPES_FOR_HLSL` (`main.odin:34`) at startup and
writes `src/shaders/gen/structs.gen.hlsl`, which the 3D shaders `#include`. If you add a
GPU-visible struct on the 3D side, add its `typeid` to `TYPES_FOR_HLSL` rather than hand-writing
HLSL. The generated file is gitignored (`*.gen.hlsl`) and regenerated every run.

**The rules the two languages actually follow:**

- HLSL **structured buffers** use plain natural alignment: scalars align to their size, *vectors
  align to their component type* (a `float4` is 4-byte aligned, **not** 16), structs align to the
  max of their members, and DXC treats structs as packed. This matches Odin's own rules for
  `[N]f32` closely.
- HLSL **constant buffers** use the legacy rules instead: 16-byte rows, no vector may straddle a
  row, and every array element is padded to 16. Odin has no equivalent, so cbuffer structs must be
  padded by hand.
- Traps: Odin `bool` is 1 byte but HLSL `bool` is 4 (use `b32`); `float3x3` is 36 bytes packed but
  44/48 in a cbuffer (prefer `float4x4`).

**Do not put `#align(16)` on a struct that backs a structured buffer.** It rounds the Odin size up
to a multiple of 16, but DXC will never do the same to the HLSL struct, so `StructureByteStride`
stops matching the shader's element size — which is invalid, and indexes differently on different
GPU vendors (AMD uses the descriptor stride, others bake in the shader's). Pad explicitly with real
fields present on both sides, and back it with `#assert(size_of(T) == …)`.

**Known open issues:**

- lucy2d's `Sprite` is hand-written in `quads.hlsl` (it is not in `TYPES_FOR_HLSL`) and, since
  `tex_idx` was added, is **48 bytes in Odin vs 36 in HLSL** because of its `#align(16)`.
- The generator emits no padding at all, so it does not yet guarantee agreement on its own — it
  only saves you from typos. Making it emit explicit `_padN` fields from `offset_of`/`size_of`, and
  pointing lucy2d at it, is the real fix.

---

## ⚙️ Conventions

- **Naming**: procedures `snake_case` (`uber_heap_create`, `scene_from_gltf`); types `PascalCase`
  (`UberDescriptorHeap`, `GBufferUnit`); constants `SCREAMING_SNAKE_CASE`. Procedures are grouped
  by noun-first prefix (`texture_*`, `swapchain_*`, `pso_*`, `window_*`) — follow the prefix of the
  thing you are extending.
- **Globals** are `g_`-prefixed (`g_dx_core`, `g_dx_context`, `g_lct`, `g_scenes`).
- **Every D3D12 call that returns an `HRESULT`** goes through `check(hr, "message")`.
- **Memory is explicit**: allocate from a virtual arena (`arena_new`) or register the resource in a
  `DXResourcePool`; avoid implicit global allocations. Use `TEMP_GUARD()` for scratch work.
- **Logging** via `lprintln` / `lprintfln`.
- The `-vet` flags in `ols.json` / `.zed/tasks.json` are on for real (unused variables, shadowing,
  `using`, casts) — code must build clean under them.
- `notes.md` is the working roadmap; check it before starting anything structural.
