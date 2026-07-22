# LucyDX12

**LucyDX12** is a real-time DirectX 12 graphics engine built from scratch in **Odin**. It features a modern deferred PBR rendering pipeline, bindless descriptor management, runtime HLSL shader hot-reloading, glTF scene loading, FXAA post-processing, and ImGui integration.

---

## 🛠️ Tech Stack & Dependencies

- **Language**: [Odin](https://odin-lang.org/)
- **Graphics API**: Direct3D 12 (`vendor:directx/d3d12`, `vendor:directx/dxgi`, `vendor:directx/dxc`)
- **Memory Allocation**: D3D12 Memory Allocator (`libs/odin-d3d12ma`) & Custom Virtual Arena Allocators (`core:mem/virtual`)
- **Platform / Windowing**: SDL2 (`vendor:sdl2`)
- **UI**: Dear ImGui (`libs/odin-imgui` with SDL2 + DX12 backends)
- **Asset Processing**: `cgltf` (`vendor:cgltf`), `stb_image` (`vendor:stb/image`), `meshopt` (`third_party/meshopt`)
- **Profiling & Debugging**: Spall Profiler (`core:prof/spall`), RAD Debugger (`lucydx12.radproject`)
- **Shader Compiler**: DXC (`dxcompiler.dll`, `dxil.dll`)

---

## 🚀 Build & Run Commands

```cmd
# Download test scenes / models (Sponza, Teapot, etc.)
dl_scenes.bat

# Compile and run the engine
odin run src

# Compile release build with optimizations
odin build src -out:lucydx12.exe -o:speed
```

---

## 📁 Repository Structure

```
.
├── src/
│   ├── main.odin                         # Entry point, main render loop, frame flow, GBuffer & Lighting passes
│   ├── dx_helpers.odin                   # D3D12 helpers, PSOs, shader compilation (DXC), Uber Descriptor Heaps
│   ├── gltf_processing.odin              # glTF 2.0 / GLB scene parser, materials, mesh optimization
│   ├── dx_upload.odin                    # Asynchronous GPU upload queue for buffers and textures
│   ├── camera.odin                       # FPS, Fly, and Orbit camera controllers & projection matrices
│   ├── dx_matrix_math.odin               # Matrix transformation helpers
│   ├── descriptor_heap_allocator.odin    # Descriptor heap index management
│   └── shaders/
│       ├── geometry.hlsl                 # GBuffer generation pass (Albedo, Normal, AO/Roughness/Metallic)
│       ├── lighting.hlsl                 # Deferred PBR lighting pass & shadow integration
│       ├── shadowmap.hlsl                # Directional depth map render pass
│       ├── post_process.hlsl             # Post-processing composite & tonemapping
│       ├── FXAA3_11.hlsl                 # Fast Approximate Anti-Aliasing (FXAA 3.11)
│       ├── ui.hlsl                       # Sluggish font & UI shader
│       └── shader_common.hlsl            # Shared HLSL structures and bindings
├── libs/                                 # Third-party Odin library bindings (odin-imgui, odin-d3d12ma)
├── models/                               # 3D models and glTF sample assets
├── notes.md                              # Developer roadmap, active tasks, and architecture refactoring notes
├── README.md                             # Quickstart guide
├── lucydx12.radproject                   # RAD Debugger project workspace
└── ols.json / odinfmt.json               # Odin Language Server (OLS) and formatter configurations
```

---

## 🏗️ Architecture Overview

### 1. Deferred Rendering Pipeline
- **Geometry Pass (GBuffer)**: Renders scene geometry to GBuffer textures:
  - `Albedo`: Base color (RGB8U)
  - `Normal`: Surface normals (packed/encoded)
  - `AO_Rough_Metal`: Ambient Occlusion, Roughness, and Metallic parameters
  - `DepthStencil`: Depth target (D32_FLOAT or D24S8)
- **Lighting Pass**: Reads GBuffer textures via bindless descriptor indices in `lighting.hlsl` to perform PBR shading with directional and point light sources.
- **Shadow Pass**: Calculates directional shadow maps to populate light visibility.
- **Post-Processing & AA**: Applies FXAA anti-aliasing filter and blits the final render target to the swapchain.

### 2. Bindless / Uber Descriptor Heap
- Single large `IDescriptorHeap` (`CBV_SRV_UAV`) managing up to 1,000,000 descriptors.
- Constant buffers and texture SRVs pass index values into root constants or push parameters, allowing shaders to look up resources dynamic indexually without rebinding individual tables.

### 3. Shader Hot-Reloading
- Shaders are dynamically recompiled at runtime via `dxc` when `.hlsl` files change, replacing active PSOs without restarting the app.

---

## ⚙️ Coding Conventions & Odin Practices

- **Explicit Memory**: Prefer custom arenas (`virtual.arena_allocator`) or explicit cleanup pools (`DXResourcePool`) over implicit global allocations.
- **DirectX 12 API Calls**: Wrap creation calls with `check(hr, "error message")` from `dx_helpers.odin`.
- **Naming Conventions**:
  - Odin procedures: `snake_case` (e.g., `uber_heap_create`, `scene_from_gltf`).
  - Struct types: `PascalCase` (e.g., `UberDescriptorHeap`, `GBufferUnit`).
  - Constants & Enums: `SCREAMING_SNAKE_CASE` or `PascalCase` for enum values.
