# Odin ImGui

## Generated Dear ImGui bindings using dear_bindings

 - Generates bindings [Dear ImGui](https://github.com/ocornut/imgui), using [`dear_bindings`](https://github.com/dearimgui/dear_bindings)
 - Contains bindings for most of the Dear ImGui implementations
	- All backends which exist in `vendor:` have bindings
	- These include: `vulkan, sdl2, opengl3, sdlrenderer2, sdl3, sdlrenderer3, sdlgpu3, glfw, dx11, dx12, win32, osx, metal, wgpu, webgl`
 - Compiles bindings as well as any wanted backends
 - Tested on Windows, Linux, and Mac
 - Includes several examples which can be used as a reference
	- `GLFW + OpenGL, SDL2 + D3D11, SDL2 + Metal, SDL2 + OpenGL, SDL2 + SDL2 Renderer, SDL3 + SDL3 Renderer, SDL3 + SDL3 GPU, SDL2 + Vulkan, GLFW + WGPU, JS/GLFW + W(eb)GPU, JS + WebGL`

## Usage
 - If you don't want to configure and or build yourself, a prebuilt binary has been committed to the repository (currently windows only).
 - It has all backends listed in `build.py` enabled, which almost definitely more than you need. I strongly suggest building yourself with your wanted backends.

## Building

Building is entirely automated, using `build.py`. All platforms should work (not not: open an issue!), but currently the Mac backends are untested as I don't have a Mac (help wanted!)

 0. Dependencies
	- `git` and `python` must be in your path
	- Linux and OSX depend on `clang`, `ar`
	- Windows builds require that [`vcvarsall.bat`](https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170) has been executed
 1. Clone this repository.
	- Optionally configure build at the top of `build.py`
 2. Run `python build.py`
 3. Repository is importable. Copy into your project, or import directly.

## Configuring

Search for `@CONFIGURE` to see everything configurable.

### `wanted_backends`
This project allows you to compile ImGui backends alongside imgui itself, which is what Dear ImGui recommends you do.
Bindings have been written for a subset of the backends provided by ImGui
 - You can see if a backend is supported by checking the `backends` table in `build.py`.
 - If a backend is supported it means that:
	- Bindings have been written in `imgui_impl_xyz/`
	- It has been successfully compiled and run in one of the `examples/`
 - Some backends have external dependencies. These will automatically be cloned into `backend_deps` if necessary.
 - You can enable a backend by adding it to `wanted_backends`
 - You can enable backends not officially supported.

### `compile_debug`
If set to true, will compile with debug flags

### `build_wasm`
If set to true, will compile WASM object files

## Examples

There are some examples in `examples/`. They are runnable directly.

## Available backends

All backends which can be supported with only `vendor` have bindings now.
It seems likely to me that SDL3, and maybe Android will exist in vendor in the future, at which point I'll add support.

| Backend        | Has bindings | Has example | Comment                                                              |
|----------------|:------------:|:-----------:|----------------------------------------------------------------------|
| Allegro 5      |      No      |     No      | No odin bindings in vendor                                           |
| Android        |      No      |     No      | No odin bindings in vendor                                           |
| Directx 9      |      No      |     No      | No odin bindings in vendor                                           |
| Directx 10     |      No      |     No      | No odin bindings in vendor                                           |
| Directx 11     |     Yes      |     Yes     |                                                                      |
| Directx 12     |     Yes      |     No      | Bindings created, but not tested                                     |
| GLFW           |     Yes      |     Yes     |                                                                      |
| GLUT           |      No      |     No      | Obsolete. Likely will never be implemented.                          |
| Metal          |     Yes      |     Yes     |                                                                      |
| Null           |      Yes     |     yes     |                                                                      |
| OpenGL 2       |      No      |     No      |                                                                      |
| OpenGL 3       |     Yes      |     Yes     |                                                                      |
| OSX            |     Yes      |     No      |                                                                      |
| SDL 2          |     Yes      |     Yes     |                                                                      |
| SDL 3          |     Yes      |     Yes     |                                                                      |
| SDL 3 GPU      |     Yes      |     Yes     |                                                                      |
| SDL_Renderer 2 |     Yes      |     Yes     | Has example, but Odin vendor library lacks required version (2.0.18) |
| SDL_Renderer 3 |     Yes      |     Yes     |                                                                      |
| Vulkan         |     Yes      |     No      | Tested in my own engine, but no example yet due to size              |
| win32          |     Yes      |     No      | Bindings created, but not tested. Note: as of v1.91.5, this backend can no longer be compiled due to ImGui_ImplWin32_WndProcHandler and ImGui_ImplWin32_WndProcHandlerEx |
| JS             |     Yes      |     Yes     | Native Odin backend - Docking doesn't work for some reason           |
| WebGPU         |     Yes      |     Yes     | Native Odin backend                                                  |
| WebGL          |     Yes      |     Yes     | Native Odin backend - WebGL 2 only                                   |
