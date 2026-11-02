# LucyDX12

A Direct3D 12 renderer I'm building in [Odin](https://odin-lang.org/) to learn modern graphics programming.

<img width="2000" height="1000" alt="image" src="https://github.com/user-attachments/assets/94c0b5f5-3202-42d3-807a-2a219abac2a6" />

[Article about this project](https://lucypero.com/lucydx12.html)

## Features

- **Bindless**: one root signature for every PSO; shaders index resources directly from the descriptor heap.
- **Deferred rendering**: G-buffer pass (albedo, normals, AO + roughness), then a lighting pass.
- **Asset streaming**: a worker thread uploads scenes on the GPU copy queue, so scene switches don't stutter.
- **Shader interop through RTTI**: input layouts and HLSL structs are generated from Odin structs.
- **Shadow maps** with PCF filtering.
- **FXAA** in a compute pass.
- **glTF loading** with a cache of GPU-ready, mipmapped textures.
- **Dear ImGui** for live tweaking.

## To work on next:

- An Accurate PBR lighting model
- Character Animation
- Whatever is needed to support my next game (unannounced)

# Build + run

```cmd
dl_scenes.bat
odin run lucy3d
```

## Gallery

### Scene switch while the camera moves; assets stream in without stutter.

https://github.com/user-attachments/assets/df3caadb-f1ec-47eb-85ac-1e4af6b54708

### Shadow maps

https://github.com/user-attachments/assets/3dedebbc-ad13-444a-9ff0-554fcaeba689

### G-Buffer view in Renderdoc
<img width="942" height="227" alt="image" src="https://github.com/user-attachments/assets/818d5790-52e7-47d1-90b4-835873c46216" />

