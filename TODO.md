# current todo

- handle transparency in textures (the flowers)

- zed: go to file:line is broken in windows. report it.

- pass indices into the descriptor heap in a constant buffer so u don't have to constantly,
	modify the magic values

- decide on the setup of your g buffer for PBR rendering. see this:
	- https://www.reddit.com/r/opengl/comments/z2kdgm/deferred_rendering_reducing_the_size_of_the/
	- possible setup:
	    Albedo (RGB8U).
	    Emmisive (RGB8U).
	    Depth and stencil (D24S8).
	
your idea is:

base color (already done)
Normal (Packed with Octahedron mapping) (RG8U). (halfway done)
AO + Roughness + Metallic (RGB8U).

Depth and stencil (already done)

- use thread pool for loading textures
	- main thread loads the files to memory
	- worker threads do the png processing using stb image
	- then as for dx.. that's unclear

- respond to the peep in the dx discord about the linalg package in odin and the perspective depth fix.
	- "What depth values does it come up with?
I'm using the linalg package too https://github.com/lodinukal/en/blob/master/app%2Fmain.odin#L261-L266 and it's working fine"
	- check depth values in renderdoc

- the image or meshes are flipped horizontally. it's all inverted horizontally.
	- curtains: left: green. right: red. lion = in front of you

- take memory management more seriously
	- use the tracking allocator
	- find out how to do memory checkpoints

- copy code from here to filter out instrumentation markers
	- https://github.com/joaocarvalhoopen/ota_profiller__Odin_Terminal_Auto_Profiller_Lib/blob/main/ota_profiller/otprofiller.odin

- rebind `g t` to "go to type" (it's at `g y`)

- TODO: find out how to abstract and/or share fences
  
- HotSwapState :: struct: 
    // TODO: store more data here so u don't have to pass the data around in the hotswap methods

- set up some basic allocator stuff
    - set up a tracking allocator for lasting allocations.

- check this out. p cool. about memory management.
https://github.com/microsoft/DirectX-Graphics-Samples/tree/master/TechniqueDemos/D3D12MemoryManagement

# possible cool goals:

- [ ] render sponza
- [ ] blinn phong lighting (just need to add some stuff to the shader to accomplish this)
- [x] deferred rendering (read it in book) (page 883)


- some sort of scene structure so you can render multiple things


# references

- seems like a general dx12 guide. check it out:

https://www.youtube.com/watch?v=foG5_BegCzU&list=PLD3tf_aBsga1A9B7UoDkM-yObxlLh9pku&index=27


- [Scene Samples](https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-research/samples.html)

# devaniti on memory management system

I'd use this strategy if you are still figuring out dx12:
- Have a Fence per queue, with incrementing value. That way, by reading current value of that fence you can figure out how far did you GPU work progressed. For this example I'm assuming that you only have direct queue, but everything described here is scalable for multiple queues.
- When recording command lists, you keep track of which value was last signaled to a fence, and which value will be signaled next. Let's call that value that will be signaled next a "safe value".
- When writing commands to a command list and using a resource in the process, you can store that "safe value" for that resource. You update that value each time you use a resource.
- When you no longer need a given resource, instead of immediately destroying it, you add it to a priority queue with its safe value used as priority (you would need a priority queue where smaller value = higher priority).
- Periodically, for example each frame, you query current completed value for a fence and destroy all objects whose safe value are less or equal to completed value.
- On exit, you signal for the last time, and then immedatelly wait for that same value. That way your fence had to reach the last possible "safe value", after which you do one more iteration of destroying objects, after which there should be none left.

The logic behind that "safe value" is: That "safe value" is only going to be signaled after the last operation done on a resource. In other words "safe value" is a value after observing which it is safe to destroy a resource.
me: "I'd use this strategy if you are still figuring out dx12"
also me: Proceeds to explain kinda complicated scheme that is actually often used in real projects
😂


# crash logs

odin crashes

## log 1

```
[DEBUG] Selected microarch: x86-64-v2
[DEBUG] Default microarch features: 64bit,64bit-mode,cmov,crc32,cx16,cx8,false-deps-popcnt,fast-15bytenop,fast-scalar-fsqrt,fast-shld-rotate,fxsr,idivq-to-divl,macrofusion,mmx,nopl,popcnt,sahf,slow-3ops-lea,slow-unaligned-mem-32,sse,sse2,sse3,sse4.1,sse4.2,ssse3,vzeroupper,x87
[DEBUG] Target triplet: x86_64-pc-windows-msvc
[DEBUG] build_paths[0]: E:/dev/dx12/
[DEBUG] build_paths[1]:
[DEBUG] build_paths[2]:
[DEBUG] build_paths[3]: C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64/
[DEBUG] build_paths[4]: C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22621.0/um/x64/
[DEBUG] build_paths[5]: C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22621.0/ucrt/x64/
[DEBUG] build_paths[6]: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.41.34120/bin/Hostx64/x64/
[DEBUG] build_paths[7]: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.41.34120/lib/x64/
[DEBUG] build_paths[8]: E:/dev/dx12/lucydx12.exe
[DEBUG] build_paths[9]: E:/dev/dx12/lucydx12.pdb
[DEBUG] [Section] init thread pool
[DEBUG] [Section] init universal
[DEBUG] [Section] parse files
[DEBUG] [Section] init checker info
[DEBUG] [Section] checker info: general
[DEBUG] [Section] init proc queues
[DEBUG] [Section] type check
[DEBUG] [Section] map full filepaths to scope
[DEBUG] [Section] init worker data
[DEBUG] [Section] create file scopes
[DEBUG] [Section] collect entities
[DEBUG] [Section] export entities - pre
[DEBUG] [Section] check_import_entities - sort packages
[DEBUG] [Section] check_import_entities - collect file decls
[DEBUG] [Section] check_import_entities - check delayed entities
[DEBUG] [Section] export entities - post
[DEBUG] [Section] add entities from packages
[DEBUG] [Section] check all global entities
[DEBUG] [Section] init preload
[DEBUG] [Section] add global untyped expression to queue
[DEBUG] [Section] check procedure bodies

⏵ Task `build debug` finished with non-zero error code: -1073741819
⏵ Command: cmd /C "odin build . -debug -out:lucydx12.exe -linker:radlink -show-debug-messages"
```


## log 2

```
[DEBUG] Selected microarch: x86-64-v2
[DEBUG] Default microarch features: 64bit,64bit-mode,cmov,crc32,cx16,cx8,false-deps-popcnt,fast-15bytenop,fast-scalar-fsqrt,fast-shld-rotate,fxsr,idivq-to-divl,macrofusion,mmx,nopl,popcnt,sahf,slow-3ops-lea,slow-unaligned-mem-32,sse,sse2,sse3,sse4.1,sse4.2,ssse3,vzeroupper,x87
[DEBUG] Target triplet: x86_64-pc-windows-msvc
[DEBUG] build_paths[0]: E:/dev/dx12/
[DEBUG] build_paths[1]:
[DEBUG] build_paths[2]:
[DEBUG] build_paths[3]: C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64/
[DEBUG] build_paths[4]: C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22621.0/um/x64/
[DEBUG] build_paths[5]: C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22621.0/ucrt/x64/
[DEBUG] build_paths[6]: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.41.34120/bin/Hostx64/x64/
[DEBUG] build_paths[7]: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.41.34120/lib/x64/
[DEBUG] build_paths[8]: E:/dev/dx12/lucydx12.exe
[DEBUG] build_paths[9]: E:/dev/dx12/lucydx12.pdb
[DEBUG] [Section] init thread pool
[DEBUG] [Section] init universal
[DEBUG] [Section] parse files
[DEBUG] [Section] init checker info
[DEBUG] [Section] checker info: general
[DEBUG] [Section] init proc queues
[DEBUG] [Section] type check
[DEBUG] [Section] map full filepaths to scope
[DEBUG] [Section] init worker data
[DEBUG] [Section] create file scopes
[DEBUG] [Section] collect entities
[DEBUG] [Section] export entities - pre
[DEBUG] [Section] check_import_entities - sort packages
[DEBUG] [Section] check_import_entities - collect file decls
[DEBUG] [Section] check_import_entities - check delayed entities
[DEBUG] [Section] export entities - post
[DEBUG] [Section] add entities from packages
[DEBUG] [Section] check all global entities
[DEBUG] [Section] init preload
[DEBUG] [Section] add global untyped expression to queue
[DEBUG] [Section] check procedure bodies

⏵ Task `build debug` finished with non-zero error code: -1073741819
⏵ Command: cmd /C "odin build . -debug -out:lucydx12.exe -linker:radlink -show-debug-messages"
```
