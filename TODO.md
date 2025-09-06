# Goals

It's a big jump from DX11 to DX12, but your DX11 experience is a great foundation. The biggest change is the shift from a high-level, driver-managed API to a low-level, developer-managed one. In DX11, the driver did a lot of work for you behind the scenes, such as managing state and resource dependencies. In DX12, you must handle all of this explicitly. Odin's low-level nature is a perfect fit for this.

## Key Differences to Master

* **Command Lists and Queues:** In DX11, you had an immediate context and deferred contexts. In DX12, everything is recorded into **command lists**, which are then submitted to a **command queue** for execution on the GPU. This is the core of the new model and the source of most of the performance benefits.
* **Resource Management:** You are responsible for managing resource state transitions using **resource barriers**. You need to explicitly tell the GPU when a resource is changing from being a render target to a shader resource, for example.
* **Descriptor Heaps and Tables:** In DX11, you would bind resources directly to shader slots. In DX12, you use **descriptor heaps** (arrays of descriptors) and **descriptor tables** (pointers to a range of descriptors within a heap).
* **Root Signatures and Pipeline State Objects (PSOs):** The **pipeline state object (PSO)** is a massive state container that bakes together all the shader and fixed-function state into a single, immutable object. The **root signature** acts as the contract for the PSO, defining what resources the shaders expect.

## Small, Achievable Goals

1.  **"Hello Triangle"**: This is your first major milestone. Replicate the basic triangle example from a DX12 tutorial, but in Odin. This will force you to set up the device, command queue, command list, swap chain, and a basic PSO and root signature.
2.  **Constant Buffer Animation:** Animate your triangle using a constant buffer. This will teach you about mapping and unmapping resources and updating them on the CPU for the GPU to consume.
3.  **Draw a Textured Cube:** Move from a simple triangle to a more complex object. This will introduce you to creating and managing descriptor heaps for your textures and understanding how to use resource barriers to get the texture data to the GPU.
4.  **Multi-Command List Rendering:** Once you have a basic scene, try to offload some work to a different command list. This will give you a hands-on understanding of parallel command recording and how fences are used to synchronize the CPU and GPU timelines.
5.  **Instancing:** Implement instancing to draw many copies of the same object efficiently. This will teach you how to pass per-instance data to the GPU and see the performance benefits of a low-level API in action.

**General Advice:**

* Start with the official Microsoft DirectX 12 documentation and samples. They are dense but are the ultimate source of truth.
* Find Odin-specific DX12 examples. While Odin has great C++ interop, seeing idiomatic Odin code for DX12 will be invaluable.
* Don't be afraid to read and re-read the documentation. Many DX12 concepts take time to sink in.
* Use a debugging tool like PIX for Windows. It is essential for seeing exactly what your code is telling the GPU and for troubleshooting.

