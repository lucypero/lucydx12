# releasing things

current unreleased number: (Refcount of device)

42
37
36
34
30
21
11
10
3




# current todo
- just draw a red screen sized quad for the lighting pass, as a test.
- deferred rendering: a second PSO for the final render.
- set up some basic allocator stuff
    - set up a tracking allocator for lasting allocations.

- experiment with `setpipelinestate`. see what u can delete after using that.

- check this out. p cool. about memory management.
https://github.com/microsoft/DirectX-Graphics-Samples/tree/master/TechniqueDemos/D3D12MemoryManagement


# possible cool goals:

- blinn phong lighting (just need to add some stuff to the shader to accomplish this)
- deferred rendering (read it in book) (page 883)


# references


- seems like a general dx12 guide. check it out:

https://www.youtube.com/watch?v=foG5_BegCzU&list=PLD3tf_aBsga1A9B7UoDkM-yObxlLh9pku&index=27


# code by kamwithk for deleting stuff

```odin

flush_allocation_queue :: proc(queue: ^Allocation_Queue) {
    #reverse for resource in queue {
        switch handle in resource {
        case vk.Device:
            vk.DestroyDevice(g.device, nil)
        case vk.Instance:
            vk.DestroyInstance(g.instance, nil)
        case vk.DebugUtilsMessengerEXT:
            vk.DestroyDebugUtilsMessengerEXT(g.instance, handle, nil)
        case vk.CommandPool:
            vk.DestroyCommandPool(g.device, handle, nil)
        case vk.DescriptorPool:
            vk.DestroyDescriptorPool(g.device, handle, nil)
        case vk.DescriptorSetLayout:
            vk.DestroyDescriptorSetLayout(g.device, handle, nil)
        case vk.Fence:
            vk.DestroyFence(g.device, handle, nil)
        case vk.Semaphore:
            vk.DestroySemaphore(g.device, handle, nil)
        case vk.Image:
            vk.DestroyImage(g.device, handle, nil)
        case vk.ImageView:
            vk.DestroyImageView(g.device, handle, nil)
        case vk.Pipeline:
            vk.DestroyPipeline(g.device, handle, nil)
        case vk.PipelineLayout:
            vk.DestroyPipelineLayout(g.device, handle, nil)
        case vk.Sampler:
            vk.DestroySampler(g.device, handle, nil)
        case vk.ShaderEXT:
            vk.DestroyShaderEXT(g.device, handle, nil)
        case vk.SurfaceKHR:
            vk.DestroySurfaceKHR(g.instance, g.surface, nil)
        case vk.Buffer:
            vk.DestroyBuffer(g.device, handle, nil)
        case vk.DeviceMemory:
            vk.FreeMemory(g.device, handle, nil)
        }
    }

    clear(queue)
}

// section b
// 
// 
Allocation_Queue :: [dynamic]Resource_Handle
Resource_Handle :: union {
    vk.DebugUtilsMessengerEXT,
    vk.Device,
    vk.Instance,
    vk.CommandPool,
    vk.DescriptorPool,
    vk.DescriptorSetLayout,
    vk.Fence,
    vk.Semaphore,
    vk.Image,
    vk.ImageView,
    vk.Pipeline,
    vk.PipelineLayout,
    vk.Sampler,
    vk.ShaderEXT,
    vk.SurfaceKHR,
    vk.Buffer,
    vk.DeviceMemory,
}

append(&g.immediate_resources.allocation_queue, g.model_buffers.scene_buffer)

```

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
