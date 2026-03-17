#+private file 
package main

import "core:math/linalg"
import "core:mem"
import "core:c"
import img "vendor:stb/image"
import "core:os"
import "core:crypto/hash"
import base64 "core:encoding/base64"
import "core:fmt"
import "vendor:cgltf"
import "core:strings"
import dx "vendor:directx/d3d12"
import "core:slice"
import dxgi "vendor:directx/dxgi"
import "base:runtime"
import "core:mem/virtual"
import dxma "libs/odin-d3d12ma"
import "core:prof/spall"

@(private="package")
scene_from_gltf :: proc(model_filepath: string) -> Scene {
	
	when PROFILE {
		load_gltf_profile_str := string_append("loading gltf file: ", model_filepath, allocator = g_temp_allocator)
		spall.SCOPED_EVENT(&g_spall_ctx, &g_spall_buffer, name = load_gltf_profile_str)
	}
	
	// loading gltf files
	
	model_filepath_c := strings.clone_to_cstring(model_filepath, g_temp_allocator)
	cgltf_options: cgltf.options
	
	data, ok := cgltf.parse_file(cgltf_options, model_filepath_c)
	defer cgltf.free(data)

	if ok != .success {
		fmt.eprintln("could not read glb")
		os.exit(1)
	}

	load_buffers_result := cgltf.load_buffers(cgltf_options, data, model_filepath_c)
	if load_buffers_result != .success {
		fmt.eprintln("Error loading buffers from gltf: {} - {}", model_filepath, load_buffers_result)
		os.exit(1)
	}
	
	// loading up the Scene
	scene : Scene
	scene_arena := arena_new()
	scene.allocator = scene_arena
	scene_allocator := virtual.arena_allocator(&scene.allocator)
	scene.resource_pool = make(DXResourcePool, scene_allocator)

	assert(len(data.scenes) == 1)
	
	gltf_load_materials_into_scene(data, model_filepath, &scene)
	
	gltf_load_nodes_into_scene(data, &scene)
	gltf_load_meshes_into_scene(data, &scene)
	
	// doing the structured buffer with the model matrices
	{
		// Copying data from cpu to upload resource
		CallbackData :: struct {
			sample_matrix_data: []ModelMatrixData,
			mesh_i: uint,
		}

		data := CallbackData {
			sample_matrix_data = make([]ModelMatrixData, scene.mesh_count, context.allocator), // freed by upload thread.
			mesh_i = 0,
		}

		scene_walk(scene, &data, proc(node: Node, scene: Scene, data: rawptr) {
			if node.mesh == -1 do return
			data := cast(^CallbackData)data
			data.sample_matrix_data[data.mesh_i].model_matrix = get_node_world_matrix(node, scene)
			data.mesh_i += 1
		})
		
		scene.sb_model_matrices = create_structured_buffer_with_data("model matrix data",
		 	&scene.resource_pool,
			slice.to_bytes(data.sample_matrix_data))
		
		// creating SRV (structured buffer) (index 2)
		srv_desc := dx.SHADER_RESOURCE_VIEW_DESC {
			Format = .UNKNOWN,
			ViewDimension = .BUFFER,
			Shader4ComponentMapping = dx.ENCODE_SHADER_4_COMPONENT_MAPPING(0, 1, 2, 3), // this is the default mapping
			Buffer = {
				FirstElement = 0,
				NumElements = u32(scene.mesh_count),
				StructureByteStride = size_of(ModelMatrixData),
				Flags = {},
			},
		}
		
		// create_srv_on_uber_heap(scene.sb_model_matrices, true, "model matrices structured buffer", &srv_desc)
		
		// TODO: REFACTOR THIS ABOMINATION. THIS IS HARDVODED AND WRONG!!!
		g_dx_context.device->CreateShaderResourceView(scene.sb_model_matrices, &srv_desc, get_descriptor_heap_cpu_address(g_dx_context.cbv_srv_uav_heap, 6))
	}
	
	// gizmos stuff
	
	// TODO: separate gizmos from scene
	
	// instance data for gizmos
	{
		instance_data := make([]InstanceData, MAX_GIZMOS, g_temp_allocator)
		scene.vb_gizmos_instance_data = create_vertex_buffer_upload(size_of(instance_data[0]), u32(slice.size(instance_data)), pool = &scene.resource_pool)
	}
	
	scene.ready_value = g_resource_id - 1
	scene.status = .Loading
	return scene
}

gltf_load_meshes_into_scene :: proc(data: ^cgltf.data, scene: ^Scene) {
	
	TEMP_GUARD(&g_temp_arena)
	
	ct := &g_dx_context
	
	vertices := make([dynamic]VertexData, context.allocator) // freed by upload thread
	indices := make([dynamic]u32, context.allocator) // freed by upload thread
	
	scene_allocator := virtual.arena_allocator(&scene.allocator)
	
	the_meshes := make_slice([]Mesh, len(data.meshes), scene_allocator)
	index_count: u32
	index_count_total: u32 = 0

	for mesh, i in data.meshes {
		
		the_meshes[i].primitives = make_slice([]Primitive, len(mesh.primitives), scene_allocator)
		
		for prim, prim_i in mesh.primitives {
			
			// process material here material

			attr_position: cgltf.attribute
			attr_normal: cgltf.attribute
			attr_tangent: cgltf.attribute
			attr_texcoord: [2]cgltf.attribute
			
			textcoord_count := 0
			

			for attribute in prim.attributes {
				#partial switch attribute.type {
				case .position:
					attr_position = attribute
				case .normal:
					attr_normal = attribute
				case .tangent:
					attr_tangent = attribute
				case .texcoord:
					attr_texcoord[attribute.index] = attribute
					textcoord_count += 1
				case:
				// it's outputting "unknown attribute COLOR_0" and it's annoying.
				//  so, i am commenting this error log.
				// fmt.eprintfln("Unkown gltf attribute: {}", attribute)
				}
			}
			

			for i in 0 ..< attr_position.data.count {
				vertex: VertexData
				ok: b32
				
				ok = cgltf.accessor_read_float(attr_position.data, i, &vertex.pos[0], 3)
				if !ok do fmt.eprintln("Error reading gltf position")
				
				ok = cgltf.accessor_read_float(attr_normal.data, i, &vertex.normal[0], 3)
				if !ok do fmt.eprintln("Error reading gltf normal")
				
				if attr_tangent.type != .invalid {
					ok = cgltf.accessor_read_float(attr_tangent.data, i, &vertex.tangent[0], 4)
					if !ok do fmt.eprintln("Error reading gltf tangent")
				}
				
				ok = cgltf.accessor_read_float(attr_texcoord[0].data, i, &vertex.uv[0], 2)
				if !ok do fmt.eprintln("Error reading gltf texcoord")
				
				if textcoord_count > 1 {
					ok = cgltf.accessor_read_float(attr_texcoord[1].data, i, &vertex.uv_2[0], 2)
					if !ok do fmt.eprintln("Error reading gltf texcoord")
				}

				position := v4{vertex.pos.x, vertex.pos.y, vertex.pos.z, 1}
				// vertex.pos = (mesh_mat * position).xyz
				vertex.pos = (position).xyz
				
				// Flipping everything because gltf is right handed and dx12 is left handed.
				vertex.pos.x *= -1
				vertex.normal.x *= -1
				vertex.tangent.x *= -1
				
				append(&vertices, vertex)
				// vertices[i] = vertex
			}

			for i in 0 ..< prim.indices.count {
				append(&indices, u32(cgltf.accessor_read_index(prim.indices, i)) + u32(index_count))
			}

			index_count += u32(attr_position.data.count)
			
			the_meshes[i].primitives[prim_i] = Primitive {
				index_offset = u32(index_count_total),
				index_count = u32(prim.indices.count),
				material_index = u32(cgltf.material_index(data, prim.material))
			}

			index_count_total += u32(prim.indices.count)
			
		}
	}
	
	scene.meshes = the_meshes
	
	// creating and filling vertex and index buffers
	
	uv_sphere_index_offset := u32(len(indices))
	uv_sphere_vertex_offset := u32(len(vertices)) 
	
	// add uv sphere (for gizmos)
	
	// TODO: separate gizmos from scene
	sphere_verts_base, sphere_indices := generate_uv_sphere(32, 32, g_temp_allocator)
	
	for v in sphere_verts_base {
		append(&vertices, VertexData {
			pos = v
		})
	}
	
	for mesh_index in sphere_indices {
		append(&indices, mesh_index + uv_sphere_vertex_offset)
	}
	
	sphere_primitive := make([]Primitive, 1, scene_allocator)
	sphere_primitive[0] = Primitive {
		index_offset = uv_sphere_index_offset,
		index_count = u32(len(sphere_indices))
	}
	
	g_uv_sphere_mesh = Mesh {
		primitives = sphere_primitive
	}
	
	vertex_count := u32(len(vertices))

	// VERTEXDATA
	// vertex data and index data is in an upload heap.
	// This isn't optimal for geometry that doesn't change much.
	// If we want to make this fast, the vertex data needs to be in
	// a DEFAULT heap (vram). you transfer the data from an upload heap
	// to the default heap. but it's more complicated.
	vertex_buffer_size := len(vertices) * size_of(vertices[0])

	resource_desc := dx.RESOURCE_DESC {
		Dimension = .BUFFER,
		Alignment = 0,
		Width = u64(vertex_buffer_size),
		Height = 1,
		DepthOrArraySize = 1,
		MipLevels = 1,
		Format = .UNKNOWN,
		SampleDesc = {Count = 1, Quality = 0},
		Layout = .ROW_MAJOR,
		Flags = {},
	}
	
	vb_allocation : ^dxma.Allocation
	hr := dxma.Allocator_CreateResource(
		pSelf = ct.dxma_allocator,
		pAllocDesc = &dxma.ALLOCATION_DESC{HeapType = .DEFAULT},
		pResourceDesc = &resource_desc,
		InitialResourceState = dx.RESOURCE_STATE_GENERIC_READ,
		pOptimizedClearValue = nil,
		ppAllocation = &vb_allocation,
		riidResource = nil,
		ppvResource = nil
	)
	
	check(hr, "failed creating upload texture")
	scene.vertex_buffer = dxma.Allocation_GetResource(vb_allocation)
	// todo this is wrong
	append(&scene.resource_pool, cast(^dx.IUnknown)vb_allocation)
	
	scene.vertex_buffer->SetName("vertex buffer")

	scene.vertex_buffer_view = dx.VERTEX_BUFFER_VIEW {
		BufferLocation = scene.vertex_buffer->GetGPUVirtualAddress(),
		StrideInBytes = u32(vertex_buffer_size) / vertex_count,
		SizeInBytes = u32(vertex_buffer_size),
	}
	
	dx_upload_order_buffer(scene.vertex_buffer, slice.to_bytes(vertices[:]))

	// creating index buffer resource

	index_buffer_size := len(indices) * size_of(indices[0])
	resource_desc.Width = u64(index_buffer_size)
	
	// upload. no flags
	
	upload_allocation_2 : ^dxma.Allocation
	hr = dxma.Allocator_CreateResource(
		pSelf = ct.dxma_allocator,
		pAllocDesc = &dxma.ALLOCATION_DESC{HeapType = .DEFAULT},
		pResourceDesc = &resource_desc,
		InitialResourceState = dx.RESOURCE_STATE_GENERIC_READ,
		pOptimizedClearValue = nil,
		ppAllocation = &upload_allocation_2,
		riidResource = nil,
		ppvResource = nil
	)
	check(hr, "failed creating upload texture")
	scene.index_buffer = dxma.Allocation_GetResource(upload_allocation_2)
	append(&scene.resource_pool, cast(^dx.IUnknown)upload_allocation_2)
	scene.index_buffer->SetName("lucy's index buffer")

	scene.index_buffer_view = dx.INDEX_BUFFER_VIEW {
		BufferLocation = scene.index_buffer->GetGPUVirtualAddress(),
		SizeInBytes = u32(index_buffer_size),
		Format = .R32_UINT,
	}
	
	dx_upload_order_buffer(scene.index_buffer, slice.to_bytes(indices[:]))
}

gltf_load_nodes_into_scene :: proc(data: ^cgltf.data, scene: ^Scene) {
	
	scene_allocator := virtual.arena_allocator(&scene.allocator)
	
	nodes := make([]Node, len(data.nodes), scene_allocator)
	root_node_count: int = 0

	for node in data.nodes {
		if node.parent == nil {
			root_node_count += 1
		}
	}

	root_nodes := make([]int, root_node_count, scene_allocator)
	root_node_i := 0
	mesh_count: uint

	for node, i in data.nodes {
		// TODO: don't leak this
		node_children := make([]int, len(node.children), scene_allocator)

		for n_child, child_i in node.children {
			node_children[child_i] = int(cgltf.node_index(data, n_child))
		}

		if node.mesh != nil {
			mesh_count += 1
		}
		
		// flipping rotation because of coordinate system differences between gltf and dx12
		flipped_rotation := node.rotation
		
		flipped_rotation.y *= -1
		flipped_rotation.z *= -1
		
		nodes[i] = Node {
			name = strings.clone_from_cstring(node.name, scene_allocator),
			transform_t = node.translation,
			transform_r = node.rotation,
			transform_s = node.scale,
			children = node_children,
			parent = node.parent == nil ? -1 : int(cgltf.node_index(data, node.parent)),
			mesh = node.mesh == nil ? -1 : int(cgltf.mesh_index(data, node.mesh)),
		}

		if node.parent == nil {
			root_nodes[root_node_i] = i
			root_node_i += 1
		}
	}
	
	scene.nodes = nodes
	scene.root_nodes = root_nodes
	scene.mesh_count = mesh_count
}

load_texture :: proc(image: ^cgltf.image, format: dxgi.FORMAT, model_filepath: string, res_pool : ^DXResourcePool, textures_srv_index: ^u32) {
	
	ct := &g_dx_context
	TEMP_GUARD(&g_temp_arena)
	
	image_name : string
	
	if image.uri != nil {
		image_name = string(image.uri)
	} else {
		image_name = string(image.name)
		// TODO: you will need to write this to a file first.
		lprintfln("TEXTURE TYPE NOT IMPLEMENTED. EXITING.")
		os.exit(1)
	}
	
	texture_final_path := texture_cache_query(model_filepath, image_name, format)
	dds_file := parse_dds_file(texture_final_path)
	
	texture_res := create_texture_with_data(dds_file.mipmap_data, u64(dds_file.width), dds_file.height, dds_file.format, 
		res_pool, string(image.name))
	
	ct.device->CreateShaderResourceView(texture_res, nil, get_descriptor_heap_cpu_address(ct.cbv_srv_uav_heap, textures_srv_index^))
	textures_srv_index^ += 1
}

get_texture_uv :: proc(data: ^cgltf.data, tex_view: cgltf.texture_view) -> u32 {
	
	if tex_view.texture != nil {
		return u32(tex_view.texcoord)
	}
	
	assert(false)
	return 0
}

gltf_load_materials_into_scene :: proc(data: ^cgltf.data, model_filepath: string, scene: ^Scene) {
	
	mats := make([]Material, len(data.materials), context.allocator)
	
	
	textures_srv_index : u32 = TEXTURE_INDEX_BASE // starting at 100 to not interfere with other views
	
	for mat, i in data.materials {
		assert(bool(mat.has_pbr_metallic_roughness))
		
		// TODO take evrything else into consideration. (scale, uvs, etc...)
		
		mats[i] = Material {}
		
		// base color
		if mat.pbr_metallic_roughness.base_color_texture.texture != nil {
			// process base color (BC 7 SRGB)
			// load mat.pbr_metallic_roughness.base_color_texture.texture.image_
			
			mats[i].base_color = TextureUV {
				texture_id = textures_srv_index,
				uv_id = get_texture_uv(data, mat.pbr_metallic_roughness.base_color_texture),
			}
			
			load_texture(mat.pbr_metallic_roughness.base_color_texture.texture.image_, 
				.BC7_UNORM_SRGB, model_filepath, &scene.resource_pool, &textures_srv_index)
		}
		
		// metallic roughness
		if mat.pbr_metallic_roughness.metallic_roughness_texture.texture != nil {
			// process metallic roughness (BC 7 UNORM)
			// load mat.pbr_metallic_roughness.metallic_roughness_texture.texture.image_
			
			mats[i].metallic_roughness = TextureUV {
				texture_id = textures_srv_index,
				uv_id = get_texture_uv(data, mat.pbr_metallic_roughness.metallic_roughness_texture),
			}
			
			load_texture(mat.pbr_metallic_roughness.metallic_roughness_texture.texture.image_,
			 .BC7_UNORM, model_filepath, &scene.resource_pool, &textures_srv_index)
		}
		
		// normal
		if mat.normal_texture.texture != nil {
			// process normal texture (BC 5 UNORM)
			// load mat.normal_texture.texture.image_
			
			mats[i].normal = TextureUV {
				texture_id = textures_srv_index,
				uv_id = get_texture_uv(data, mat.normal_texture),
			}
			
			load_texture(mat.normal_texture.texture.image_,
			 .BC5_UNORM, model_filepath, &scene.resource_pool, &textures_srv_index)
		}
	}
	
	scene.sb_materials = create_structured_buffer_with_data(
		"material buffer",
		&scene.resource_pool,
		slice.to_bytes(mats)
	)
	
	srv_desc := dx.SHADER_RESOURCE_VIEW_DESC {
		Format = .UNKNOWN,
		ViewDimension = .BUFFER,
		Shader4ComponentMapping = dx.ENCODE_SHADER_4_COMPONENT_MAPPING(0, 1, 2, 3), // this is the default mapping
		Buffer = {
			FirstElement = 0,
			NumElements = u32(len(mats)),
			StructureByteStride = size_of(mats[0]),
			Flags = {},
		},
	}
	
	// create_srv_on_uber_heap(scene.sb_materials, true, "materials srv", &srv_desc)
	
	// TODO: REFACTOR THIS ABOMINATION. THIS IS HARDVODED AND WRONG!!!
	g_dx_context.device->CreateShaderResourceView(scene.sb_materials, &srv_desc, get_descriptor_heap_cpu_address(g_dx_context.cbv_srv_uav_heap, 5))
}
