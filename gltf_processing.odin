package main

import "core:fmt"
import "vendor:cgltf"
import "core:strings"
import dx "vendor:directx/d3d12"
import img "vendor:stb/image"
import "core:slice"
import "core:path/filepath"
import "core:c"
import dxgi "vendor:directx/dxgi"

gltf_process_data :: proc() -> (vertices: []VertexData, indices: []u32) {
	model_filepath_c := strings.clone_to_cstring(model_filepath, context.temp_allocator)

	cgltf_options: cgltf.options

	data, ok := cgltf.parse_file(cgltf_options, model_filepath_c)
	defer cgltf.free(data)

	if ok != .success {
		fmt.eprintln("could not read glb")
	}

	load_buffers_result := cgltf.load_buffers(cgltf_options, data, model_filepath_c)
	if load_buffers_result != .success {
		fmt.eprintln("Error loading buffers from gltf: {} - {}", model_filepath, load_buffers_result)
	}

	// new stuff start

	assert(len(data.scenes) == 1)

	gltf_scene := data.scenes[0]

	vertices_dyn := make([dynamic]VertexData)
	indices_dyn := make([dynamic]u32)

	g_materials = gltf_load_materials(data)
	g_meshes = gltf_load_meshes(data, &vertices_dyn, &indices_dyn)
	
	gltf_load_textures(data)
	
	scene = gltf_load_nodes(data)

	// printing nodes
	// scene_walk(scene, nil, proc(node: Node, scene: Scene, data: rawptr) {
	//     lprintfln("node name: %v", node.name)
	// })

	// for root_node in gltf_scene.nodes {
	//     gltf_load_nodes(data, root_node, &vertices_dyn, &indices_dyn)
	// }

	return vertices_dyn[:], indices_dyn[:]
}


gltf_load_meshes :: proc(data: ^cgltf.data, vertices: ^[dynamic]VertexData, indices: ^[dynamic]u32) -> []Mesh {

	the_meshes := make_slice([]Mesh, len(data.meshes))
	index_count: u32
	index_count_total: u32 = 0

	for mesh, i in data.meshes {
		
		the_meshes[i].primitives = make_slice([]Primitive, len(mesh.primitives))
		
		for prim, prim_i in mesh.primitives {
			
			// process material here material

			attr_position: cgltf.attribute
			attr_normal: cgltf.attribute
			attr_texcoord: [2]cgltf.attribute
			
			textcoord_count := 0
			

			for attribute in prim.attributes {
				#partial switch attribute.type {
				case .position:
					attr_position = attribute
				case .normal:
					attr_normal = attribute
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
				
				append(vertices, vertex)
				// vertices[i] = vertex
			}

			for i in 0 ..< prim.indices.count {
				append(indices, u32(cgltf.accessor_read_index(prim.indices, i)) + u32(index_count))
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

	return the_meshes
}


gltf_load_nodes :: proc(data: ^cgltf.data) -> Scene {

	// TODO: don't leak this
	nodes := make([]Node, len(data.nodes))
	root_node_count: int = 0

	for node, i in data.nodes {
		if node.parent == nil {
			root_node_count += 1
		}
	}

	// TODO: don't leak this
	root_nodes := make([]int, root_node_count)
	root_node_i := 0
	mesh_count: uint

	for node, i in data.nodes {
		// TODO: don't leak this
		node_children := make([]int, len(node.children))

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
			name = strings.clone_from_cstring(node.name),
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

	return Scene{nodes = nodes, root_nodes = root_nodes, mesh_count = mesh_count}
}


gltf_load_textures :: proc(data : ^cgltf.data) {
	
	ct := dx_context
	
	upload_resources := make([dynamic]^dx.IUnknown, 0, len(data.images))
	defer delete(upload_resources)
	
	dx_context.cmdlist->Reset(dx_context.command_allocator, ct.pipeline_gbuffer)
	
	load_white_texture(&upload_resources)
	
	if TEXTURE_LIMIT == 0 {
		return
	}
	
	textures_srv_index : u32 = TEXTURE_INDEX_BASE // starting at 100 to not interfere with other views
	
	for image, i in data.images {
		// lprintfln("loading image %v", image.name)
		// assert(image.mime_type == "image/png")
		w, h, channels : c.int
		channel_count :: 4
		image_data : [^]byte
		image_name : string
		
		if image.uri != nil {
			channel_count :: 4
			//  u gotta concatenate the name
			image_dir := filepath.dir(model_filepath, context.temp_allocator)
			image_path := filepath.join({image_dir, string(image.uri)}, context.temp_allocator)
			image_path_cstring := strings.clone_to_cstring(image_path, context.temp_allocator)
			
			image_data = img.load(image_path_cstring, &w, &h, &channels, channel_count)
			image_name = string(image.uri)
			assert(image_data != nil)
		} else {
			png_data := cgltf.buffer_view_data(image.buffer_view)
			png_size := image.buffer_view.size
			
			image_data = img.load_from_memory(png_data, i32(png_size), &w, &h, &channels, channel_count)
			image_name = string(image.name)
			assert(image_data != nil)
		}
		
		defer img.image_free(image_data)
		
		texture_format : dxgi.FORMAT : .R8G8B8A8_UNORM
		
		texture_res := create_texture_with_data(auto_cast(image_data), u64(w), u32(h), channel_count, texture_format, 
			&resources_longterm, &upload_resources, ct.cmdlist, string(image.name))
		
		// lprintfln("name: %v, index in the heap: %v", image.name, textures_srv_index)
		
		// creating srv on uber heap
		ct.device->CreateShaderResourceView(texture_res, nil, get_descriptor_heap_cpu_address(ct.cbv_srv_uav_heap, textures_srv_index))
		textures_srv_index += 1
		
		
		// enforcing a limit bc it's so slow
		if i > TEXTURE_LIMIT do break
	}
	
	// execute command list
	execute_command_list_and_wait(ct.cmdlist, ct.queue)
	
	// free upload resources
	for res in upload_resources {
		res->Release()
	}
}

@(private="file")
get_texture_index_uv :: proc(data: ^cgltf.data, tex_view: cgltf.texture_view) -> (u32, u32){
	
	base_color_img_index : u32 = TEXTURE_WHITE_INDEX
	base_color_uv_index : u32 = 0
	texture_name : cstring = "no base texture"
	
	if TEXTURE_LIMIT != 0 && tex_view.texture != nil {
		base_color_img_index = TEXTURE_INDEX_BASE + u32(cgltf.image_index(data, tex_view.texture.image_))
		texture_name = tex_view.texture.image_.name
		base_color_uv_index = u32(tex_view.texcoord)
	}
	
	return base_color_img_index, base_color_uv_index
}

gltf_load_materials :: proc(data: ^cgltf.data) -> []Material {
	
	ct := &dx_context
	
	mats := make([]Material, len(data.materials))
	
	for mat, i in data.materials {
		assert(bool(mat.has_pbr_metallic_roughness))
		
		// loading base color
		// mat.pbr_metallic_roughness.base_color_texture.
		
		// TODO take evrything else into consideration. (scale, uvs, etc...)
		
		// metallic roughness texture
		// mat.pbr_metallic_roughness.metallic_roughness_texture
		
		// lprintfln("name: %v, cgltf index: %v", texture_name, base_color_img_index)
		// base color
		bc_i, bc_uv := get_texture_index_uv(data, mat.pbr_metallic_roughness.base_color_texture)
		
		// metallic, roughness
		mr_i, mr_uv := get_texture_index_uv(data, mat.pbr_metallic_roughness.metallic_roughness_texture)
		
		
		mats[i] = Material {
			base_color_index = bc_i,
			base_color_uv_index = bc_uv,
			metallic_roughness_index = mr_i,
			metallic_roughness_uv_index = mr_uv
		}
	}
	
	ct.sb_materials = create_structured_buffer_with_data(
		ct.cmdlist,
		"material buffer",
		&resources_longterm,
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
	
	create_srv_on_uber_heap(ct.sb_materials, true, "materials srv", &srv_desc)
	
	return mats
}
