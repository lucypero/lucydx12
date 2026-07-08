package main

import "core:text/regex"
import "core:time"
import "core:c"
import img "vendor:stb/image"
import "core:math/linalg"
import "core:mem/virtual"
import "core:reflect"
import base64 "core:encoding/base64"
import "core:crypto/hash"
import "core:path/filepath"
import "core:encoding/endian"
import "core:slice"
import "core:mem"
import dx "vendor:directx/d3d12"
import dxgi "vendor:directx/dxgi"
import dxc "vendor:directx/dxc"
import "core:strings"
import "core:os"
import "core:sys/windows"
import "core:fmt"
import "base:runtime"
import "core:math"
import dxma "../libs/odin-d3d12ma"
import im "../libs/odin-imgui"

/*
transition_resource_from_copy_to_read :: proc(res: ^dx.IResource, cmd_list: ^dx.IGraphicsCommandList) {
	transition_resource(res, cmd_list, {.COPY_DEST}, dx.RESOURCE_STATE_GENERIC_READ)
}
*/

UberDescriptorHeap :: struct {
	heap: ^dx.IDescriptorHeap,
	next_descriptor_index: int,
}

uber_heap_create :: proc(type: dx.DESCRIPTOR_HEAP_TYPE, pool: ^DXResourcePool) -> UberDescriptorHeap {

	ct := &g_dx_context

	num_descriptors :: 1000000
	heap : ^dx.IDescriptorHeap
	desc : dx.DESCRIPTOR_HEAP_DESC

	#partial switch type {
	case .CBV_SRV_UAV:
		desc = dx.DESCRIPTOR_HEAP_DESC {
			NumDescriptors = 1000000,
			Type = .CBV_SRV_UAV,
			Flags = {.SHADER_VISIBLE},
		}
	case .DSV:
		desc = dx.DESCRIPTOR_HEAP_DESC {
			NumDescriptors = 1000000,
			Type = .DSV,
			Flags = {},
		}
	case .RTV:
		desc = dx.DESCRIPTOR_HEAP_DESC {
			NumDescriptors = 1000000,
			Type = .RTV,
			Flags = {},
		}
	case: 
		panic("heap type not supported")
	}

	hr := ct.device->CreateDescriptorHeap(&desc, dx.IDescriptorHeap_UUID, (^rawptr)(&heap))
	check(hr, "Failed creating descriptor heap")
	append(pool, heap)

	return UberDescriptorHeap {
		heap = heap,
		next_descriptor_index = 0
	}
}

uber_heap_count :: proc(heap: ^UberDescriptorHeap) {
	heap.next_descriptor_index += 1
}

// gets next cpu address to create the next view
uber_heap_get_next_cpu_addr :: proc(uber_heap: UberDescriptorHeap) -> dx.CPU_DESCRIPTOR_HANDLE {
	return get_descriptor_heap_cpu_address(uber_heap.heap, uber_heap.next_descriptor_index)
}

// returns cpu descriptor handle of resource at offset `index`
uber_heap_get_cpu_addr :: proc(uber_heap: UberDescriptorHeap, index: int) -> dx.CPU_DESCRIPTOR_HANDLE {
	return get_descriptor_heap_cpu_address(uber_heap.heap, index)
}

BlendState :: enum {
	Normal,
	Off
}

FillMode :: enum {
	Solid,
	Wireframe
}

CullMode :: enum {
	Back,
	Front,
	None
}

RootSignatureChoice :: enum {
	Standard,
	Custom1
}

PSOParameters :: struct {
	vertex_input: typeid,
	instance_vertex_input: Maybe(typeid),
	fill_mode: FillMode,
	blend_state: BlendState,
	cull_mode: CullMode,
	enable_depth: bool,
	depth_write: bool,
	root_signature: RootSignatureChoice,
	front_counter_clockwise: bool,
	rtv_count: int,
	rtv_formats: [8]dxgi.FORMAT,
}

VertexInputA :: struct {
	asd : v3 `POSITION`,
	dsa : v4 `COLOR`,
	lala: dxm `MATRIX`,
}

add_to_input_element_desc :: proc(buffer_type: typeid, is_instance: bool, result: ^[dynamic]dx.INPUT_ELEMENT_DESC) {
	names := reflect.struct_field_names(buffer_type)
	types := reflect.struct_field_types(buffer_type)
	tags  := reflect.struct_field_tags(buffer_type)

	assert(len(names) == len(types) && len(names) == len(tags))

	for type, i in types {

		elem_format : dxgi.FORMAT

		tags_c, err := strings.clone_to_cstring(string(tags[i]), context.temp_allocator)
		assert(err == .None)

		#partial switch v in type.variant {
		case reflect.Type_Info_Array:
			// assume float

			switch v.count {

			case 2:
				switch v.elem_size {
				case 4:
					elem_format = .R32G32_FLOAT
				case 2:
					elem_format = .R16G16_FLOAT
				case:
					panic("format not supported")
				}
			case 3:
				switch v.elem_size {
				case 4:
					elem_format = .R32G32B32_FLOAT
				case:
					panic("format not supported")
				}
			case 4:
				switch v.elem_size {
				case 4:
					elem_format = .R32G32B32A32_FLOAT
				case 2:
					elem_format = .R16G16B16A16_FLOAT
				case:
					panic("format not supported")
				}
			case:
				panic("count not supported")
			}

			if !reflect.is_float(v.elem) {
				// convert from FLOAT type to another type.
				panic("format not supported")
			} 		

			append(result, dx.INPUT_ELEMENT_DESC{
				SemanticName = tags_c,
				Format = elem_format,
				AlignedByteOffset = dx.APPEND_ALIGNED_ELEMENT,
				InputSlotClass = is_instance ? .PER_INSTANCE_DATA : .PER_VERTEX_DATA,
				InputSlot = is_instance ? 1 : 0,
				InstanceDataStepRate = is_instance ? 1 : 0,
			})
		case reflect.Type_Info_Matrix:
			assert(v.row_count == 4 && v.column_count == 4, "matrix not supported")

			elem_format = .R32G32B32A32_FLOAT

			for j in 0..=3 {
				append(result, dx.INPUT_ELEMENT_DESC{
					SemanticName = tags_c,
					SemanticIndex = u32(j),
					Format = elem_format,
					AlignedByteOffset = dx.APPEND_ALIGNED_ELEMENT,
					InputSlotClass = is_instance ? .PER_INSTANCE_DATA : .PER_VERTEX_DATA,
					InputSlot = is_instance ? 1 : 0,
					InstanceDataStepRate = is_instance ? 1 : 0,
				})
			}
		}
	}
}

get_dx_vertex_input :: proc(input_layout_vertex: typeid, input_layout_instance: Maybe(typeid)) -> []dx.INPUT_ELEMENT_DESC {
	res := make_dynamic_array_len_cap([dynamic]dx.INPUT_ELEMENT_DESC, 0, 0, context.temp_allocator)

	add_to_input_element_desc(input_layout_vertex, false, &res)

	if il, ok := input_layout_instance.?; ok {
		add_to_input_element_desc(il, true, &res)
	}

	return res[:]
}


pso_create :: proc(shader_filename: string, parameters: PSOParameters, render_proc: proc(pso:PSO), pso_name: string = "") -> PSO {
	ct := &g_dx_context
	vs, ps, ok := compile_shader(ct.dxc_compiler, shader_filename)
	assert(ok, "could not compile shader!! check logs")

	defer {
		vs->Release()
		ps->Release()
	}

	assert(parameters.root_signature == .Standard, "custom root signatures are not supported")
	root_signature := ct.root_signatures[parameters.root_signature]

	// Creating the actual PSO
	pso_dx: ^dx.IPipelineState = create_pso_dx(parameters, root_signature, vs, ps, pso_name)

	pso_index := len(g_resources_longterm)
	append(&g_resources_longterm, pso_dx)

	pso := PSO {
		pipeline_state = pso_dx,
		root_signature = root_signature,
		shader_filename = shader_filename,
		parameters = parameters,
		pso_index = pso_index,
		pso_name = pso_name,
		render_proc = render_proc
	}

	pso_hotswap_init(&pso)
	return pso
}

transition_resource :: proc(res: ^dx.IResource, cmd_list: ^dx.IGraphicsCommandList, state_before, state_after: dx.RESOURCE_STATES, subresource: u32 = 0) {
	barrier : dx.RESOURCE_BARRIER = {
		Type = .TRANSITION,
		Flags = {},
		Transition = {
			pResource = res,
			StateBefore = state_before,
			StateAfter = state_after,
			Subresource = subresource
		}
	}

	// run resource barrier
	cmd_list->ResourceBarrier(1, &barrier)
}


g_include_handler : ^dxc.IIncludeHandler

dxc_init :: proc() -> ^dxc.ICompiler3 {
	// todo here
	utils : ^dxc.IUtils
	compiler : ^dxc.ICompiler3

	dxc.CreateInstance(dxc.Utils_CLSID, dxc.IUtils_UUID, (^rawptr)(&utils))
	dxc.CreateInstance(dxc.Compiler_CLSID, dxc.ICompiler3_UUID, (^rawptr)(&compiler))

	utils->CreateDefaultIncludeHandler(&g_include_handler)

	return compiler
}

// compiles vertex and pixel shader, for a graphics pipeline
compile_shader :: proc(compiler: ^dxc.ICompiler3, shader_filename: string) -> (vs, ps: ^dxc.IBlob, ok: bool) {

	data, ok_f := os.read_entire_file_from_path(shader_filename, context.allocator)

	if ok_f != os.General_Error.None {
		lprintfln("could not read file")
		os.exit(1)
	}

	defer(delete(data))

	if len(data) == 0 do return vs, ps, false

	source_buffer := dxc.Buffer {
		Ptr = &data[0],
		Size = len(data),
		Encoding = dxc.CP_ACP
	}

	vs, ok = compile_individual_shader(shader_filename, &source_buffer, compiler, .Vertex)

	if !ok do return vs, ps, false

	ps, ok = compile_individual_shader(shader_filename, &source_buffer, compiler, .Pixel)

	if !ok do return vs, ps, false

	return vs, ps, true
}

compile_shader_compute :: proc(compiler: ^dxc.ICompiler3, shader_filename: string) -> (cs: ^dxc.IBlob, ok: bool) {

	data, ok_f := os.read_entire_file_from_path(shader_filename, context.allocator)

	if ok_f != os.General_Error.None {
		lprintfln("could not read file")
		os.exit(1)
	}

	defer(delete(data))

	if len(data) == 0 do return

	source_buffer := dxc.Buffer {
		Ptr = &data[0],
		Size = len(data),
		Encoding = dxc.CP_ACP
	}

	cs, ok = compile_individual_shader(shader_filename, &source_buffer, compiler, .Compute)
	if !ok do return

	return cs, true
}

ShaderKind :: enum {
	Vertex,
	Pixel,
	Compute
}

compile_individual_shader :: proc(shader_filename: string, source_buffer: ^dxc.Buffer, compiler: ^dxc.ICompiler3, shader_kind: ShaderKind) -> (res:^dxc.IBlob, ok: bool) {

	arguments : [dynamic; 10]string 

	append(&arguments, "-E", "???", "-T", "???", "-O3")

	switch shader_kind {
	case .Vertex:
		arguments[1] = "VSMain"
		arguments[3] = "vs_6_6"
	case .Pixel:
		arguments[1] = "PSMain"
		arguments[3] = "ps_6_6"
	case .Compute:
		arguments[1] = "CSMain"
		arguments[3] = "cs_6_6"
	}

	when ODIN_DEBUG {
	// disabling optimization, so we can debug the shader.
	arguments[4] = "-Od"
	append(&arguments, "-Zi", "-Fd", ".\\pdbs\\")

	// sb := strings.builder_make_none(context.temp_allocator)
	// fmt.sbprintf("%v.pdb")

	}

	switch g_config.aa_options {
	case .NoAA:
	case .FXAA:
		append(&arguments, "-D", "FXAA_ENABLE")
	}

	arguments_wide := make([]windows.wstring, len(arguments), context.temp_allocator)

	for arg, i in arguments {
		arguments_wide[i] = windows.utf8_to_wstring_alloc(arg, allocator = context.temp_allocator)
	}

	results : ^dxc.IResult
	compiler->Compile(source_buffer, &arguments_wide[0], cast(u32)len(arguments_wide), g_include_handler, dxc.IOperationResult_UUID, (^rawptr)(&results))

	errors : ^dxc.IBlobUtf8
	results->GetOutput(.ERRORS, dxc.IBlobUtf8_UUID, (^rawptr)(&errors), nil)
	if errors != nil && errors->GetStringLength() > 0 {
		error_str := strings.string_from_ptr((^u8)(errors->GetBufferPointer()), int(errors->GetBufferSize()))
		lprintfln("dxc: errors at %v: %v", shader_filename, error_str)
	}

	output_blob : ^dxc.IBlob

	hr : dxc.HRESULT
	results->GetStatus(&hr)
	if hr < 0 {
		return output_blob, false
	}

	results->GetOutput(.OBJECT, dxc.IBlob_UUID, (^rawptr)(&output_blob), nil)


	when ODIN_DEBUG {

	pdb_blob: ^dxc.IBlob
	pdb_name_blob: ^dxc.IBlobUtf16
	// Getting PDB and saving it to file
	results->GetOutput(.PDB, dxc.IBlob_UUID, (^rawptr)(&pdb_blob), &pdb_name_blob)

	name_ptr := (^u16)(pdb_name_blob->GetBufferPointer())
	name_len := pdb_name_blob->GetStringLength()
	name_slice := mem.slice_ptr(name_ptr, cast(int)name_len)
	name_utf8, _ := windows.utf16_to_utf8(name_slice, context.temp_allocator)

	full_pdb_path := fmt.tprintf(".\\pdbs\\%s", name_utf8)

	pdb_data := mem.slice_ptr(cast(^byte)pdb_blob->GetBufferPointer(), cast(int)pdb_blob->GetBufferSize())
	err := os.write_entire_file_from_bytes(full_pdb_path, pdb_data)
	assert(err == os.ERROR_NONE)

	}


	return output_blob, true
}

DDSFile :: struct {
	width: u32,
	height: u32,
	format: dxgi.FORMAT,
	mipmap_data: [][]byte,
}

// indexes are -1 if the view was not created
Texture :: struct {
	buffer: ^dx.IResource,
	format: dxgi.FORMAT,
	name: string,
	srv_index: int,
	dsv_index: int,
	rtv_index: int,
	uav_index: int,
	width: int,
	height: int,
	initial_view_flags: BufferViewFlags,
	opt_clear_value: Maybe(dx.CLEAR_VALUE)
}

StructuredBuffer :: struct {
	buffer: ^dx.IResource,
	fence_value: u64,
	srv_index: int,
	buffer_type: typeid,
	count: int,
	gpu_pointer: rawptr // only valid if it's on an UPLOAD heap
}

texture_get_srv_cpu_address :: proc(tex: Texture) -> dx.CPU_DESCRIPTOR_HANDLE {
	return uber_heap_get_cpu_addr(g_dx_context.cbv_srv_uav_heap, tex.srv_index)
}

texture_get_dsv_cpu_address :: proc(tex: Texture) -> dx.CPU_DESCRIPTOR_HANDLE {
	return uber_heap_get_cpu_addr(g_dx_context.dsv_heap, tex.dsv_index)
}

texture_get_rtv_cpu_address :: proc(tex: Texture) -> dx.CPU_DESCRIPTOR_HANDLE {
	return uber_heap_get_cpu_addr(g_dx_context.rtv_heap, tex.rtv_index)
}

TextureViewFlag :: enum {
	DSV,
	RTV,
	SRV,
	UAV
}

BufferViewFlags :: bit_set[TextureViewFlag]

// creates texture on default heap
// schedules an upload of data
// TODO: i think you can further simplify the API here. you can deduce things like res_flags and srv_desc by
//  the view flags.
texture_create :: proc(
	image_data: Maybe([][]byte), // slice of mipmap data. nil for texture initialized with no data
	width: u64,
	height: u32,
	format: dxgi.FORMAT,
	pool_textures : ^DXResourcePool,
	view_flags: BufferViewFlags = {},
	mip_levels: int = 1,
	texture_name : string = "",
	opt_clear_value: Maybe(dx.CLEAR_VALUE) = nil,
	tex_reuse: ^Texture = nil // reuse texture's view indices
) -> Texture {

	ct := &g_dx_context

	res_flags: dx.RESOURCE_FLAGS = {}

	if .DSV in view_flags {
		res_flags += {.ALLOW_DEPTH_STENCIL}
	}

	if .RTV in view_flags {
		res_flags += {.ALLOW_RENDER_TARGET}
	}

	if .UAV in view_flags {
		res_flags += {.ALLOW_UNORDERED_ACCESS}
	}

	texture_desc := dx.RESOURCE_DESC {
		Width = (u64)(width),
		Height = (u32)(height),
		Dimension = .TEXTURE2D,
		Layout = .UNKNOWN,
		Format = format,
		DepthOrArraySize = 1,
		MipLevels = cast(u16)mip_levels,
		SampleDesc = {Count = 1},
		Flags = res_flags
	}


	clear_val_param : ^dx.CLEAR_VALUE = nil

	if val, ok := opt_clear_value.?; ok {
		clear_val_param = &val
	}

	allocation : ^dxma.Allocation
	dxma.Allocator_CreateResource(
		pSelf = ct.dxma_allocator,
		pAllocDesc = &dxma.ALLOCATION_DESC{HeapType = .DEFAULT, ExtraHeapFlags = dx.HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES},
		pResourceDesc = &texture_desc,
		InitialResourceState = dx.RESOURCE_STATE_COMMON, // upload queue promotes the resource implicitly, so we set it here as common
		pOptimizedClearValue = clear_val_param,
		ppAllocation = &allocation,
		riidResource = nil,
		ppvResource = nil
	)

	res := dxma.Allocation_GetResource(allocation)
	append(pool_textures, cast(^dxgi.IUnknown)allocation)

	if len(texture_name) > 0 {
		texture_name_cstring := windows.utf8_to_wstring_alloc(texture_name, allocator = context.temp_allocator)
		res->SetName(texture_name_cstring)
	}

	if image_data_inner, ok := image_data.?; ok {
		dx_upload_texture_trigger(&g_upload_service, res, image_data_inner, &texture_desc)
	}

	// creating views
	// if format is typeless, then you need to pass a type to the views.

	// special case: {DSV, SRV} (depth buffer than then gets read)
	is_typeless := is_typeless(format)

	if view_flags == {.DSV, .SRV} {
		assert(is_typeless)
	}

	srv_index : int = -1

	if .SRV in view_flags {
		srv_desc := dx.SHADER_RESOURCE_VIEW_DESC {
			Format = format,
			ViewDimension = .TEXTURE2D,
			Shader4ComponentMapping = dx.ENCODE_SHADER_4_COMPONENT_MAPPING(0, 1, 2, 3), // this is the default mapping
			Texture2D = {
				MostDetailedMip = 0,
				MipLevels = cast(u32)mip_levels,
			}
		}

		if is_typeless {
			// converting typeless to srv type
			#partial switch format {
			case .R32_TYPELESS:
				srv_desc.Format = .R32_FLOAT
			case .R8_TYPELESS:
				srv_desc.Format = .R8_SNORM
			case:
				panic("format for srv not supported")
			}
		}

		if tex_reuse == nil {
			srv_index = create_srv(res, &srv_desc)
		} else {
			srv_index = tex_reuse.srv_index
			create_srv_at(res, &srv_desc, tex_reuse.srv_index)
		}
	}

	dsv_index : int = -1

	if .DSV in view_flags {
		dsv_format: dxgi.FORMAT
		#partial switch format {
		case .R32_TYPELESS:
			dsv_format = .D32_FLOAT
		case:
			panic("format for dsv not supported")
		}

		if tex_reuse == nil {
			dsv_index = create_dsv(res, dsv_format)
		} else {
			dsv_index = tex_reuse.dsv_index
			create_dsv_at(res, dsv_format, tex_reuse.dsv_index)
		}
	}

	uav_index : int = -1

	if .UAV in view_flags {
		if tex_reuse == nil {
			uav_index = create_uav(res)
		} else {
			uav_index = tex_reuse.uav_index
			create_uav_at(res, tex_reuse.uav_index)
		}
	}

	rtv_index : int = -1
	if .RTV in view_flags {
		if tex_reuse == nil {
			rtv_index = create_rtv(res)
		} else {
			rtv_index = tex_reuse.rtv_index
			create_rtv_at(res, tex_reuse.rtv_index)
		}
	}

	return Texture {
		buffer = res,
		format = format,
		name = texture_name,
		srv_index = srv_index,
		dsv_index = dsv_index,
		uav_index = uav_index,
		rtv_index = rtv_index,
		width = cast(int)width,
		height = cast(int)height,
		initial_view_flags = view_flags,
		opt_clear_value = opt_clear_value,
	}
}

texture_resize :: proc(tex: ^Texture, new_size: v2i) {

	tex.buffer->Release()

	tex.width = new_size.x
	tex.height = new_size.y

	tex_desc : dx.RESOURCE_DESC
	tex.buffer->GetDesc(&tex_desc)

	texture_create(nil, cast(u64)new_size.x, cast(u32)new_size.y, tex.format,
		&g_resources_longterm, tex.initial_view_flags, cast(int)tex_desc.MipLevels, tex.name, tex.opt_clear_value)

	tex.width = new_size.x
	tex.height = new_size.y
}

is_typeless :: proc(format: dxgi.FORMAT) -> bool {
	#partial switch format {
	case .R8_TYPELESS, .X32_TYPELESS_G8X24_UINT, .X24_TYPELESS_G8_UINT, .R32_TYPELESS, .R16_TYPELESS, .BC1_TYPELESS, .BC2_TYPELESS, .BC3_TYPELESS, .BC4_TYPELESS, .BC5_TYPELESS, .BC7_TYPELESS, .R8G8_TYPELESS, .BC6H_TYPELESS, .R24G8_TYPELESS, .R32G32_TYPELESS, .R16G16_TYPELESS, .R32G8X24_TYPELESS, .R8G8B8A8_TYPELESS, .B8G8R8A8_TYPELESS, .B8G8R8X8_TYPELESS, .R32G32B32_TYPELESS, .R10G10B10A2_TYPELESS, .R32G32B32A32_TYPELESS, .R16G16B16A16_TYPELESS, .R24_UNORM_X8_TYPELESS, .R32_FLOAT_X8X24_TYPELESS:
		return true
	case:
		return false
	}
}

// copies data to a dx UPLOAD resource. then unmaps the memory
copy_to_buffer :: proc(buffer: ^dx.IResource, data: []byte) {
	gpu_data: rawptr
	buffer->Map(0, &dx.RANGE{}, &gpu_data)
	mem.copy(gpu_data, raw_data(data), len(data))
	buffer->Unmap(0, nil)
}

copy_to_buffer_already_mapped :: proc(gpu_data: rawptr, data: []byte){
	mem.copy(gpu_data, raw_data(data), len(data))
}

copy_to_buffer_already_mapped_value :: proc(gpu_data: rawptr, data: ^$T){
	mem.copy(gpu_data, data, size_of(T))
}

// creates a SRV for the resource on the uber SRV heap
// use this after creating the uber heap
create_srv :: proc(res : ^dx.IResource, srv_desc : ^dx.SHADER_RESOURCE_VIEW_DESC = nil) -> (srv_index: int){
	ct := &g_dx_context
	ct.device->CreateShaderResourceView(res, srv_desc, uber_heap_get_next_cpu_addr(ct.cbv_srv_uav_heap))
	uber_heap_count(&ct.cbv_srv_uav_heap)
	return ct.cbv_srv_uav_heap.next_descriptor_index - 1
}

create_srv_at :: proc(res : ^dx.IResource, srv_desc : ^dx.SHADER_RESOURCE_VIEW_DESC = nil, new_srv_index: int) {
	ct := &g_dx_context
	ct.device->CreateShaderResourceView(res, srv_desc, get_descriptor_heap_cpu_address(ct.cbv_srv_uav_heap.heap, new_srv_index))
}

// creates a UAV for the resource on the uber SRV heap
// use this after creating the uber heap
create_uav :: proc(res : ^dx.IResource) -> (uav_index: int) {
	ct := &g_dx_context
	ct.device->CreateUnorderedAccessView(res, nil, nil, uber_heap_get_next_cpu_addr(ct.cbv_srv_uav_heap))
	uber_heap_count(&ct.cbv_srv_uav_heap)
	return ct.cbv_srv_uav_heap.next_descriptor_index - 1
}

create_uav_at :: proc(res : ^dx.IResource, new_uav_index: int) {
	ct := &g_dx_context
	ct.device->CreateUnorderedAccessView(res, nil, nil, get_descriptor_heap_cpu_address(ct.cbv_srv_uav_heap.heap, new_uav_index))
}

create_cbv :: proc(cbv_desc : ^dx.CONSTANT_BUFFER_VIEW_DESC) -> (srv_index: int) {
	ct := &g_dx_context
	ct.device->CreateConstantBufferView(cbv_desc, uber_heap_get_next_cpu_addr(ct.cbv_srv_uav_heap))
	uber_heap_count(&ct.cbv_srv_uav_heap)
	return ct.cbv_srv_uav_heap.next_descriptor_index - 1
}

create_dsv :: proc(res: ^dx.IResource, format: dxgi.FORMAT) -> (dsv_index: int) {

	ct := &g_dx_context
	dsv_desc := dx.DEPTH_STENCIL_VIEW_DESC {
		ViewDimension = .TEXTURE2D,
		Format = format,
	}

	ct.device->CreateDepthStencilView(res, &dsv_desc, uber_heap_get_next_cpu_addr(ct.dsv_heap))
	uber_heap_count(&ct.dsv_heap)
	return ct.dsv_heap.next_descriptor_index - 1
}

create_dsv_at :: proc(res: ^dx.IResource, format: dxgi.FORMAT, dsv_index: int) {

	ct := &g_dx_context
	dsv_desc := dx.DEPTH_STENCIL_VIEW_DESC {
		ViewDimension = .TEXTURE2D,
		Format = format,
	}

	ct.device->CreateDepthStencilView(res, &dsv_desc, get_descriptor_heap_cpu_address(ct.cbv_srv_uav_heap.heap, dsv_index))
}

create_rtv :: proc(res: ^dx.IResource) -> (rtv_index: int) {
	ct := &g_dx_context
	ct.device->CreateRenderTargetView(res, nil, uber_heap_get_next_cpu_addr(ct.rtv_heap))
	uber_heap_count(&ct.rtv_heap)
	return ct.rtv_heap.next_descriptor_index - 1
}

create_rtv_at :: proc(res: ^dx.IResource, new_rtv_index: int) {
	ct := &g_dx_context
	ct.device->CreateRenderTargetView(res, nil, get_descriptor_heap_cpu_address(ct.rtv_heap.heap, new_rtv_index))
}

close_and_execute_cmdlist :: proc() {
	ct := &g_dx_context
	ct.cmdlist->Close()
	cmdlists := [?]^dx.IGraphicsCommandList{ct.cmdlist}
	g_dx_context.queue->ExecuteCommandLists(len(cmdlists), (^^dx.ICommandList)(&cmdlists[0]))
}

// it's a vertex buffer in the upload heap.
// meant for buffers that are modified often.
// keeps it mapped
create_vertex_buffer_upload :: proc(stride_in_bytes, size_in_bytes: u32, pool: ^DXResourcePool) -> VertexBuffer {

	vb: ^dx.IResource

	// For now we'll just store stuff in an upload heap.
	// it's not optimal for most things but it's more practical for me

	resource_desc := dx.RESOURCE_DESC {
		Dimension = .BUFFER,
		Alignment = 0,
		Width = u64(size_in_bytes),
		Height = 1,
		DepthOrArraySize = 1,
		MipLevels = 1,
		Format = .UNKNOWN,
		SampleDesc = {Count = 1, Quality = 0},
		Layout = .ROW_MAJOR,
		Flags = {},
	}

	allocation : ^dxma.Allocation
	hr := dxma.Allocator_CreateResource(
		pSelf = g_dx_context.dxma_allocator,
		pAllocDesc = &dxma.ALLOCATION_DESC{HeapType = .UPLOAD},
		pResourceDesc = &resource_desc,
		InitialResourceState = dx.RESOURCE_STATE_GENERIC_READ,
		pOptimizedClearValue = nil,
		ppAllocation = &allocation,
		riidResource = nil,
		ppvResource = nil
	)
	check(hr, "Failed creating vertex buffer")
	vb = dxma.Allocation_GetResource(allocation)
	append(pool, cast(^dxgi.IUnknown)allocation)

	vbv := dx.VERTEX_BUFFER_VIEW {
		BufferLocation = vb->GetGPUVirtualAddress(),
		StrideInBytes = stride_in_bytes,
		SizeInBytes = size_in_bytes,
	}

	gpu_data: rawptr
	vb->Map(0, &dx.RANGE{}, &gpu_data)

	return VertexBuffer {
		buffer = vb,
		gpu_pointer = gpu_data,
		vbv = vbv,
		vertex_count = size_in_bytes / stride_in_bytes,
		buffer_size = size_in_bytes,
		buffer_stride = stride_in_bytes,
	}
}

// created buffer on the upload heap, and maps it. keeps it mapped
cb_upload_create :: proc(size_in_bytes: u32, pool: ^DXResourcePool, name: string = "") -> ConstantBufferUpload {

	vb: ^dx.IResource

	// For now we'll just store stuff in an upload heap.
	// it's not optimal for most things but it's more practical for me

	resource_desc := dx.RESOURCE_DESC {
		Dimension = .BUFFER,
		Alignment = 0,
		Width = u64(size_in_bytes),
		Height = 1,
		DepthOrArraySize = 1,
		MipLevels = 1,
		Format = .UNKNOWN,
		SampleDesc = {Count = 1, Quality = 0},
		Layout = .ROW_MAJOR,
		Flags = {},
	}

	allocation : ^dxma.Allocation
	hr := dxma.Allocator_CreateResource(
		pSelf = g_dx_context.dxma_allocator,
		pAllocDesc = &dxma.ALLOCATION_DESC{HeapType = .UPLOAD, ExtraHeapFlags = dx.HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES},
		pResourceDesc = &resource_desc,
		InitialResourceState = dx.RESOURCE_STATE_GENERIC_READ,
		pOptimizedClearValue = nil,
		ppAllocation = &allocation,
		riidResource = nil,
		ppvResource = nil
	)
	check(hr, "Failed creating upload buffer")
	vb = dxma.Allocation_GetResource(allocation)
	append(pool, cast(^dxgi.IUnknown)allocation)

	gpu_data: rawptr
	vb->Map(0, &dx.RANGE{}, &gpu_data)

	if len(name) > 0 {
		name_cstring := windows.utf8_to_wstring_alloc(name, allocator = context.temp_allocator)
		vb->SetName(name_cstring)
	}

	// creating our constant buffer
	srv_index := create_cbv(&dx.CONSTANT_BUFFER_VIEW_DESC{
		BufferLocation = vb->GetGPUVirtualAddress(),
		SizeInBytes = size_in_bytes
	})

	return ConstantBufferUpload {
		buffer = vb,
		gpu_pointer = gpu_data,
		buffer_size = size_in_bytes,
		srv_index = srv_index
	}
}

generate_uv_sphere :: proc(meridians: u32, parallels: u32, allocator: runtime.Allocator) -> ([]v3, []u32) {

	expected_verts := 2 + (parallels - 1) * meridians
	verts := make([dynamic]v3, 0, expected_verts, allocator)
	indices := make([dynamic]u32, 0, expected_verts * 6, allocator)

	append(&verts, v3{0.0, 1.0, 0})

	for j in 0..<parallels - 1 {
		polar : f32 = math.PI * f32(j+1) / f32(parallels)
		sp : f32 = math.sin(polar)
		cp : f32 = math.cos(polar)

		for i in 0..<meridians {

			azimuth : f32 = 2.0 * math.PI * f32(i) / f32(meridians)
			sa : f32 = math.sin(azimuth)
			ca : f32 = math.cos(azimuth)

			append(&verts, v3{sp * ca, cp, sp * sa})
		}
	}

	append(&verts, v3{0, -1, 0})

	for i in 0..<meridians {
		a : u32 = i + 1
		b : u32 = (i + 1) % meridians + 1
		// add tiangle? what is that
		// 		mesh.addTriangle(0, b, a);
		append(&indices, 0, b, a)
	}

	for j in 0..<parallels - 2 {
		aStart : u32 = j * meridians + 1
		bStart : u32 = (j + 1) * meridians + 1

		for i in 0..<meridians {
			a : u32 = aStart + i
			a1 : u32 = aStart + (i + 1) % meridians
			b: u32 = bStart + i
			b1: u32 = bStart + (i + 1) % meridians
			// add quad???? what is this
			// mesh.addQuad(a, a1, b1, b);

			append(&indices, a, a1, b1)
			append(&indices, a, b1, b)
		}

	}

	last_ring_start := (parallels - 2) * meridians + 1
	for i in 0..<meridians {
		a := last_ring_start + i
		b := last_ring_start + (i + 1) % meridians
		append(&indices, u32(len(verts) - 1), a, b)
	}

	return verts[:], indices[:]
}

// DDS file format docs: https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-header
parse_dds_file :: proc(dds_filepath: string) -> DDSFile {

	file_data, err := os.read_entire_file_from_path(dds_filepath, context.temp_allocator)
	assert(err == os.General_Error.None)
	magic_num, ok := endian.get_u32(file_data, .Little)
	assert(ok)
	assert(magic_num == 0x20534444)

	PixelFormatFlagsEnum :: enum {
		ALPHAPIXELS = 0,
		DDPF_ALPHA = 1,
		DDPF_FOURCC = 2,
		DDPF_RGB = 6,
		DDPF_YUV = 9,
		DDPF_LUMINANCE = 17,
	}

	DDSPixelFormat :: struct #packed {
		Size: u32,
		Flags: bit_set[PixelFormatFlagsEnum;u32],
		FourCC: u32,
		RGBBitCount: u32,
		RBitMask: u32,
		GBitMask: u32,
		BBitMask: u32,
		ABitMask: u32,
	}

	DDSHeader :: struct #packed {
		Size : u32,
		Flags: u32,
		Height: u32,
		Width: u32,
		PitchOrLinearSize: u32,
		Depth: u32,
		MipMapCount: u32,
		Reserved1 : [11]u32,
		pixel_format : DDSPixelFormat,
		Caps : u32,
		Caps2: u32,
		Caps3: u32,
		Caps4: u32,
		Reserved2: u32
	}

	DDSHeaderDXT10 :: struct #packed {
		dxgi_format: dxgi.FORMAT,
		resource_dimension: dx.RESOURCE_DIMENSION,
		miscFlag: u32,
		arraySize: u32,
		miscFlags2: u32,
	}

	advance :u32
	advance += size_of(magic_num)
	header := (^DDSHeader)(raw_data(file_data[advance:]))
	advance += size_of(DDSHeader)

	// assume we have the DX10 header.
	assert(.DDPF_FOURCC in header.pixel_format.Flags && header.pixel_format.FourCC == 0x30315844)

	header_dx10 := (^DDSHeaderDXT10)(raw_data(file_data[advance:]))
	advance += size_of(DDSHeaderDXT10)

	text_format : dxgi.FORMAT = header_dx10.dxgi_format
	format_num :i32 = cast(i32)header_dx10.dxgi_format 

	// It's compressed if it's a BC format
	is_compressed : bool = (format_num >= 70 && format_num <= 84) || (format_num >= 94 && format_num <= 99)

	bytes_per_block : u32 = 0
	switch format_num {
	case 70..=72: // BC1
	case 79..=81: // BC4
		bytes_per_block = 8
	case: // Rest of BCs
		bytes_per_block = 16
	}

	current_w := header.Width
	current_h := header.Height

	// TODO handle freeing

	dds_output := DDSFile {
		width = header.Width,
		height = header.Height,
		format = text_format,
		mipmap_data = make([][]byte, header.MipMapCount, context.temp_allocator)
	}

	for i in 0..<header.MipMapCount {
		level_size := get_mip_level_size(current_w, current_h, is_compressed, bytes_per_block)

		// Slice the specific mip level data
		mip_data := file_data[advance : advance + level_size]

		// Move cursor and halve dimensions for next level
		advance += level_size
		current_w = max(1, current_w >> 1)
		current_h = max(1, current_h >> 1)
		dds_output.mipmap_data[i] = slice.clone(mip_data, context.temp_allocator)
	}

	return dds_output
}

get_mip_level_size :: proc(width, height: u32, format_is_compressed: bool, bytes_per_block_or_pixel: u32) -> u32 {
	w := max(1, width)
	h := max(1, height)

	if format_is_compressed {
		// Round up to the nearest 4-pixel block
		bw := max(1, (w + 3) / 4)
		bh := max(1, (h + 3) / 4)
		return bw * bh * bytes_per_block_or_pixel // 8 or 16
	} else {
		return w * h * bytes_per_block_or_pixel // e.g., 4 for RGBA8
	}
}

texture_cache_query :: proc(model_filepath, image_name: string, format: dxgi.FORMAT, image_data: Maybe([]byte)) -> (texture_out_path: string) {

	// test if this exists already

	// lprintln("image texture cache miss. creating texture with mipmaps")
	alloc_err : runtime.Allocator_Error

	filepath_hash := hash_thing(model_filepath)
	// image_name_hash := hash_thing(image_name)

	cache_dir : string
	cache_dir, alloc_err = filepath.join({"cache", filepath_hash}, context.temp_allocator)
	assert(alloc_err == .None)

	image_name_dss := strings.concatenate({filepath.stem(image_name), ".dds"}, context.temp_allocator)
	texture_out_path, alloc_err = filepath.join({cache_dir, image_name_dss}, context.temp_allocator)
	assert(alloc_err == os.ERROR_NONE)

	// checking if it exists already
	if os.exists(texture_out_path) {
		return texture_out_path
	}

	lprintfln("converting texture %v...", image_name)

	// create dirs
	dir_err := os.make_directory_all(cache_dir)

	assert(dir_err == os.ERROR_NONE)

	input_image_dir := filepath.dir(model_filepath)
	input_image_path, alloc_err_2 := filepath.join({input_image_dir, image_name}, context.temp_allocator)

	assert(alloc_err_2 == .None)

	// Writing data to a file if a data slice was sent (necessary for texconv)
	if image_data_inner, ok := image_data.?; ok {
		err := os.write_entire_file_from_bytes(input_image_path, image_data_inner)
		assert(err == os.General_Error.None)
	}

	state, _, _, err := os.process_exec(os.Process_Desc {
		command = {
			"./texconv.exe",
			"-f", reflect.enum_string(format), // select output format
			"-m", "0", // all mip levels
			"-y", // overwrite
			"-dx10", // force adding dx10 header
			"-o", cache_dir,
			"-nologo",
			input_image_path
		}
	}, context.temp_allocator)

	// lprintln(string(stdout))
	// lprintln(string(stderr))
	assert(state.exited && state.exit_code == 0 && err == os.General_Error.None)

	lprintfln("texture %v converted correctly", image_name)

	return texture_out_path
}

hash_thing :: proc(thing: string) -> string {
	ENC_TABLE := [64]byte { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '_', }
	thing_hash_temp := hash.hash_string(.SHA256, thing, context.temp_allocator)
	thing_hash, ok := base64.encode(thing_hash_temp, ENC_TABLE, context.temp_allocator)
	assert(ok == .None)
	return thing_hash
}

check :: proc(res: dx.HRESULT, message: string = "dx call error") {
	if (res >= 0) {
		return
	}

	lprintfln("%v. Error code: %0x", message, u32(res))
	os.exit(-1)
}

// Convenience function for clearing used memory in scope
// NOTE: this assumes temp allocator is a virtual arena
@(deferred_out=virtual.arena_temp_end)
TEMP_GUARD :: #force_inline proc(loc := #caller_location) -> (virtual.Arena_Temp, runtime.Source_Code_Location) {
	arena := (^virtual.Arena)(context.temp_allocator.data)
	return virtual.arena_temp_begin(arena, loc), loc
}

// Prints to windows debug, with a lprintln() interface
lprintln :: proc(args: ..any, sep := " ") {
	str: strings.Builder
	strings.builder_init(&str, context.temp_allocator)
	fmt.sbprintln(&str, ..args, sep = sep)
	final_string_c, err := strings.to_cstring(&str)

	if err != .None {
		os.exit(1)
	}

	fmt.print(final_string_c)
	windows.OutputDebugStringA(final_string_c)
}

lprintfln :: proc(fmt_s: string, args: ..any) {
	str: strings.Builder
	strings.builder_init(&str, context.temp_allocator)
	final_string := fmt.sbprintf(&str, fmt_s, ..args, newline = true)

	final_string_c, err := strings.to_cstring(&str)

	if err != .None {
		os.exit(1)
	}

	fmt.print(final_string)
	windows.OutputDebugStringA(final_string_c)
}

get_descriptor_heap_cpu_address :: proc(
	heap: ^dx.IDescriptorHeap,
	offset: int = 0,
) -> (
	cpu_descriptor_handle: dx.CPU_DESCRIPTOR_HANDLE,
) {
	heap->GetCPUDescriptorHandleForHeapStart(&cpu_descriptor_handle)
	desc: dx.DESCRIPTOR_HEAP_DESC
	heap->GetDesc(&desc)
	increment := g_dx_context.device->GetDescriptorHandleIncrementSize(desc.Type)
	cpu_descriptor_handle.ptr += uint(cast(uint)offset * cast(uint)increment)
	return
}

proc_walk :: proc(node: Node, scene: Scene, data: rawptr)

// Walks through the scene tree and runs a proc per node
scene_walk :: proc(scene: Scene, data: rawptr, thing_to_do: proc_walk) {
	nodes := scene.nodes

	for root_node in scene.root_nodes {
		node_i := scene.nodes[root_node]

		// algorithm state
		cur_child_i: uint = 0
		depth := 0
		child_i_levels: [10]uint
		children_are_explored: bool

		for {

			if !children_are_explored {

				// do the stuff here
				thing_to_do(node_i, scene, data)
			}

			if node_i.children == nil || children_are_explored {
				children_are_explored = false

				// go to next sibling
				cur_child_i += 1

				if node_i.parent == -1 {
					break
				}

				// if there is no next sibling, go up

				node_parent := nodes[node_i.parent]

				if cur_child_i >= len(node_parent.children) {

					depth -= 1

					// if the current's node's parent doesn't have a parent, we're done!
					if node_parent.parent == -1 {
						break
					}

					// check if this one has a sibling

					node_grandparent := nodes[node_parent.parent]

					node_i = nodes[node_grandparent.children[child_i_levels[depth]]]
					cur_child_i = child_i_levels[depth]
					children_are_explored = true
					continue
				}

				node_i = nodes[node_parent.children[cur_child_i]]
			} else {
				// go to first child
				child_i_levels[depth] = cur_child_i
				cur_child_i = 0
				node_i = nodes[node_i.children[cur_child_i]]
				depth += 1
			}
		}
	}
}

load_white_texture :: proc() {
	ct := g_dx_context

	w, h, channels : c.int
	image_data := img.load("white.png", &w, &h, &channels, 4)
	defer img.image_free(image_data)
	assert(image_data != nil)

	img_data_mipmaps := make([][]byte, 1, context.temp_allocator)
	img_data_mipmaps[0] = slice.clone(slice.from_ptr(image_data, cast(int)(w * h * channels)), context.temp_allocator)

	texture := texture_create(img_data_mipmaps[:], u64(w), u32(h), .R8G8B8A8_UNORM, 
		&g_resources_longterm, {}, texture_name = "white")

	// creating srv on uber heap
	cpu_addr := get_descriptor_heap_cpu_address(ct.cbv_srv_uav_heap.heap, TEXTURE_WHITE_INDEX)
	ct.device->CreateShaderResourceView(texture.buffer, nil, cpu_addr)
}

// TODO: implement global tracking here
arena_new :: proc() -> virtual.Arena {
	arena : virtual.Arena
	alloc_err := virtual.arena_init_growing(&arena, mem.Megabyte)
	assert(alloc_err == .None)
	return arena
}

get_node_world_matrix :: proc(node: Node, scene: Scene) -> dxm {

	res: dxm = 1

	node_i := node

	for {

		boosted_t := node_i.transform_t * 1
		translation_mat := linalg.matrix4_translate_f32(boosted_t)

		boosted_s := node_i.transform_s * 1
		scale_mat := linalg.matrix4_scale_f32(boosted_s)

		rot_quat: quaternion128 = quaternion(
			w = node_i.transform_r[3],
			x = node_i.transform_r[0],
			y = node_i.transform_r[1],
			z = node_i.transform_r[2],
		)
		rot_mat: dxm = linalg.matrix4_from_quaternion_f32(rot_quat)

		// mesh_world : dxm = translation_mat * rot_mat * scale_mat
		// no rot
		mesh_world: dxm = translation_mat * rot_mat * scale_mat
		// mesh_world : dxm = scale_mat * rot_mat * translation_mat

		res = res * mesh_world
		// res = mesh_world * res
		// break

		if node_i.parent == -1 do break
		node_i = scene.nodes[node_i.parent]
	}

	return res
}

string_append :: proc(the_strs: ..string, allocator: mem.Allocator = context.allocator) -> string {
	sb: strings.Builder

	// calc len
	sb_cap : int
	for str in the_strs {
		sb_cap += len(str)
	}

	strings.builder_init_len_cap(&sb, 0, sb_cap, allocator)

	for str in the_strs {
		strings.write_string(&sb, str)
	}

	return strings.to_string(sb)
}

structured_buffer_create :: proc(
	buffer_name: string,
	pool_resource : ^DXResourcePool,
	buffer_type: typeid,
	count: int,
	buffer_data : Maybe([]byte) = nil, // nil for no initial data
	view_flags: BufferViewFlags = {.SRV},
	heap_type: dx.HEAP_TYPE = .DEFAULT
) -> StructuredBuffer {

	ct := &g_dx_context

	buffer_desc := dx.RESOURCE_DESC {
		Width = cast(u64)(reflect.size_of_typeid(buffer_type) * count),
		Height = 1,
		Dimension = .BUFFER,
		Layout = .ROW_MAJOR,
		Format = .UNKNOWN,
		DepthOrArraySize = 1,
		MipLevels = 1,
		SampleDesc = {Count = 1},
		Flags = {},
	}

	allocation : ^dxma.Allocation
	dxma.Allocator_CreateResource(
		pSelf = ct.dxma_allocator,
		pAllocDesc = &dxma.ALLOCATION_DESC{HeapType = heap_type, ExtraHeapFlags = dx.HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES},
		pResourceDesc = &buffer_desc,
		InitialResourceState = dx.RESOURCE_STATE_COMMON,
		pOptimizedClearValue = nil,
		ppAllocation = &allocation,
		riidResource = nil,
		ppvResource = nil
	)

	fence_value: u64

	res := dxma.Allocation_GetResource(allocation)

	if buffer_data_inner, ok := buffer_data.?; ok {
		assert(heap_type == .DEFAULT)
		fence_value = dx_upload_trigger(&g_upload_service, res, buffer_data_inner)
	}

	append(pool_resource, cast(^dxgi.IUnknown)allocation)

	buffer_name_cstring := windows.utf8_to_wstring_alloc(buffer_name, context.temp_allocator)
	res->SetName(buffer_name_cstring)

	// return default_res, fence_value
	srv_index : int = -1

	// probably should always be a SRV
	if .SRV in view_flags {
		srv_desc := dx.SHADER_RESOURCE_VIEW_DESC {
			Format = .UNKNOWN,
			ViewDimension = .BUFFER,
			Shader4ComponentMapping = dx.ENCODE_SHADER_4_COMPONENT_MAPPING(0, 1, 2, 3), // this is the default mapping
			Buffer = {
				FirstElement = 0,
				NumElements = u32(count),
				StructureByteStride = cast(u32)reflect.size_of_typeid(buffer_type),
				Flags = {},
			},
		}

		srv_index = create_srv(res, &srv_desc)
	}

	gpu_data: rawptr
	if heap_type == .UPLOAD {
		res->Map(0, &dx.RANGE{}, &gpu_data)
	}

	return StructuredBuffer {
		buffer = res,
		fence_value = fence_value,
		srv_index = srv_index,
		buffer_type = buffer_type,
		count = count,
		gpu_pointer = gpu_data
	}
}

set_viewport_stuff :: proc(viewport_width, viewport_height: int) {
	ct := &g_dx_context

	viewport := dx.VIEWPORT {
		Width = f32(viewport_width),
		Height = f32(viewport_height),
		MinDepth = 0,
		MaxDepth = 1,
	}

	scissor_rect := dx.RECT {
		left = 0,
		right = cast(i32)viewport_width,
		top = 0,
		bottom = cast(i32)viewport_height,
	}

	ct.cmdlist->RSSetViewports(1, &viewport)
	ct.cmdlist->RSSetScissorRects(1, &scissor_rect)
}

PSO :: struct {
	pipeline_state: ^dx.IPipelineState,
	root_signature: ^dx.IRootSignature,
	shader_filename: string,

	// false = graphics PSO. true = compute PSO
	is_compute: bool,

	// GRAPHICS PSO parameters (does not apply to Compute)
	parameters: PSOParameters,

	// index in the queue array to free the resource. i use this to swap the pointer when the pso gets hot swapped
	pso_index: int,

	// for debugging on renderdoc / debug layers
	pso_name: string,

	/// For hot swapping
	last_write_time: time.Time,
	pso_swap: ^dx.IPipelineState,

	render_proc: proc(pso: PSO)
}

create_pso_dx :: proc(parameters: PSOParameters,
	root_signature: ^dx.IRootSignature,
	vs, ps: ^dxc.IBlob,
	pso_name: string = "") -> ^dx.IPipelineState {
	default_blend_state : dx.RENDER_TARGET_BLEND_DESC
	switch parameters.blend_state {
	case .Normal:
		default_blend_state = dx.RENDER_TARGET_BLEND_DESC {
			BlendEnable = true,
			LogicOpEnable = false,
			SrcBlend = .ONE,
			DestBlend = .ZERO,
			BlendOp = .ADD,
			SrcBlendAlpha = .ONE,
			DestBlendAlpha = .ZERO,
			BlendOpAlpha = .ADD,
			LogicOp = .NOOP,
			RenderTargetWriteMask = u8(dx.COLOR_WRITE_ENABLE_ALL),
		}
	case .Off:
		default_blend_state = dx.RENDER_TARGET_BLEND_DESC {
			BlendEnable = false,
			LogicOpEnable = false,
			SrcBlend = .ONE,
			DestBlend = .ZERO,
			BlendOp = .ADD,
			SrcBlendAlpha = .ONE,
			DestBlendAlpha = .ZERO,
			BlendOpAlpha = .ADD,
			LogicOp = .NOOP,
			RenderTargetWriteMask = u8(dx.COLOR_WRITE_ENABLE_ALL),
		}
	}

	vertex_input_dx := get_dx_vertex_input(parameters.vertex_input, parameters.instance_vertex_input)

	rtv_formats := [8]dxgi.FORMAT {
		0 ..= 7 = .UNKNOWN,
	}

	for rtv_format, i in parameters.rtv_formats {
		rtv_formats[i] = rtv_format
	}

	pipeline_state_desc := dx.GRAPHICS_PIPELINE_STATE_DESC {
		pRootSignature = root_signature,
		VS = {pShaderBytecode = vs->GetBufferPointer(), BytecodeLength = vs->GetBufferSize()},
		PS = {pShaderBytecode = ps->GetBufferPointer(), BytecodeLength = ps->GetBufferSize()},
		StreamOutput = {},
		BlendState = {
			AlphaToCoverageEnable = false,
			IndependentBlendEnable = false,
			RenderTarget = {0 = default_blend_state, 1 ..= 7 = {}},
		},
		SampleMask = 0xFFFFFFFF,
		RasterizerState = {
			FillMode = parameters.fill_mode == .Solid ? .SOLID : .WIREFRAME,
			CullMode = .BACK,
			FrontCounterClockwise = cast(dx.BOOL)parameters.front_counter_clockwise,
			DepthBias = 0,
			DepthBiasClamp = 0,
			SlopeScaledDepthBias = 0,
			DepthClipEnable = true,
			MultisampleEnable = false,
			AntialiasedLineEnable = false,
			ForcedSampleCount = 0,
			ConservativeRaster = .OFF,
		},
		DSVFormat = .D32_FLOAT,
		InputLayout = {pInputElementDescs = nil, NumElements = 0},
		PrimitiveTopologyType = .TRIANGLE,
		NumRenderTargets = u32(parameters.rtv_count),
		RTVFormats = rtv_formats,
		SampleDesc = {Count = 1, Quality = 0},
	}

	if len(vertex_input_dx) > 0 {
		pipeline_state_desc.InputLayout = {pInputElementDescs = &vertex_input_dx[0], NumElements = u32(len(vertex_input_dx))}
	}

	switch parameters.cull_mode {
	case .Front:
		pipeline_state_desc.RasterizerState.CullMode = .FRONT
	case .Back:
		pipeline_state_desc.RasterizerState.CullMode = .BACK
	case .None:
		pipeline_state_desc.RasterizerState.CullMode = .NONE
	}

	pipeline_state_desc.DepthStencilState = {
		DepthEnable = cast(dx.BOOL)parameters.enable_depth, StencilEnable = false, DepthWriteMask = .ALL, DepthFunc = .LESS
	}

	pipeline_state_desc.DepthStencilState.DepthWriteMask = parameters.depth_write ? .ALL : .ZERO

	if parameters.depth_write && !parameters.enable_depth {
		panic("depth needs to be enabled for depth write.")
	}

	pso_dx : ^dx.IPipelineState
	hr := g_dx_context.device->CreateGraphicsPipelineState(&pipeline_state_desc, dx.IPipelineState_UUID, (^rawptr)(&pso_dx))
	check(hr, "Pipeline creation failed")
	if pso_name != "" {
		pso_name_wide := windows.utf8_to_wstring_alloc(pso_name, allocator = context.temp_allocator)
		pso_dx->SetName(pso_name_wide)
	}
	return pso_dx
}

create_pso_compute_dx :: proc(root_signature: ^dx.IRootSignature, cs: ^dxc.IBlob, pso_name: string = "") -> ^dx.IPipelineState {

	ct := &g_dx_context

	desc := dx.COMPUTE_PIPELINE_STATE_DESC {
		pRootSignature = root_signature,
		CS = dx.SHADER_BYTECODE{pShaderBytecode = cs->GetBufferPointer(), BytecodeLength = cs->GetBufferSize()}
	}

	compute_pso_dx : ^dx.IPipelineState

	hr := ct.device->CreateComputePipelineState(&desc, dx.IPipelineState_UUID, (^rawptr)(&compute_pso_dx))
	check(hr, "compute pipeline creation fail")

	if pso_name != "" {
		pso_name_wide := windows.utf8_to_wstring_alloc(pso_name, allocator = context.temp_allocator)
		compute_pso_dx->SetName(pso_name_wide)
	}
	return compute_pso_dx
}

pso_compute_create :: proc(shader_filename: string, render_proc: proc(pso:PSO), pso_name: string = "") -> PSO {

	ct := &g_dx_context

	cs, ok := compile_shader_compute(ct.dxc_compiler, shader_filename)
	assert(ok, "could not compile compute shader")
	defer cs->Release()

	compute_pso_dx := create_pso_compute_dx(ct.root_signatures[.Standard], cs, pso_name)
	pso_index := len(g_resources_longterm)
	append(&g_resources_longterm, compute_pso_dx)

	pso := PSO {
		pipeline_state = compute_pso_dx,
		root_signature = ct.root_signatures[.Standard],
		shader_filename = shader_filename,
		pso_index = pso_index,
		pso_name = pso_name,
		render_proc = render_proc,
		is_compute = true,
	}

	pso_hotswap_init(&pso)
	return pso
}

// checks if it should rebuild a shader
// if it should then compiles the new shader and makes a new PSO with it
pso_hotswap_watch :: proc(pso: ^PSO) {
	// watch for shader change
	game_dll_mod, game_dll_mod_err := os.last_write_time_by_name(pso.shader_filename)

	reload := false

	if game_dll_mod_err == os.ERROR_NONE && pso.last_write_time != game_dll_mod {
		pso.last_write_time = game_dll_mod
		reload = true
	}

	if reload {
		pso_reload(pso)
	}
}

pso_reload :: proc(pso: ^PSO) {
	if pso.is_compute {
		// handle releasing resources
		cs, ok := compile_shader_compute(g_dx_context.dxc_compiler, pso.shader_filename)
		if !ok {
			lprintln("Could not compile new shader!! check logs")
		} else {
			// create the new PSO to be swapped later
			pso.pso_swap = create_pso_compute_dx(pso.root_signature, cs, pso.pso_name)
			cs->Release()
			lprintfln("Shader reloaded successfully: %v", pso.shader_filename)
		}
	} else {
		// handle releasing resources
		vs, ps, ok := compile_shader(g_dx_context.dxc_compiler, pso.shader_filename)
		if !ok {
			lprintln("Could not compile new shader!! check logs")
		} else {
			// create the new PSO to be swapped later
			pso.pso_swap = create_pso_dx(pso.parameters, pso.root_signature, vs, ps, pso.pso_name)
			vs->Release()
			ps->Release()
			lprintfln("Shader reloaded successfully: %v", pso.shader_filename)
		}
	}
}

pso_hotswap_init :: proc(pso : ^PSO) {
	game_dll_mod, game_dll_mod_err := os.last_write_time_by_name(pso.shader_filename)
	if game_dll_mod_err == os.ERROR_NONE {
		pso.last_write_time = game_dll_mod
	}
}

pso_hotswap_swap :: proc(pso: ^PSO) {
	if pso.pso_swap != nil {
		pso.pipeline_state->Release()
		pso.pipeline_state = pso.pso_swap
		// replace pointer from freeing queue
		pso_pointer := &g_resources_longterm[pso.pso_index]
		pso_pointer^ = pso.pipeline_state
		pso.pso_swap = nil
	}
}

odin_base_type_to_hlsl_type :: proc(base_type: ^runtime.Type_Info) -> string {
	#partial switch element_type in base_type.variant {
	case reflect.Type_Info_Integer:
		if reflect.is_signed(base_type) {
			return "int"
		} else {
			return "uint"
		}
	case reflect.Type_Info_Float:
		return "float"
	case reflect.Type_Info_Boolean:
		assert(base_type.size == 4) // only allow b32 bools (bool in hlsl)
		return "bool"
	case: 
		panic("base element type not supported")
	}
}

odin_type_to_hlsl_core_type :: proc(ti: ^runtime.Type_Info, sb_out: ^strings.Builder) {

	#partial switch type_variant in ti.variant {
	case reflect.Type_Info_Named:
		strings.write_string(sb_out, type_variant.name)
	case reflect.Type_Info_Integer, reflect.Type_Info_Float, reflect.Type_Info_Boolean:
		strings.write_string(sb_out, odin_base_type_to_hlsl_type(ti))
	case reflect.Type_Info_Matrix:
		base_type := type_variant.elem
		strings.write_string(sb_out, odin_base_type_to_hlsl_type(base_type))
		fmt.sbprintf(sb_out, "%vx%v", type_variant.row_count, type_variant.column_count)
	case reflect.Type_Info_Array:
		base_type := type_variant.elem
		strings.write_string(sb_out, odin_base_type_to_hlsl_type(base_type))
		fmt.sbprintf(sb_out, "%v", type_variant.count)
	case:
		panic("core type not supported")
	}
}

// also enums
convert_struct_odin_to_hlsl :: proc(struct_type: typeid, allocator: runtime.Allocator) -> string {

	struct_type_info := type_info_of(struct_type)

	sb := strings.builder_make_none(allocator)
	named := struct_type_info.variant.(reflect.Type_Info_Named)

	if reflect.is_enum(struct_type_info) {

		enum_type := reflect.type_info_base(struct_type_info).variant.(reflect.Type_Info_Enum)
		enum_type_core := reflect.type_info_core(struct_type_info)

		fmt.sbprintf(&sb, "enum %v : ", named.name)
		odin_type_to_hlsl_core_type(enum_type_core, &sb)
		fmt.sbprintfln(&sb, " {{")

		for enum_name in enum_type.names {
			fmt.sbprintfln(&sb, "	%v,", enum_name)
		}
		fmt.sbprintf(&sb, "};")
	} else if reflect.is_struct(struct_type_info) {
		names := reflect.struct_field_names(struct_type)
		types := reflect.struct_field_types(struct_type)

		fmt.sbprintfln(&sb, "struct %v {{", named.name)

		// struct fields

		for type, i in types {
			field_name := names[i]
			fmt.sbprintf(&sb, "	")
			odin_type_to_hlsl_core_type(type, &sb)
			fmt.sbprintfln(&sb, " %v;", field_name)
		}

		// end
		fmt.sbprintf(&sb, "};")
	} else {
		panic("type not supported")
	}

	return strings.to_string(sb)
}


// imgui helpers
do_imgui_enum :: proc(imgui_field_name:string, enum_v: ^$T) -> (changed: bool) { 

	current_val := cast(^c.int) enum_v

	enum_type_info := type_info_of(T)
	assert(reflect.is_enum(enum_type_info))

	field_names := reflect.enum_field_names(T)

	names_c := make([]cstring, len(field_names), allocator = context.temp_allocator)

	for f_n, i in field_names {
		names_c[i] = strings.clone_to_cstring(f_n, allocator = context.temp_allocator)
	}

	imgui_field_name_c := strings.clone_to_cstring(imgui_field_name, context.temp_allocator)

	return im.ComboChar(imgui_field_name_c, current_val, raw_data(names_c), cast(i32)len(names_c))
}
