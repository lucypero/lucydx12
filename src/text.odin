package main

import "core:slice"
import "core:mem/virtual"
import "core:sys/windows"
import "core:time"
import dx "vendor:directx/d3d12"
import dxgi "vendor:directx/dxgi"
import sdl "vendor:sdl2"
import "core:math"
import "core:math/linalg"
import "base:runtime"
import dxc "vendor:directx/dxc"
import dxma "../libs/odin-d3d12ma"
import sg "sluggish_generator"

TextVertexInput :: struct #packed {
	// attrib 0
	pos: v2, // object-space vertex coords
	normal: v2, // object space normal vector
	// attrib 1
	tex: v2, // em-space sample coordinates
	tex_band_location: u32, // location of glyph data in band texture (interpreted as integer):
	tex_max_band_index_flags: u32, // max band indexes and flags (interpreted as integer):
	// attrib 2
	jac: v4, // inverse Jacobian matrix entries (00, 01, 10, 11).
	// attrib 3
	bnd: v4, // (band scale x, band scale y, band offset x, band offset y).
	// attrib 4
	col: v4 // vertex color (red, green, blue, alpha).
}

TextState :: struct {
	sluggish_data: sg.LucySluggishData,
	vertex_buffer: VertexBuffer,
	param_struct_buffer: ConstantBufferUpload,
	curve_texture: Texture,
	band_texture: Texture,
}

ParamStruct :: struct #align(256) {
	slug_matrix: dxm, // The four rows of the MVP matrix.
	slug_viewport: v4 // The viewport dimensions, in pixels.
}

populate_text_vertex_buffer :: proc() {

}

// one quad. test
VERTEX_COUNT :: 6

// sluggish file text
text_init :: proc() {
	ct := &g_dx_context
	sluggish_in :: "fonts/ttf/arial.ttf"
	sluggish_out :: "fonts/sluggish/arial.sluggish"
	sluggish_data, ok := sg.build_sluggish_lucy("fonts/ttf/arial.ttf", band_count = 16, allocator = context.allocator)
	assert(ok)

	param_struct_buffer := create_constant_buffer_upload(size_of(ParamStruct), &g_resources_longterm, name = "text constants cbv")

	// creating textures


	// curve texture
	curve_texture_data := make([][]byte, 1)
	curve_texture_data[0] = slice.to_bytes(sluggish_data.curves_data)

	TEX_WIDTH :: 4096
	CURVE_TEXEL_SIZE :: 2 * 4

	total_texels := len(curve_texture_data[0]) / CURVE_TEXEL_SIZE
	tex_height : u32 = u32((total_texels + TEX_WIDTH - 1) / TEX_WIDTH)

	curve_texture := create_texture_with_data_new(curve_texture_data, TEX_WIDTH, tex_height,
		.R16G16B16A16_FLOAT, &g_resources_longterm, "curve texture for slug")

	// band texture - ????
	// the Sluggish format is sooo BAD!!
	// band_texture := create_texture_with_data_new()

	band_texture_data := make([][]byte, 1)
	band_texture_data_inner := make([]byte, len(sluggish_data.bands_texture_band_offsets) * 2 + len(sluggish_data.bands_texture_curve_offsets) * 2)
	copy(band_texture_data_inner, slice.to_bytes(sluggish_data.bands_texture_band_offsets))
	copy(band_texture_data_inner[len(sluggish_data.bands_texture_band_offsets):], slice.to_bytes(sluggish_data.bands_texture_curve_offsets))
	band_texture_data[0] = band_texture_data_inner

	BAND_TEXEL_SIZE :: 2 * 2
	total_texels = len(band_texture_data_inner) / BAND_TEXEL_SIZE
	tex_height = u32((total_texels + TEX_WIDTH - 1) / TEX_WIDTH)

	band_texture := create_texture_with_data_new(band_texture_data, TEX_WIDTH, tex_height,
		.R16G16_UINT, &g_resources_longterm, "band texture for slug")

	ct.text_state = TextState {
		sluggish_data = sluggish_data,
		vertex_buffer = create_vertex_buffer_upload(size_of(TextVertexInput), VERTEX_COUNT * size_of(TextVertexInput), &g_resources_longterm),
		param_struct_buffer = param_struct_buffer,
		curve_texture = curve_texture,
		band_texture = band_texture,
	}
}

pso_text_render :: proc() {
	ct := &g_dx_context

	// Writing vertex data here
	{
		vertex_data_test := make([]TextVertexInput, VERTEX_COUNT, allocator = context.temp_allocator)

		sd := &ct.text_state.sluggish_data
		a_glyph := &sd.codepoints[2]

		tx0 := a_glyph.tex_top_left.x
		ty0 := a_glyph.tex_top_left.y

		tx1 := a_glyph.tex_bottom_right.x
		ty1 := a_glyph.tex_bottom_right.y

		vertices := [4]TextVertexInput {
			TextVertexInput { // top left
				pos = {0, 0},
				normal = {-1, -1},
				tex = {tx0, ty1},
			},
			TextVertexInput { // top right
				pos = {cast(f32)a_glyph.width, 0},
				normal = {1, -1},
				tex = {tx1, ty1},
			},
			TextVertexInput { // bottom left
				pos = {0, cast(f32)a_glyph.height},
				normal = {-1, 1},
				tex = {tx0, ty0},
			},
			TextVertexInput { // bottom right
				pos = {cast(f32)a_glyph.width, cast(f32)a_glyph.height},
				normal = {1, 1},
				tex = {tx1, ty0},
			},
		}

		// tex flags

		max_x: u32 = a_glyph.bandCount - 1
		max_y: u32 = a_glyph.bandCount - 1
		e_flag: u32 = 0

		// Shift max_y up by 16, and the E flag up by 28. Combine them.
		tex_max_band_index_flags := (e_flag << 28) | (max_y << 16) | (max_x & 0xFF)

		// bnd
		em_width  := f32(a_glyph.width)
		em_height := f32(a_glyph.height)

		bnd_scale_x := f32(a_glyph.bandCount) / em_width
		bnd_scale_y := -f32(a_glyph.bandCount) / em_height

		bnd_offset_x := f32(-a_glyph.tex_top_left.x) * bnd_scale_x
		bnd_offset_y := f32(-a_glyph.tex_top_left.y) * bnd_scale_y

		bnd := [4]f32{bnd_scale_x, bnd_scale_y, bnd_offset_x, bnd_offset_y}

		for &vertex, i in vertex_data_test {

			vertex_v_data : TextVertexInput

			switch i {
				// first triangle
			case 0: // top left
				vertex_v_data = vertices[0]
			case 1: // top right
				vertex_v_data = vertices[1]
			case 2: // bottom left
				vertex_v_data = vertices[2]
				// second triangle
			case 3: // top right
				vertex_v_data = vertices[1]
			case 4: // bottom right
				vertex_v_data = vertices[3]
			case 5: // bottom left
				vertex_v_data = vertices[2]
			}

			vertex = TextVertexInput {
				pos = vertex_v_data.pos * 0.2 + {30, 500},
				normal = vertex_v_data.normal,
				tex = vertex_v_data.tex,
				tex_band_location = cast(u32)a_glyph.bandsTexCoordY << 16 | cast(u32)a_glyph.bandsTexCoordX & 0xFFFF,
				tex_max_band_index_flags = tex_max_band_index_flags,
				jac = v4{sd.inverse_scale, 0.0, 0.0, -sd.inverse_scale},
				bnd = bnd,
				col = v4{1, 0, 0, 1}
			}
		}

		copy_to_buffer_already_mapped(ct.text_state.vertex_buffer.gpu_pointer, slice.to_bytes(vertex_data_test))
	}

	// updating cbv
	{

		get_text_view_projection :: proc(cam: Camera) -> (dxm, dxm) {
			view := linalg.MATRIX4F32_IDENTITY
			proj := linalg.matrix_ortho3d_f32(0, WINDOW_WIDTH, WINDOW_HEIGHT, 0, -10, 10, true)

			return view, proj
		}

		view, projection := get_text_view_projection(cur_cam)

		ps := ParamStruct {
			slug_matrix = projection * view,
			slug_viewport = {WINDOW_WIDTH, WINDOW_HEIGHT, 0, 0}
		}
		ps.slug_matrix = linalg.transpose(projection * view)

		copy_to_buffer_already_mapped_value(ct.text_state.param_struct_buffer.gpu_pointer, &ps)
	}

	ct.cmdlist->SetPipelineState(ct.psos[.Text].pipeline_state)
	ct.cmdlist->SetDescriptorHeaps(1, &ct.cbv_srv_uav_heap)
	ct.cmdlist->SetGraphicsRootSignature(ct.psos[.Text].root_signature)

	set_viewport_stuff()

	// Setting render targets. Clearing RTV.
	{
		rtv_handles := [1]dx.CPU_DESCRIPTOR_HANDLE {
			get_descriptor_heap_cpu_address(ct.swapchain_rtv_descriptor_heap, cast(uint)ct.frame_index),
		}

		ct.cmdlist->OMSetRenderTargets(1, &rtv_handles[0], false, nil)
	}

	// draw call
	ct.cmdlist->IASetPrimitiveTopology(.TRIANGLELIST)

	// binding vertex buffer view and instance buffer view
	vertex_buffers_views := [?]dx.VERTEX_BUFFER_VIEW{ct.text_state.vertex_buffer.vbv}
	ct.cmdlist->IASetVertexBuffers(0, len(vertex_buffers_views), &vertex_buffers_views[0])
	ct.cmdlist->DrawInstanced(ct.text_state.vertex_buffer.vertex_count, 1, 0, 0)
}


pso_text_create_pipeline_state :: proc(root_signature: ^dx.IRootSignature, vs, ps: ^dxc.IBlob) -> ^dx.IPipelineState {

	ct := &g_dx_context

	vertex_format := [?]dx.INPUT_ELEMENT_DESC {
		{
			SemanticName = "ATTRIB",
			SemanticIndex = 0,
			Format = .R32G32B32A32_FLOAT,
			AlignedByteOffset = dx.APPEND_ALIGNED_ELEMENT,
			InputSlotClass = .PER_VERTEX_DATA,
		},
		{
			SemanticName = "TEXCOORD", // tex.xy
			SemanticIndex = 0,
			Format = .R32G32_FLOAT,
			AlignedByteOffset = dx.APPEND_ALIGNED_ELEMENT,
			InputSlotClass = .PER_VERTEX_DATA,
		},
		{
			SemanticName = "TEXCOORD", // tex.zw
			SemanticIndex = 1,
			Format = .R32G32_UINT,
			AlignedByteOffset = dx.APPEND_ALIGNED_ELEMENT,
			InputSlotClass = .PER_VERTEX_DATA,
		},
		{
			SemanticName = "ATTRIB",
			SemanticIndex = 1,
			Format = .R32G32B32A32_FLOAT,
			AlignedByteOffset = dx.APPEND_ALIGNED_ELEMENT,
			InputSlotClass = .PER_VERTEX_DATA,
		},
		{
			SemanticName = "ATTRIB",
			SemanticIndex = 2,
			Format = .R32G32B32A32_FLOAT,
			AlignedByteOffset = dx.APPEND_ALIGNED_ELEMENT,
			InputSlotClass = .PER_VERTEX_DATA,
		},
		{
			SemanticName = "ATTRIB",
			SemanticIndex = 3,
			Format = .R32G32B32A32_FLOAT,
			AlignedByteOffset = dx.APPEND_ALIGNED_ELEMENT,
			InputSlotClass = .PER_VERTEX_DATA,
		},
	}

	default_blend_state := dx.RENDER_TARGET_BLEND_DESC {
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

	pipeline_state_desc := dx.GRAPHICS_PIPELINE_STATE_DESC {
		pRootSignature = root_signature,
		VS = {pShaderBytecode = vs->GetBufferPointer(), BytecodeLength = vs->GetBufferSize()},
		PS = {pShaderBytecode = ps->GetBufferPointer(), BytecodeLength = ps->GetBufferSize()},
		StreamOutput = {},
		BlendState = {
			AlphaToCoverageEnable = false,
			IndependentBlendEnable = false,
			RenderTarget = {0 = default_blend_state, 1..< 7 = {}}
		},
		SampleMask = 0xFFFFFFFF,
		RasterizerState = {
			FillMode = .SOLID,
			CullMode = .NONE,
			// true because we flipped positions and normals in gltf to convert between coord systems.
			FrontCounterClockwise = true, 
			DepthBias = 0,
			DepthBiasClamp = 0,
			SlopeScaledDepthBias = 0,
			DepthClipEnable = true,
			MultisampleEnable = false,
			AntialiasedLineEnable = false,
			ForcedSampleCount = 0,
			ConservativeRaster = .OFF,
		},
		// enabling depth testing
		DepthStencilState = {DepthEnable = false, StencilEnable = false},
		InputLayout = {pInputElementDescs = &vertex_format[0], NumElements = u32(len(vertex_format))},
		PrimitiveTopologyType = .TRIANGLE,
		NumRenderTargets = GBUFFER_COUNT,
		RTVFormats = {0 = .R8G8B8A8_UNORM, 1 ..< 7 = .UNKNOWN},
		// DSVFormat = .D32_FLOAT,
		SampleDesc = {Count = 1, Quality = 0},
	}

	pso: ^dx.IPipelineState

	ct.device->CreateGraphicsPipelineState(&pipeline_state_desc, dx.IPipelineState_UUID, (^rawptr)(&pso))
	pso->SetName("PSO for text")

	return pso
}
