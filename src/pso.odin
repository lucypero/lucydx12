package main

import "core:thread"
import "core:sync/chan"
import "core:mem/virtual"
import "core:debug/trace"
import "core:reflect"
import img "vendor:stb/image"
import "core:fmt"
import "core:mem"
import "core:slice"
import "core:os"
import "core:strings"
import "core:sys/windows"
import "core:time"
import dx "vendor:directx/d3d12"
import dxgi "vendor:directx/dxgi"
import sdl "vendor:sdl2"
import "core:c"
import "core:math"
import "core:math/linalg"
import "base:runtime"
import "core:math/rand"
import "core:sync"
import dxc "vendor:directx/dxc"
import "core:prof/spall"
import dxma "../libs/odin-d3d12ma"
import sa "core:container/small_array"

// vertex buffer
Gizmos_Vertex_IA :: struct {
	pos: v3 `POSITION`,
}

// All our PSO's
PSOName :: enum {
	GBuffer_Pass,
	Lighting_Pass,
	Shadowmap,
	Gizmos,
	Text
}

PSO :: struct {
	pipeline_state: ^dx.IPipelineState,
	root_signature: ^dx.IRootSignature,
	shader_filename: string,

	parameters: PSOParameters,

	// index in the queue array to free the resource. i use this to swap the pointer when the pso gets hot swapped
	pso_index: int,

	// for debugging on renderdoc / debug layers
	pso_name: string,

	/// For hot swapping
	last_write_time: time.Time,
	pso_swap: ^dx.IPipelineState,
}

create_pso_dx :: proc(shader_filename: string, parameters: PSOParameters,
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
		0 ..< 7 = .UNKNOWN,
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
			RenderTarget = {0 = default_blend_state, 1 ..< 7 = {}},
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

	if parameters.enable_depth {
		pipeline_state_desc.DepthStencilState = {
			DepthEnable = true, StencilEnable = false, DepthWriteMask = .ALL, DepthFunc = .LESS
		}
	} else {
		pipeline_state_desc.DepthStencilState = {
			DepthEnable = false, StencilEnable = false
		}
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
		lprintln("Recompiling shader...")
		// handle releasing resources
		vs, ps, ok := compile_shader(g_dx_context.dxc_compiler, pso.shader_filename)
		if !ok {
			lprintln("Could not compile new shader!! check logs")
		} else {
			// create the new PSO to be swapped later
			pso.pso_swap = create_pso_dx(pso.shader_filename, pso.parameters, pso.root_signature, vs, ps, pso.pso_name)
			vs->Release()
			ps->Release()
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
