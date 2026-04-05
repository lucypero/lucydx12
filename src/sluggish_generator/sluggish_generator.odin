package sluggish_generator

import "core:c"
import "core:encoding/endian"
import "core:strconv/decimal"
import "core:strconv"
import "core:io"
import "core:math"
import "core:slice"
import "core:fmt"
import "core:os"
import tt "vendor:stb/truetype"
import "core:mem"
import "base:intrinsics"

// if you change this, the pixel shader needs to change too
TEXTURE_WIDTH :: 4096
TEXTURE_MASK :: 0xFFF
TEXTURE_SHIFT :: 12

/*
Sluggish font file format

SLUGGISH (8 bytes)
# code points (u16)
array of SluggishCodePoint
curves texture width (u16)
curves texture height (u16)
curves texture bytes (u32)
curves texture data (RGBA 32f)
bands texture width (u16)
bands texture height (u16)
bands texture bytes (u32)
bands texture data (RG 16)
*/

SluggishData :: struct {
	codepoints: []SluggishCodePoint, // g_codePoints
	curves_data: []f32, // g_curvesTexture
	bands_texture_band_offsets: []u16, // g_bandsTextureBandOffsets
	bands_texture_curve_offsets: []u16, // g_bandsTextureCurveOffsets
}

SluggishCodePoint :: struct #packed {
	codePoint: u32,
	width: u32,
	height: u32,
	bandCount: u32,
	bandDimX: u32,
	bandDimY: u32,
	bandsTexCoordX: u16,
	bandsTexCoordY: u16,
}

LucySluggishData :: struct {
	codepoints: []LucySluggishCodePoint, // g_codePoints
	inverse_scale: f32, // for inverse jacobian matrix
	curves_data: []f32, // g_curvesTexture
	bands_texture_band_offsets: []u16, // g_bandsTextureBandOffsets
	bands_texture_curve_offsets: []u16, // g_bandsTextureCurveOffsets
}

LucySluggishCodePoint :: struct {
	codePoint: u32,
	width: u32,
	height: u32,
	// ttf glyph data
	tex_top_left: [2]f32,
	tex_bottom_right: [2]f32,
	advance_width: i32,
	left_side_bearing: i32,
	// bands
	bandCount: u32,
	bandDimX: u32,
	bandDimY: u32,
	bandsTexCoordX: u16,
	bandsTexCoordY: u16,
}

Curve :: struct {
	x1, y1: f32,
	x2, y2: f32,
	x3, y3: f32,
	texelIndex: u32, // indexes the curves texture
	first: bool // 1st curve of a shape
}

build_sluggish_lucy :: proc(tt_truetype_filepath: string, band_count: u32 = 16, allocator: mem.Allocator = context.allocator) ->
	(sluggish_data: LucySluggishData, ok : bool) {
		
	tt_font: tt.fontinfo
	tt_file_data, err := os.read_entire_file(tt_truetype_filepath, context.temp_allocator)
	assert(err == os.General_Error.None)
	
	tt.InitFont(&tt_font, raw_data(tt_file_data), 0)
	
	ignored_codepoints : int
	curves := make([dynamic]Curve, allocator = allocator) // this doesn't get written to the file
	curves_data := make([dynamic]f32, allocator = allocator) // GL_RGBA32F [x1 y1 x2 y2] (g_curvesTexture)
	
	bands_texture_band_offsets := make([dynamic]u16, allocator = allocator); // GL_RG16 [curve_count band_offset] (g_bandsTextureBandOffsets)
	bands_texture_curve_offsets := make([dynamic]u16, allocator = allocator) // GL_RG16 [curve_offset curve_offset] (g_bandsTextureCurveOffsets)
	
	codepoints_dyn := make([dynamic]LucySluggishCodePoint, allocator = allocator) // g_codePoints
	
	
	for codepoint in 33..=126 {
		// process code points
		glyph_index := tt.FindGlyphIndex(&tt_font, rune(codepoint))
		vertices_mp : [^]tt.vertex
		vertex_count := tt.GetGlyphShape(&tt_font, glyph_index, &vertices_mp)
		glyph_advance_width: c.int
		glyph_left_side_bearing: c.int
		tt.GetGlyphHMetrics(&tt_font, glyph_index, &glyph_advance_width, &glyph_left_side_bearing)
		vertices: []tt.vertex = slice.from_ptr(vertices_mp, cast(int)vertex_count)
		
		// no cubic bezier support
		
		for vertex in vertices {
			vertex_type := cast(tt.vmove)vertex.type
			if vertex_type == .vcubic {
				fmt.printfln("%v this codepoint has bicubic curves. not supported", codepoint)
				ignored_codepoints += 1
				continue
			}
		}
		
		// get the glyph's visible data bounding box
		igx1, igy1, igx2, igy2: i32
		tt.GetGlyphBox(&tt_font, glyph_index, &igx1, &igy1, &igx2, &igy2)
		gx1: f32 = cast(f32)igx1
		gy1: f32 = cast(f32)igy1
		
		//
		// build temporary curve list
		//
		curve: Curve
		clear(&curves)
		
		for vertex in vertices {
			vertex_type := cast(tt.vmove)vertex.type
			
			#partial switch vertex_type {
			case .vcurve:
				curve.x1 = curve.x3;
				curve.y1 = curve.y3;
				curve.x2 = cast(f32)vertex.cx - gx1
				curve.y2 = cast(f32)vertex.cy - gy1
				curve.x3 = cast(f32)vertex.x - gx1
				curve.y3 = cast(f32)vertex.y - gy1
				append(&curves, curve)
				curve.first = false
			case .vline:
				curve.x1 = curve.x3;
				curve.y1 = curve.y3;
				curve.x3 = cast(f32)vertex.x - gx1
				curve.y3 = cast(f32)vertex.y - gy1
				curve.x2 = math.floor((curve.x1 + curve.x3) / 2.0)
				curve.y2 = math.floor((curve.y1 + curve.y3) / 2.0);
				append(&curves, curve)
				curve.first = false;
			case .vmove:
				curve.first = true;
				curve.x3 = cast(f32)vertex.x - gx1;
				curve.y3 = cast(f32)vertex.y - gy1;
			}
		}
		
		
		//
		// fix up curves where the control point is one of the endpoints
		//
		
		for &c in curves {
			if c.x2 == c.x1 && c.y2 == c.y1 ||
			   c.x2 == c.x3 && c.y2 == c.y3 {
				c.x2 = (c.x1 + c.x3) / 2.0;
				c.y2 = (c.y1 + c.y3) / 2.0;
			}
		}
		
		//
		// write curves texture
		//
		
		for &c in curves {
			// make sure we start a curve at a texel's boundary
			
			if c.first && len(curves_data) % 4 != 0 {
				to_add : int = 4 - len(curves_data) % 4
				
				for _ in 0..<to_add {
					append(&curves_data, -1)
				}
			}
			
			// make sure a curve doesn't cross a row boundary
			newRow : bool = (len(curves_data) / 4) % TEXTURE_WIDTH == TEXTURE_WIDTH - 1
			if newRow {
				to_add : int = 8 - len(curves_data) % 4
				
				for _ in 0..<to_add {
					append(&curves_data, -1)
				}
			}
			
			// [A1 B1] [C1=A2 B2] [C2=A3 B3] ...
			if c.first || newRow
			{
				c.texelIndex = cast(u32)len(curves_data) / 4
				assert(len(curves_data) % 4 == 0)
				append(&curves_data, c.x1)
				append(&curves_data, c.y1)
			}
			else
			{
				c.texelIndex = ((cast(u32)len(curves_data) / 2) - 1) / 2
			}
			
			assert(len(curves_data) % 2 == 0)
			append(&curves_data, c.x2)
			append(&curves_data, c.y2)
			append(&curves_data, c.x3)
			append(&curves_data, c.y3)
		}
		
		//
		// band
		//
		
		sizeX : u32 = 1 + cast(u32)(igx2 - igx1);
		sizeY : u32 = 1 + cast(u32)(igy2 - igy1);
		bandCount : u32 = band_count;
		if sizeX < bandCount || sizeY < bandCount
		{
			bandCount = math.min(sizeX, sizeY) / 2;
		}
		
		// idk where to put this.. worry about this later.
		
		fbandDelta : f32 = 0.0
		bandsTexelIndex : u32 = cast(u32)(len(bands_texture_band_offsets) / 2);
		
		//
		// horizontal bands
		//
		
		bandDimY : u32 = (sizeY + bandCount - 1) / bandCount;
		fbandDimY : f32 = cast(f32)bandDimY;
		
		bandMinY : f32 = -fbandDelta;
		bandMaxY : f32 = fbandDimY + fbandDelta;
		
		slice.stable_sort_by(curves[:], proc(a, b: Curve) -> bool {
			return math.max(a.x1, a.x2, a.x3) > math.max(b.x1, b.x2, b.x3)
		})
		
		for _ in 0..<bandCount {
			bandTexelOffset: u16 = cast(u16)(len(bands_texture_curve_offsets) / 2); // 2x 16 bits
			curveCount : u16 = 0;
			
			for c in curves {
				
				// reject perfectly horizontal curves
				if c.y1 == c.y2 && c.y2 == c.y3 do continue

				// reject curves that don't cross the band
				curveMinY : f32 = math.min(c.y1, c.y2, c.y3);
				curveMaxY : f32 = math.max(c.y1, c.y2, c.y3);
				if curveMinY > bandMaxY || curveMaxY < bandMinY do continue

				// push the curve offsets
				texelIndex : u32 = c.texelIndex
				curveOffsetX : u16 = cast(u16)(texelIndex % cast(u32)TEXTURE_WIDTH)
				curveOffsetY : u16 = cast(u16)(texelIndex / cast(u32)TEXTURE_WIDTH)
				
				append(&bands_texture_curve_offsets, curveOffsetX)
				append(&bands_texture_curve_offsets, curveOffsetY)
				
				curveCount += 1;
			}
			
			// @TODO: don't push more data if this band is the same as the previous one
			
			// push the horizontal band
			append(&bands_texture_band_offsets, curveCount)
			append(&bands_texture_band_offsets, bandTexelOffset)
			
			bandMinY += fbandDimY;
			bandMaxY += fbandDimY;

			if bandTexelOffset >= 0xFFFF || (len(bands_texture_curve_offsets) / 2) >= 0xFFFF
			{
				fmt.eprintfln("(horizontal bands) Too much data generated to be indexed! Try a lower band count.");
				return {}, false
			}
			
		}
		
		//
		// vertical bands
		//
		
		bandDimX : u32 = (sizeX + bandCount - 1) / bandCount;
		fbandDimX : f32 = cast(f32)bandDimX;
		
		bandMinX : f32 = -fbandDelta;
		bandMaxX : f32 = fbandDimX + fbandDelta;
		
		slice.stable_sort_by(curves[:], proc(a, b: Curve) -> bool {
			return math.max(a.y1, a.y2, a.y3) > math.max(b.y1, b.y2, b.y3)
		})
		
		for _ in 0..<bandCount {
			bandTexelOffset: u16 = cast(u16)(len(bands_texture_curve_offsets) / 2); // 2x 16 bits
			curveCount : u16 = 0;
			
			for c in curves {
				
				// reject perfectly vertical curves
				if c.x1 == c.x2 && c.x2 == c.x3 do continue

				// reject curves that don't cross the band
				curveMinX : f32 = math.min(c.x1, c.x2, c.x3);
				curveMaxX : f32 = math.max(c.x1, c.x2, c.x3);
				if curveMinX > bandMaxX || curveMaxX < bandMinX do continue

				// push the curve offsets
				texelIndex : u32 = c.texelIndex
				curveOffsetX : u16 = cast(u16)(texelIndex % cast(u32)TEXTURE_WIDTH)
				curveOffsetY : u16 = cast(u16)(texelIndex / cast(u32)TEXTURE_WIDTH)
				
				append(&bands_texture_curve_offsets, curveOffsetX)
				append(&bands_texture_curve_offsets, curveOffsetY)
				
				curveCount += 1;
			}
			
			// @TODO: don't push more data if this band is the same as the previous one
			
			// push the vertical band
			append(&bands_texture_band_offsets, curveCount)
			append(&bands_texture_band_offsets, bandTexelOffset)
			
			bandMinX += fbandDimX;
			bandMaxX += fbandDimX;

			if bandTexelOffset >= 0xFFFF || len(bands_texture_curve_offsets) / 2 >= 0xFFFF
			{
				fmt.eprintfln("(vertical bands) Too much data generated to be indexed! Try a lower band count.");
				return {}, false
			}
			
		}
		
		//
		// push the code point
		//
		
		cp := LucySluggishCodePoint {
			codePoint = cast(u32)codepoint,
			width = cast(u32)(igx2 - igx1),
			height = cast(u32)(igy2 - igy1),
			// ttf glyph data
			tex_top_left = [2]f32{cast(f32)igx1, cast(f32)igy1},
			tex_bottom_right = [2]f32{cast(f32)igx2, cast(f32)igy2},
			advance_width = glyph_advance_width,
			left_side_bearing = glyph_left_side_bearing,
			// bands
			bandCount = bandCount,
			bandDimX = bandDimX,
			bandDimY = bandDimY,
			bandsTexCoordX = cast(u16)(bandsTexelIndex % cast(u32)TEXTURE_WIDTH),
			bandsTexCoordY = cast(u16)(bandsTexelIndex / cast(u32)TEXTURE_WIDTH),
		}
		
		append(&codepoints_dyn, cp)

		if bandsTexelIndex / cast(u32)TEXTURE_WIDTH >= 0xFFFF
		{
			fmt.eprintln("Too much curve data generated! :-(");
			return {}, false
		}

		//
		// check the data's validity
		//
		
		for c in curves {
			sameRow: bool = (c.texelIndex / TEXTURE_WIDTH) == ((c.texelIndex + 1) / TEXTURE_WIDTH)
			if !sameRow
			{
				fmt.printfln("%v encoding failed! Texel indices %v and %v are not in the same row\n",
				 cast(uint)codepoint, cast(uint)c.texelIndex, cast(uint)c.texelIndex + 1);
			}
		}
	}
	
	font_size_pixels :: 22
	font_scale := tt.ScaleForPixelHeight(&tt_font, font_size_pixels)
	
	return LucySluggishData {
		codepoints = codepoints_dyn[:],
		curves_data = curves_data[:],
		bands_texture_band_offsets = bands_texture_band_offsets[:],
		bands_texture_curve_offsets = bands_texture_curve_offsets[:],
		inverse_scale = 1.0 / font_scale
	}, true
}

build_sluggish :: proc(tt_truetype_filepath: string, band_count: u32 = 16, allocator: mem.Allocator = context.allocator) ->
	(sluggish_data: SluggishData, ok : bool) {
	
	tt_font: tt.fontinfo
	tt_file_data, err := os.read_entire_file(tt_truetype_filepath, context.temp_allocator)
	assert(err == os.General_Error.None)
	
	tt.InitFont(&tt_font, raw_data(tt_file_data), 0)
	
	ignored_codepoints : int
	curves := make([dynamic]Curve, allocator = allocator) // this doesn't get written to the file
	curves_data := make([dynamic]f32, allocator = allocator) // GL_RGBA32F [x1 y1 x2 y2] (g_curvesTexture)
	
	bands_texture_band_offsets := make([dynamic]u16, allocator = allocator); // GL_RG16 [curve_count band_offset] (g_bandsTextureBandOffsets)
	bands_texture_curve_offsets := make([dynamic]u16, allocator = allocator) // GL_RG16 [curve_offset curve_offset] (g_bandsTextureCurveOffsets)
	
	codepoints_dyn := make([dynamic]SluggishCodePoint, allocator = allocator) // g_codePoints
	
	for codepoint in 33..=126 {
		// process code points
		glyph_index := tt.FindGlyphIndex(&tt_font, rune(codepoint))
		vertices_mp : [^]tt.vertex
		vertex_count := tt.GetGlyphShape(&tt_font, glyph_index, &vertices_mp)
		vertices: []tt.vertex = slice.from_ptr(vertices_mp, cast(int)vertex_count)
		
		// no cubic bezier support
		
		for vertex in vertices {
			vertex_type := cast(tt.vmove)vertex.type
			if vertex_type == .vcubic {
				fmt.printfln("%v this codepoint has bicubic curves. not supported", codepoint)
				ignored_codepoints += 1
				continue
			}
		}
		
		// get the glyph's visible data bounding box
		igx1, igy1, igx2, igy2: i32
		tt.GetGlyphBox(&tt_font, glyph_index, &igx1, &igy1, &igx2, &igy2)
		gx1: f32 = cast(f32)igx1
		gy1: f32 = cast(f32)igy1
		
		//
		// build temporary curve list
		//
		curve: Curve
		clear(&curves)
		
		for vertex in vertices {
			vertex_type := cast(tt.vmove)vertex.type
			
			#partial switch vertex_type {
			case .vcurve:
				curve.x1 = curve.x3;
				curve.y1 = curve.y3;
				curve.x2 = cast(f32)vertex.cx - gx1
				curve.y2 = cast(f32)vertex.cy - gy1
				curve.x3 = cast(f32)vertex.x - gx1
				curve.y3 = cast(f32)vertex.y - gy1
				append(&curves, curve)
				curve.first = false
			case .vline:
				curve.x1 = curve.x3;
				curve.y1 = curve.y3;
				curve.x3 = cast(f32)vertex.x - gx1
				curve.y3 = cast(f32)vertex.y - gy1
				curve.x2 = math.floor((curve.x1 + curve.x3) / 2.0)
				curve.y2 = math.floor((curve.y1 + curve.y3) / 2.0);
				append(&curves, curve)
				curve.first = false;
			case .vmove:
				curve.first = true;
				curve.x3 = cast(f32)vertex.x - gx1;
				curve.y3 = cast(f32)vertex.y - gy1;
			}
		}
		
		
		//
		// fix up curves where the control point is one of the endpoints
		//
		
		for &c in curves {
			if c.x2 == c.x1 && c.y2 == c.y1 ||
			   c.x2 == c.x3 && c.y2 == c.y3 {
				c.x2 = (c.x1 + c.x3) / 2.0;
				c.y2 = (c.y1 + c.y3) / 2.0;
			}
		}
		
		//
		// write curves texture
		//
		
		for &c in curves {
			// make sure we start a curve at a texel's boundary
			
			if c.first && len(curves_data) % 4 != 0 {
				to_add : int = 4 - len(curves_data) % 4
				
				for _ in 0..<to_add {
					append(&curves_data, -1)
				}
			}
			
			// make sure a curve doesn't cross a row boundary
			newRow : bool = (len(curves_data) / 4) % TEXTURE_WIDTH == TEXTURE_WIDTH - 1
			if newRow {
				to_add : int = 8 - len(curves_data) % 4
				
				for _ in 0..<to_add {
					append(&curves_data, -1)
				}
			}
			
			// [A1 B1] [C1=A2 B2] [C2=A3 B3] ...
			if c.first || newRow
			{
				c.texelIndex = cast(u32)len(curves_data) / 4
				assert(len(curves_data) % 4 == 0)
				append(&curves_data, c.x1)
				append(&curves_data, c.y1)
			}
			else
			{
				c.texelIndex = ((cast(u32)len(curves_data) / 2) - 1) / 2
			}
			
			assert(len(curves_data) % 2 == 0)
			append(&curves_data, c.x2)
			append(&curves_data, c.y2)
			append(&curves_data, c.x3)
			append(&curves_data, c.y3)
		}
		
		//
		// band
		//
		
		sizeX : u32 = 1 + cast(u32)(igx2 - igx1);
		sizeY : u32 = 1 + cast(u32)(igy2 - igy1);
		bandCount : u32 = band_count;
		if sizeX < bandCount || sizeY < bandCount
		{
			bandCount = math.min(sizeX, sizeY) / 2;
		}
		
		// idk where to put this.. worry about this later.
		
		fbandDelta : f32 = 0.0
		bandsTexelIndex : u32 = cast(u32)(len(bands_texture_band_offsets) / 2);
		
		//
		// horizontal bands
		//
		
		bandDimY : u32 = (sizeY + bandCount - 1) / bandCount;
		fbandDimY : f32 = cast(f32)bandDimY;
		
		bandMinY : f32 = -fbandDelta;
		bandMaxY : f32 = fbandDimY + fbandDelta;
		
		slice.stable_sort_by(curves[:], proc(a, b: Curve) -> bool {
			return math.max(a.x1, a.x2, a.x3) > math.max(b.x1, b.x2, b.x3)
		})
		
		for _ in 0..<bandCount {
			bandTexelOffset: u16 = cast(u16)(len(bands_texture_curve_offsets) / 2); // 2x 16 bits
			curveCount : u16 = 0;
			
			for c in curves {
				
				// reject perfectly horizontal curves
				if c.y1 == c.y2 && c.y2 == c.y3 do continue

				// reject curves that don't cross the band
				curveMinY : f32 = math.min(c.y1, c.y2, c.y3);
				curveMaxY : f32 = math.max(c.y1, c.y2, c.y3);
				if curveMinY > bandMaxY || curveMaxY < bandMinY do continue

				// push the curve offsets
				texelIndex : u32 = c.texelIndex
				curveOffsetX : u16 = cast(u16)(texelIndex % cast(u32)TEXTURE_WIDTH)
				curveOffsetY : u16 = cast(u16)(texelIndex / cast(u32)TEXTURE_WIDTH)
				
				append(&bands_texture_curve_offsets, curveOffsetX)
				append(&bands_texture_curve_offsets, curveOffsetY)
				
				curveCount += 1;
			}
			
			// @TODO: don't push more data if this band is the same as the previous one
			
			// push the horizontal band
			append(&bands_texture_band_offsets, curveCount)
			append(&bands_texture_band_offsets, bandTexelOffset)
			
			bandMinY += fbandDimY;
			bandMaxY += fbandDimY;

			if bandTexelOffset >= 0xFFFF || (len(bands_texture_curve_offsets) / 2) >= 0xFFFF
			{
				fmt.eprintfln("(horizontal bands) Too much data generated to be indexed! Try a lower band count.");
				return {}, false
			}
			
		}
		
		//
		// vertical bands
		//
		
		bandDimX : u32 = (sizeX + bandCount - 1) / bandCount;
		fbandDimX : f32 = cast(f32)bandDimX;
		
		bandMinX : f32 = -fbandDelta;
		bandMaxX : f32 = fbandDimX + fbandDelta;
		
		slice.stable_sort_by(curves[:], proc(a, b: Curve) -> bool {
			return math.max(a.y1, a.y2, a.y3) > math.max(b.y1, b.y2, b.y3)
		})
		
		for _ in 0..<bandCount {
			bandTexelOffset: u16 = cast(u16)(len(bands_texture_curve_offsets) / 2); // 2x 16 bits
			curveCount : u16 = 0;
			
			for c in curves {
				
				// reject perfectly vertical curves
				if c.x1 == c.x2 && c.x2 == c.x3 do continue

				// reject curves that don't cross the band
				curveMinX : f32 = math.min(c.x1, c.x2, c.x3);
				curveMaxX : f32 = math.max(c.x1, c.x2, c.x3);
				if curveMinX > bandMaxX || curveMaxX < bandMinX do continue

				// push the curve offsets
				texelIndex : u32 = c.texelIndex
				curveOffsetX : u16 = cast(u16)(texelIndex % cast(u32)TEXTURE_WIDTH)
				curveOffsetY : u16 = cast(u16)(texelIndex / cast(u32)TEXTURE_WIDTH)
				
				append(&bands_texture_curve_offsets, curveOffsetX)
				append(&bands_texture_curve_offsets, curveOffsetY)
				
				curveCount += 1;
			}
			
			// @TODO: don't push more data if this band is the same as the previous one
			
			// push the vertical band
			append(&bands_texture_band_offsets, curveCount)
			append(&bands_texture_band_offsets, bandTexelOffset)
			
			bandMinX += fbandDimX;
			bandMaxX += fbandDimX;

			if bandTexelOffset >= 0xFFFF || len(bands_texture_curve_offsets) / 2 >= 0xFFFF
			{
				fmt.eprintfln("(vertical bands) Too much data generated to be indexed! Try a lower band count.");
				return {}, false
			}
			
		}
		
		//
		// push the code point
		//
		
		cp := SluggishCodePoint {
			codePoint = cast(u32)codepoint,
			width = cast(u32)(igx2 - igx1),
			height = cast(u32)(igy2 - igy1),
			bandCount = bandCount,
			bandDimX = bandDimX,
			bandDimY = bandDimY,
			bandsTexCoordX = cast(u16)(bandsTexelIndex % cast(u32)TEXTURE_WIDTH),
			bandsTexCoordY = cast(u16)(bandsTexelIndex / cast(u32)TEXTURE_WIDTH),
		}
		
		append(&codepoints_dyn, cp)

		if bandsTexelIndex / cast(u32)TEXTURE_WIDTH >= 0xFFFF
		{
			fmt.eprintln("Too much curve data generated! :-(");
			return {}, false
		}

		//
		// check the data's validity
		//
		
		for c in curves {
			sameRow: bool = (c.texelIndex / TEXTURE_WIDTH) == ((c.texelIndex + 1) / TEXTURE_WIDTH)
			if !sameRow
			{
				fmt.printfln("%v encoding failed! Texel indices %v and %v are not in the same row\n",
				 cast(uint)codepoint, cast(uint)c.texelIndex, cast(uint)c.texelIndex + 1);
			}
		}
	}
	
	return SluggishData {
		codepoints = codepoints_dyn[:],
		curves_data = curves_data[:],
		bands_texture_band_offsets = bands_texture_band_offsets[:],
		bands_texture_curve_offsets = bands_texture_curve_offsets[:],
	}, true
}

build_sluggish_to_file :: proc(tt_truetype_filepath: string, out_file:string, band_count: u32 = 16) ->
		(ok : bool) {
			
	sluggish_data, s_ok := build_sluggish(tt_truetype_filepath, band_count, allocator = context.temp_allocator)
	assert(s_ok)
	
	
	// fix up the bands' texel offsets first
	bandsTexTexels: u32 = cast(u32)(len(sluggish_data.bands_texture_band_offsets) + len(sluggish_data.bands_texture_curve_offsets)) / 2;
	bandHeaderTexels: u16 = cast(u16)(len(sluggish_data.bands_texture_band_offsets) / 2);
	
	for i := 1; i < len(sluggish_data.bands_texture_band_offsets); i += 2 {
		sluggish_data.bands_texture_band_offsets[i] += bandHeaderTexels;
		if cast(u32)sluggish_data.bands_texture_band_offsets[i] >= bandsTexTexels {
			fmt.eprintln("Too much data generated to be indexed! Try a lower band count.\n");
			return false
		}
	}
	
	file, err := os.open(out_file, {.Create, .Write, .Trunc})
	assert(err == os.ERROR_NONE)
	stream := os.to_writer(file)
	io_err : io.Error
	
	_, io_err = io.write_string(stream, "SLUGGISH")
	assert(io_err == .None)
	
	write_num(stream, cast(u16)len(sluggish_data.codepoints))
	io.write_slice(stream, sluggish_data.codepoints)
	
	curvesTexWidth: u16 = TEXTURE_WIDTH
	curvesTexTexels: u32 = cast(u32)len(sluggish_data.curves_data) / 4
	curvesTexBytes: u32 = cast(u32)len(sluggish_data.curves_data) * cast(u32)size_of(sluggish_data.curves_data[0])
	curvesTexHeight: u16 = cast(u16)((curvesTexTexels + cast(u32)curvesTexWidth - 1) / cast(u32)curvesTexWidth)
	
	write_num(stream, curvesTexWidth)
	write_num(stream, curvesTexHeight)
	write_num(stream, curvesTexBytes)
	io.write_slice(stream, sluggish_data.curves_data)
	
	bandsTexWidth: u16 = TEXTURE_WIDTH;
	
	bandsTexBytes: u32 = bandsTexTexels * cast(u32)size_of(u16) * 2;
	bandsTexHeight: u16 = cast(u16)((bandsTexTexels + cast(u32)bandsTexWidth - 1) / cast(u32)bandsTexWidth);
	
	write_num(stream, bandsTexWidth)
	write_num(stream, bandsTexHeight)
	write_num(stream, bandsTexBytes)
	io.write_slice(stream, sluggish_data.bands_texture_band_offsets)
	io.write_slice(stream, sluggish_data.bands_texture_curve_offsets)
	
	io_err = io.close(stream)
	assert(io_err == .None)
	
	return true
}

// NOTE: this probably shouldn't be a generic proc.
// I just wanted to do a generic proc.
@(private="file")
write_num :: proc(w: io.Writer, i: $T, n_written: ^int = nil) -> (n: int, err: io.Error) 
	where intrinsics.type_is_integer(T) {
		
	buf : [size_of(T)]byte
	
	when T == u32 {
		endian.put_u32(buf[:], .Little, i)
	} else when T == u16 {
		endian.put_u16(buf[:], .Little, i)
	}
	
	return io.write(w, buf[:])
}

// Run this command to compare files
// E:\dev\Sluggish: fc arial.sluggish ..\dx12\fonts\sluggish\arial.sluggish
// It should say:
// Comparing files arial.sluggish and ..\DX12\FONTS\SLUGGISH\ARIAL.SLUGGISH
// FC: no differences encountered
