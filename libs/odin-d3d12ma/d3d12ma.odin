// Copyright (c) 2025 Mohit Sethi
// Licensed under the MIT license (https://opensource.org/license/mit/)
package d3d12ma

import "core:c"
import "vendor:directx/d3d12"
import "vendor:directx/dxgi"
import "core:sys/windows"
_ :: c

foreign import d3d12ma "d3d12ma.lib"

COMPILER_MSVC :: 1

OS_WINDOWS :: 1

ARCH_X64 :: 1

ARCH_X86 :: 0

ARCH_ARM64 :: 0

ARCH_ARM32 :: 0

COMPILER_MSVC_YEAR :: 2019

COMPILER_CLANG :: 0

OS_MAC :: 0

OS_LINUX :: 0

COMPILER_GCC :: 0

ARCH_64BIT :: 1

ARCH_32BIT :: 0

LANG_CPP :: 0

LANG_C :: 1

// d3d12.BARRIER_LAYOUT
BARRIER_LAYOUT :: enum i32 {
	UNDEFINED                          = 0,
	COMMON                             = 1,
	PRESENT                            = 2,
	GENERIC_READ                       = 3,
	RENDER_TARGET                      = 4,
	UNORDERED_ACCESS                   = 5,
	DEPTH_STENCIL_WRITE                = 6,
	DEPTH_STENCIL_READ                 = 7,
	SHADER_RESOURCE                    = 8,
	COPY_SOURCE                        = 9,
	COPY_DEST                          = 10,
	RESOLVE_SOURCE                     = 11,
	RESOLVE_DEST                       = 12,
	SHADING_RATE_SOURCE                = 13,
	VIDEO_DECODE_READ                  = 14,
	VIDEO_DECODE_WRITE                 = 15,
	VIDEO_PROCESS_READ                 = 16,
	VIDEO_PROCESS_WRITE                = 17,
	VIDEO_ENCODE_READ                  = 18,
	VIDEO_ENCODE_WRITE                 = 19,
	DIRECT_QUEUE_COMMON                = 20,
	DIRECT_QUEUE_GENERIC_READ          = 30,
	DIRECT_QUEUE_UNORDERED_ACCESS      = 31,
	DIRECT_QUEUE_SHADER_RESOURCE       = 32,
	DIRECT_QUEUE_COPY_SOURCE           = 33,
	DIRECT_QUEUE_COPY_DEST             = 34,
	COMPUTE_QUEUE_COMMON               = 35,
	COMPUTE_QUEUE_GENERIC_READ         = 36,
	COMPUTE_QUEUE_UNORDERED_ACCESS     = 37,
	COMPUTE_QUEUE_SHADER_RESOURCE      = 38,
	COMPUTE_QUEUE_COPY_SOURCE          = 39,
	COMPUTE_QUEUE_COPY_DEST            = 40,
	VIDEO_QUEUE_COMMON                 = 41,
}

////////////////////////////////
// Forward declarations
Pool                   :: struct {}
VirtualBlock           :: struct {}
Allocator              :: struct {}
Allocation             :: struct {}
DefragmentationContext :: struct {}

////////////////////////////////
// Allocation Types
AllocHandle :: u64

AllocateFunctionType :: proc "c" (c.size_t, c.size_t, rawptr) -> rawptr

FreeFunctionType :: proc "c" (rawptr, rawptr)

ALLOCATION_CALLBACKS :: struct {
	pAllocate:    AllocateFunctionType,
	pFree:        FreeFunctionType,
	pPrivateData: rawptr,
}

ALLOCATION_FLAGS :: enum c.int {
	NONE                = 0,
	COMMITTED           = 1,
	NEVER_ALLOCATE      = 2,
	WITHIN_BUDGET       = 4,
	UPPER_ADDRESS       = 8,
	CAN_ALIAS           = 16,
	STRATEGY_MIN_MEMORY = 65536,
	STRATEGY_MIN_TIME   = 131072,
	STRATEGY_MIN_OFFSET = 262144,
	STRATEGY_BEST_FIT   = 65536,
	STRATEGY_FIRST_FIT  = 131072,
	STRATEGY_MASK       = 458752,
}

ALLOCATION_DESC :: struct {
	Flags:          ALLOCATION_FLAGS,
	HeapType:       d3d12.HEAP_TYPE,
	ExtraHeapFlags: d3d12.HEAP_FLAGS,
	CustomPool:     ^Pool,
	pPrivateData:   rawptr,
}

Statistics :: struct {
	BlockCount:      u32,
	AllocationCount: u32,
	BlockBytes:      u64,
	AllocationBytes: u64,
}

DetailedStatistics :: struct {
	Stats:              Statistics,
	UnusedRangeCount:   u32,
	AllocationSizeMin:  u64,
	AllocationSizeMax:  u64,
	UnusedRangeSizeMin: u64,
	UnusedRangeSizeMax: u64,
}

TotalStatistics :: struct {
	HeapType:           [5]DetailedStatistics,
	MemorySegmentGroup: [2]DetailedStatistics,
	Total:              DetailedStatistics,
}

Budget :: struct {
	Stats:       Statistics,
	UsageBytes:  u64,
	BudgetBytes: u64,
}

VirtualAllocation :: struct {
	AllocHandle: AllocHandle,
}

////////////////////////////////
// Defragmentation Types
DEFRAGMENTATION_FLAGS :: enum c.int {
	_3D12MA_DEFRAGMENTATION_FLAG_ALGORITHM_FAST    = 1,
	_3D12MA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED = 2,
	_3D12MA_DEFRAGMENTATION_FLAG_ALGORITHM_FULL    = 4,
	EFRAGMENTATION_FLAG_ALGORITHM_MASK             = 7,
}

DEFRAGMENTATION_DESC :: struct {
	Flags:                 DEFRAGMENTATION_FLAGS,
	MaxBytesPerPass:       u64,
	MaxAllocationsPerPass: u32,
}

DEFRAGMENTATION_MOVE_OPERATION :: enum c.int {
	COPY    = 0,
	IGNORE  = 1,
	DESTROY = 2,
}

DEFRAGMENTATION_MOVE :: struct {
	Operation:         DEFRAGMENTATION_MOVE_OPERATION,
	pSrcAllocation:    ^Allocation,
	pDstTmpAllocation: ^Allocation,
}

DEFRAGMENTATION_PASS_MOVE_INFO :: struct {
	MoveCount: u32,
	pMoves:    ^DEFRAGMENTATION_MOVE,
}

DEFRAGMENTATION_STATS :: struct {
	BytesMoved:       u64,
	BytesFreed:       u64,
	AllocationsMoved: u32,
	HeapsFreed:       u32,
}

////////////////////////////////
// Pool Types
POOL_FLAGS :: enum c.int {
	NONE                           = 0,
	ALGORITHM_LINEAR               = 1,
	MSAA_TEXTURES_ALWAYS_COMMITTED = 2,
	ALGORITHM_MASK                 = 1,
}

POOL_DESC :: struct {
	Flags:                  POOL_FLAGS,
	HeapProperties:         d3d12.HEAP_PROPERTIES,
	HeapFlags:              d3d12.HEAP_FLAGS,
	BlockSize:              u64,
	MinBlockCount:          u32,
	MaxBlockCount:          u32,
	MinAllocationAlignment: u64,
	pProtectedSession:      ^rawptr,
	ResidencyPriority:      d3d12.RESIDENCY_PRIORITY,
}

////////////////////////////////
// Allocator Types
ALLOCATOR_FLAGS :: enum c.int {
	NONE                                = 0,
	SINGLETHREADED                      = 1,
	ALWAYS_COMMITTED                    = 2,
	DEFAULT_POOLS_NOT_ZEROED            = 4,
	MSAA_TEXTURES_ALWAYS_COMMITTED      = 8,
	DONT_PREFER_SMALL_BUFFERS_COMMITTED = 16,
}

ALLOCATOR_DESC :: struct {
	Flags:                ALLOCATOR_FLAGS,
	pDevice:              ^d3d12.IDevice,
	PreferredBlockSize:   u64,
	pAllocationCallbacks: ^ALLOCATION_CALLBACKS,
	pAdapter:             ^dxgi.IAdapter,
}

////////////////////////////////
// Virtual Block Types
VIRTUAL_BLOCK_FLAGS :: enum c.int {
	NONE             = 0,
	ALGORITHM_LINEAR = 1,
	ALGORITHM_MASK   = 1,
}

VIRTUAL_BLOCK_DESC :: struct {
	Flags:                VIRTUAL_BLOCK_FLAGS,
	Size:                 u64,
	pAllocationCallbacks: ^ALLOCATION_CALLBACKS,
}

VIRTUAL_ALLOCATION_FLAGS :: enum c.int {
	NONE                = 0,
	UPPER_ADDRESS       = 8,
	STRATEGY_MIN_MEMORY = 65536,
	STRATEGY_MIN_TIME   = 131072,
	STRATEGY_MIN_OFFSET = 262144,
	STRATEGY_MASK       = 458752,
}

VIRTUAL_ALLOCATION_DESC :: struct {
	Flags:        VIRTUAL_ALLOCATION_FLAGS,
	Size:         u64,
	Alignment:    u64,
	pPrivateData: rawptr,
}

VIRTUAL_ALLOCATION_INFO :: struct {
	Offset:       u64,
	Size:         u64,
	pPrivateData: rawptr,
}

@(default_calling_convention="c", link_prefix="D3D12MA")
foreign d3d12ma {
	////////////////////////////////
	// Virtual Allocation Top-Level API
	Allocation_GetOffset      :: proc(pSelf: rawptr) -> u64 ---
	Allocation_GetAlignment   :: proc(pSelf: rawptr) -> u64 ---
	Allocation_GetSize        :: proc(pSelf: rawptr) -> u64 ---
	Allocation_GetResource    :: proc(pSelf: rawptr) -> ^d3d12.IResource ---
	Allocation_SetResource    :: proc(pSelf: rawptr, pResource: ^d3d12.IResource) ---
	Allocation_GetHeap        :: proc(pSelf: rawptr) -> ^d3d12.IHeap ---
	Allocation_SetPrivateData :: proc(pSelf: rawptr, pPrivateData: rawptr) ---
	Allocation_GetPrivateData :: proc(pSelf: rawptr) -> rawptr ---
	Allocation_SetName        :: proc(pSelf: rawptr, Name: windows.LPCWSTR) ---
	Allocation_GetName        :: proc(pSelf: rawptr) -> windows.LPCWSTR ---

	////////////////////////////////
	// Defragmentation Context Top-Level API
	DefragmentationContext_BeginPass :: proc(pSelf: rawptr, pPassInfo: ^DEFRAGMENTATION_PASS_MOVE_INFO) -> windows.HRESULT ---
	DefragmentationContext_EndPass   :: proc(pSelf: rawptr, pPassInfo: ^DEFRAGMENTATION_PASS_MOVE_INFO) -> windows.HRESULT ---
	DefragmentationContext_GetStats  :: proc(pSelf: rawptr, pStats: ^DEFRAGMENTATION_STATS) ---

	////////////////////////////////
	// Pool Top-Level API
	Pool_GetDesc              :: proc(pSelf: rawptr) -> POOL_DESC ---
	Pool_GetStatistics        :: proc(pSelf: rawptr, pStats: ^Statistics) ---
	Pool_CalculateStatistics  :: proc(pSelf: rawptr, pStats: ^DetailedStatistics) ---
	Pool_SetName              :: proc(pSelf: rawptr, Name: windows.LPCWSTR) ---
	Pool_GetName              :: proc(pSelf: rawptr) -> windows.LPCWSTR ---
	Pool_BeginDefragmentation :: proc(pSelf: rawptr, pDesc: ^DEFRAGMENTATION_DESC, ppContext: ^^DefragmentationContext) -> windows.HRESULT ---

	////////////////////////////////
	// Allocator Top-Level API
	Allocator_GetD3D12Options          :: proc(pSelf: rawptr) -> ^rawptr ---
	Allocator_IsUMA                    :: proc(pSelf: rawptr) -> bool ---
	Allocator_IsCacheCoherentUMA       :: proc(pSelf: rawptr) -> bool ---
	Allocator_IsGPUUploadHeapSupported :: proc(pSelf: rawptr) -> bool ---
	Allocator_GetMemoryCapacity        :: proc(pSelf: rawptr, MemorySegmentGroup: u32) -> u64 ---
	Allocator_CreateResource           :: proc(pSelf: rawptr, pAllocDesc: ^ALLOCATION_DESC, pResourceDesc: ^d3d12.RESOURCE_DESC, InitialResourceState: d3d12.RESOURCE_STATES, pOptimizedClearValue: ^d3d12.CLEAR_VALUE, ppAllocation: ^^Allocation, riidResource: ^d3d12.IID, ppvResource: ^rawptr) -> windows.HRESULT ---
	Allocator_CreateResource2          :: proc(pSelf: rawptr, pAllocDesc: ^ALLOCATION_DESC, pResourceDesc: ^d3d12.RESOURCE_DESC1, InitialResourceState: d3d12.RESOURCE_STATES, pOptimizedClearValue: ^d3d12.CLEAR_VALUE, ppAllocation: ^^Allocation, riidResource: ^d3d12.IID, ppvResource: ^rawptr) -> windows.HRESULT ---
	Allocator_CreateResource3          :: proc(pSelf: rawptr, pAllocDesc: ^ALLOCATION_DESC, pResourceDesc: ^d3d12.RESOURCE_DESC1, InitialLayout: BARRIER_LAYOUT, pOptimizedClearValue: ^d3d12.CLEAR_VALUE, NumCastableFormats: u32, pCastableFormats: ^dxgi.FORMAT, ppAllocation: ^^Allocation, riidResource: ^d3d12.IID, ppvResource: ^rawptr) -> windows.HRESULT ---
	Allocator_AllocateMemory           :: proc(pSelf: rawptr, pAllocDesc: ^ALLOCATION_DESC, pAllocInfo: ^d3d12.RESOURCE_ALLOCATION_INFO, ppAllocation: ^^Allocation) -> windows.HRESULT ---
	Allocator_CreateAliasingResource   :: proc(pSelf: rawptr, pAllocation: ^Allocation, AllocationLocalOffset: u64, pResourceDesc: ^d3d12.RESOURCE_DESC, InitialResourceState: d3d12.RESOURCE_STATES, pOptimizedClearValue: ^d3d12.CLEAR_VALUE, riidResource: ^d3d12.IID, ppvResource: ^rawptr) -> windows.HRESULT ---
	Allocator_CreateAliasingResource1  :: proc(pSelf: rawptr, pAllocation: ^Allocation, AllocationLocalOffset: u64, pResourceDesc: ^d3d12.RESOURCE_DESC1, InitialResourceState: d3d12.RESOURCE_STATES, pOptimizedClearValue: ^d3d12.CLEAR_VALUE, riidResource: ^d3d12.IID, ppvResource: ^rawptr) -> windows.HRESULT ---
	Allocator_CreateAliasingResource2  :: proc(pSelf: rawptr, pAllocation: ^Allocation, AllocationLocalOffset: u64, pResourceDesc: ^d3d12.RESOURCE_DESC1, InitialLayout: BARRIER_LAYOUT, pOptimizedClearValue: ^d3d12.CLEAR_VALUE, NumCastableFormats: u32, pCastableFormats: ^dxgi.FORMAT, riidResource: ^d3d12.IID, ppvResource: ^rawptr) -> windows.HRESULT ---
	Allocator_CreatePool               :: proc(pSelf: rawptr, pPoolDesc: ^POOL_DESC, ppPool: ^^Pool) -> windows.HRESULT ---
	Allocator_SetCurrentFrameIndex     :: proc(pSelf: rawptr, FrameIndex: u32) ---
	Allocator_GetBudget                :: proc(pSelf: rawptr, pLocalBudget: ^Budget, pNonLocalBudget: ^Budget) ---
	Allocator_CalculateStatistics      :: proc(pSelf: rawptr, pStats: ^TotalStatistics) ---
	Allocator_BuildStatsString         :: proc(pSelf: rawptr, ppStatsString: ^^windows.WCHAR, DetailedMap: bool) ---
	Allocator_FreeStatsString          :: proc(pSelf: rawptr, pStatsString: ^windows.WCHAR) ---
	Allocator_BeginDefragmentation     :: proc(pSelf: rawptr, pDesc: ^DEFRAGMENTATION_DESC, ppContext: ^^DefragmentationContext) ---

	////////////////////////////////
	// Virtual Block Top-Level API
	VirtualBlock_IsEmpty                  :: proc(pSelf: rawptr) -> bool ---
	VirtualBlock_GetAllocationInfo        :: proc(pSelf: rawptr, Allocation: VirtualAllocation, pInfo: ^VIRTUAL_ALLOCATION_INFO) ---
	VirtualBlock_Allocate                 :: proc(pSelf: rawptr, pDesc: ^VIRTUAL_ALLOCATION_DESC, pAllocation: ^VirtualAllocation, pOffset: ^u64) -> windows.HRESULT ---
	VirtualBlock_FreeAllocation           :: proc(pSelf: rawptr, Allocation: VirtualAllocation) ---
	VirtualBlock_Clear                    :: proc(pSelf: rawptr) ---
	VirtualBlock_SetAllocationPrivateData :: proc(pSelf: rawptr, Allocation: VirtualAllocation, pPrivateData: rawptr) ---
	VirtualBlock_GetStatistics            :: proc(pSelf: rawptr, pStats: ^Statistics) ---
	VirtualBlock_CalculateStatistics      :: proc(pSelf: rawptr, pStats: ^DetailedStatistics) ---
	VirtualBlock_BuildStatsString         :: proc(pSelf: rawptr, ppStatsString: ^^windows.WCHAR) ---
	VirtualBlock_FreeStatsString          :: proc(pSelf: rawptr, pStatsString: ^windows.WCHAR) ---

	////////////////////////////////
	// Top-Level API
	CreateAllocator    :: proc(pDesc: ^ALLOCATOR_DESC, ppAllocator: ^^Allocator) -> windows.HRESULT ---
	CreateVirtualBlock :: proc(pDesc: ^VIRTUAL_BLOCK_DESC, ppVirtualBlock: ^^VirtualBlock) -> windows.HRESULT ---
}
