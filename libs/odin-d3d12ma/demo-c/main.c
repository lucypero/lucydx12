#include <stdio.h>
#include <windows.h>
#include <dxgi.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include "d3d12ma.h"

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d12ma.lib")

int main() {
    IDXGIAdapter1* adapter = NULL;
    ID3D12Device* device = NULL;
    D3D12MAAllocator* allocator = NULL;
    D3D12MAAllocation* allocation = NULL;
    ID3D12Resource* resource = NULL;

    HRESULT hr;

    IDXGIFactory2* factory;
    hr = CreateDXGIFactory2(0, &IID_IDXGIFactory2, (void**)&factory);
    if (FAILED(hr)) {
        printf("Failed to create DXGI factory\n");
        return 1;
    }

    hr = factory->lpVtbl->EnumAdapters1(factory, 0, &adapter);
    if (FAILED(hr)) {
        printf("Failed to enumerate adapters\n");
        return 1;
    }

    hr = D3D12CreateDevice((IUnknown*)adapter, D3D_FEATURE_LEVEL_12_0, &IID_ID3D12Device, (void**)&device);
    if (FAILED(hr)) {
        printf("Failed to create D3D12 device\n");
        return 1;
    }

    D3D12MA_ALLOCATOR_DESC allocatorDesc = {0};
    allocatorDesc.Flags = D3D12MA_ALLOCATOR_FLAG_NONE;
    allocatorDesc.pDevice = device;
    allocatorDesc.pAdapter = (IDXGIAdapter*)adapter;

    hr = D3D12MACreateAllocator(&allocatorDesc, &allocator);
    if (FAILED(hr)) {
        printf("Failed to create D3D12MA allocator\n");
        return 1;
    }

    D3D12_RESOURCE_DESC resourceDesc = {0};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Width = 1024 * 1024;
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    D3D12MA_ALLOCATION_DESC allocationDesc = {0};
    allocationDesc.Flags = D3D12MA_ALLOCATION_FLAG_NONE;
    allocationDesc.HeapType = D3D12_HEAP_TYPE_DEFAULT;

    hr = D3D12MAAllocator_CreateResource(allocator, &allocationDesc, &resourceDesc, D3D12_RESOURCE_STATE_COPY_DEST, NULL, &allocation, &IID_ID3D12Resource, (void**)&resource);
    if (FAILED(hr)) {
        printf("Failed to create resource\n");
        return 1;
    }

    printf("Hello, World with D3D12MA!\n");
    printf("Allocated buffer: %llu bytes\n", (unsigned long long)D3D12MAAllocation_GetSize(allocation));

    ULONG ref;
    ref = resource->lpVtbl->Release(resource);
    printf("resource refcount: %lu\n", ref);
    ref = ((IUnknown*)allocation)->lpVtbl->Release((IUnknown*)allocation);
    printf("allocation refcount: %lu\n", ref);
    ref = ((IUnknown*)allocator)->lpVtbl->Release((IUnknown*)allocator);
    printf("allocator refcount: %lu\n", ref);
    ref = device->lpVtbl->Release(device);
    printf("device refcount: %lu\n", ref);
    ref = adapter->lpVtbl->Release(adapter);
    printf("adapter refcount: %lu\n", ref);
    ref = factory->lpVtbl->Release(factory);
    printf("factory refcount: %lu\n", ref);

    return 0;
}
