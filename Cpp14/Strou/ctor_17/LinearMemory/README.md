cf. [3.2.2. Device Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory) of 3.2 CUDA C Runtime of Programming Guide of CUDA Toolkit Documentation.

Device memory can be allocated either as *linear memory* or as *CUDA arrays*.

CUDA arrays are opaque memory layouts optimized for texture fetching, e.g. [Texture and Surface Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory)

Linear memory exists on device in 40-bit address space, so separately allocated entities can reference 1 another via ptrs.

Linear memory allocated using `cudaMalloc()`, also `cudaMallocPitch()`, `cudaMalloc3D`, freed using `cudaFree()`


cf. https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html

```
__host__ cudaError_t cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent)
```

Allocates logical 1D, 2D, or 3D memory objects on the device.

Allocates at least width * height * depth bytes of linear memory on device and returns `cudaPitchedPtr` in which (?) pointer to allocated memory.

Function may pad allocation to ensure hardware alignment requirements met.

Pitch returned in `pitch` field of `pitchedDevPtr` is width in bytes of allocation.

[`cudaMallocManaged`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gd228014f19cc0975ebe3e0dd2af6dd1b)

```
__host__ cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags = cudaMemAttachGlobal)
```
Allocates memory that'll automatically be managed by the Unified Memory system.

Allocates `size` bytes of managed memory on device and returns in `*devPtr` pointer to allocated memory.

Memory allocated with `cudaMallocManaged` should be released with `cudaFree`.
