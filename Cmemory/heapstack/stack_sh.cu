/**
 * 	@file 	stack_sh.cu
 * 	@brief 	CUDA C implementation with arrays of stack on shared memory
 * 	@ref	https://devtalk.nvidia.com/default/topic/417881/implementing-stack-in-cuda/
 * 	@details 3 (or 4?) basic operations; 
 * 	- push, if stack is full, then it's said to be an Overflow condition
 * 	- pop, if stack empty, then it's said to be an Underflow condition 
 * 	- peek or top, returns top element of stack.  
 * 	- isEmpty, returns true if stack empty, else false
 * 
 * Pros; easy to implement, memory not saved as pointers; Cons; no dynamic, doesn't grow and shrink depending on runtime needs
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * gcc -Wall -g stack_stack_arr.c -o stack_stack_arr
 * */
#include <stdio.h>   

const int BLOCK_SIZE_Y = 1 << 5;  // 1 << 5 = 32 
const int WARP_SIZE = 1 << 5;  	// 1 << 5 = 32
const int STACK_SIZE = 10;  
 
__shared__ int stackindex[BLOCK_SIZE_Y]; 
__global__ int stack[STACK_SIZE * BLOCK_SIZE_Y * WARP_SIZE]; 
__device__ void Push(int v) {
	// blockDim.x = WARP_SIZE
	stack[stackIndex[blockIdx.y] + blockIdx.x] = v; // Broadcast from shared + Coalesced global write
	stackIndex[blockIDx.y] += WARP_SIZE; // All threads in a warp write the same value
}

__device__ int Pop() {
	stackIndex[blockIdx.y] -= WARP_SIZE;
	return stack[stackIndex[blockIdx.y] + blockIdx.x] ; // Broadcast + Coalesced global read 
	
}

