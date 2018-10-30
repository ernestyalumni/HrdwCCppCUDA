/**
 * 	@file 	derefnullfptr.c
 * 	@brief 	Dereference null ptr for undefined behavior, maybe Segmentation Fault (not guaranteed by C/C++ standard)
 * 	@ref	https://en.wikipedia.org/wiki/Segmentation_fault
 * https://stackoverflow.com/questions/12645647/what-happens-in-os-when-we-dereference-a-null-pointer-in-c 
 * 	@details Segmentation Fault not guaranteed; you could be allowed to write to 0 (0x00).  
 * Otherwise, as Adam Rosenfield explained, virtual memory address of 0x0 is invalid
 *   
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * gcc -Wall -g derefnullfptr.c -o derefnullfptr  
 * Execution result:  
 * ./derefnullfptr 
 * Segmentation fault (core dumped)
 * 
 * */
#include <stdio.h>

int main() {
	float *fptr = NULL; // doesn't "initialize" *fptr to 0, instead, C reads it as it does NULL; // this line alone still compiles	
	// *fptr; // this line also DOES NOT trigger a Segmentation Fault 
	*fptr = 10.f;  // results in Segmentation Fault  
//	float fout = *fptr;  // results in Segmentation Fault 
} 
 
