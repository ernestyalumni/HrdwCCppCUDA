/**
 * 	@file 	derefdanglingptr.c
 * 	@brief 	Dereference wild ptr for undefined behavior, maybe Segmentation Fault 
 * 	@ref	https://en.wikipedia.org/wiki/Segmentation_fault
 * 	@details Segmentation Fault not guaranteed; dangling ptr (freed ptr, which points to memory that has been freed/deallocated/deleted) may point to memory that may or may not exist 
 * 	and may or may not be readable or writable.  Reading from a dangling ptr may result in valid data for a while, and then random data as it's overwritten.    
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * gcc -Wall -g derefwildptr.c -o derefwildptr  
 * Execution result:  
 * ./derefwildptr 
 * Segmentation fault (core dumped)
 * 
 * */
#include <stdio.h> // printf

int main() {
	char *cptr = malloc(10*sizeof(char)); // sizeof(char) = 1 anyways 
	free(cptr);
//	*cptr;  // this is still fine, NO Segmentation Fault signal triggered  
	char chout = *cptr;   
//	printf("chout : %x \n " , chout); 
} 
