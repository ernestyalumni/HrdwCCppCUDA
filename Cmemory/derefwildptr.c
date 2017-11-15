/**
 * 	@file 	derefwildptr.c
 * 	@brief 	Dereference wild ptr for undefined behavior, maybe Segmentation Fault 
 * 	@ref	https://en.wikipedia.org/wiki/Segmentation_fault
 * 	@details Segmentation Fault not guaranteed; wild ptr (uninitialized ptr) may point to memory that may or may not exist 
 * 	and may or may not be readable or writable.  Reading from a wild ptr may result in random data but no segmentation fault.  
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
	char *cptr;  
	*cptr;	// this is fine 
	char chout = *cptr;  // Segmentation Fault occurs here 
//	printf(" chout, what cptr dereferences to : %c \n" , chout); 
} 
