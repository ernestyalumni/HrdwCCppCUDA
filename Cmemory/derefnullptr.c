/**
 * 	@file 	derefnullptr.c
 * 	@brief 	Dereference null ptr for undefined behavior, maybe Segmentation Fault (not guaranteed by C/C++ standard)
 * 	@ref	https://en.wikipedia.org/wiki/Segmentation_fault
 * 	@details Segmentation Fault not guaranteed; you could be allowed to write to 0 (0x00)
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * gcc -Wall -g derefnullptr.c -o derefnullptr  
 * Execution result:  
 * ./derefnullptr 
 * Segmentation fault (core dumped)
 * 
 * */
#include <stdio.h> // printf, NULL

int main() {
	char *cptr = NULL;  
	char chout = *cptr;  // Segmentation Fault occurs here 
//	printf(" chout, what cptr dereferences to : %c \n" , chout); 
} 

