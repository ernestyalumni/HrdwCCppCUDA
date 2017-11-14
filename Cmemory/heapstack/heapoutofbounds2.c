/**
 * 	@file 	heapoutofbounds2.c
 * 	@brief 	Heap Memory leak example, 2 problems; heap block overrun, & no freeing
 * 	@ref	http://www.inf.udec.cl/~leo/teoX.pdf
 * 	@details 2 problems; heap block overrun, 2. memory leak from x not freed
 * 
 * gdb note; doing disassemble main, break *main+13, brea *main+41 (strategy is to break at 
 * function calls and before ret (return), then run, step, continue, and use 
 * print p and x , including x/100x or x, to examine specific variables 
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * gcc -Wall -g heapinfmalloc.c -o heapinfmalloc  
 * 
 * */
#include <stdlib.h> // malloc

int main(void) 
{
	int* x  = malloc(10*sizeof(int));
	x[11] = 1;				

	return 0;
}
