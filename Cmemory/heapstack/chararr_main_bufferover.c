/**
 * 	@file 	chararr_main_bufferover.c
 * 	@brief 	buffer overflow examples; char array in main and memcpy
 * 	@ref	https://www.exploit-db.com/docs/28475.pdf
 * 	http://insecure.org/stf/smashstack.html
 * 	@details write past end of a variable, overwriting vital info  
 * 	
 * 	When foo() returns, it pops return address off stack and jumps to that address 
 * 	(i.e. starts executing instructions from the address).  
 *  
 * 	memcpy - copies count characters from object pointed to by src to object pointed to by dest 
 * 	Both objects are interpreted as arrays of unsigned char
 * 	void* memcpy(void *dest, const void *src, size t_count) 
 *  cf. http://en.cppreference.com/w/c/string/byte/memcpy
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * gcc -Wall -g chararr_main_bufferover.c -o chararr__main_bufferover 
 * 
 * ./chararr_main_bufferover 
 * Segmentation fault (core dumped)
 * ./chararr_main_bufferover "great" // great # okay
 * ./chararr_main_bufferover "Come back with your shield or on it" // #okay
 * ./chararr_main_bufferover "Every day you may make progress.  Every step may be fruitful.  Yet there will stretch out before you an ever-lengthening, ever-ascending, ever-improving path.  You know you will never get to the end of the journey.  But this, so far from discouraging, only adds to the joy and glory of the climb."  // Segmentation fault, 296 letter count
 * ./chararr_main_bufferover $(python -c 'print "A"*263') # ok 
 * ./chararr_main_bufferover $(python -c 'print "A"*264') # ok, but prints out 
 * AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA1�Է�
 * ./chararr_main_bufferover $(python -c 'print "A"*265')
 * Segmentation fault (core dumped)
 * 
 * */
#include <stdio.h>  // printf
#include <string.h> // memcpy, strlen
 
int main(int argc, char *argv[]) 
{
	 char buf[256];
	 memcpy(buf, argv[1], strlen(argv[1]));
	 printf(buf);
}
