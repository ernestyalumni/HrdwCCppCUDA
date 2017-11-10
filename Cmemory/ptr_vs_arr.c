/**
 * 	@file 	ptr_vs_arr.c
 * 	@brief 	Pointers vs. arrays; differences      
 * 	@ref	Zed A. Shaw.  Learn C the Hard Way (2015); Exercise 15
 * 			https://eli.thegreenplace.net/2009/10/21/are-pointers-and-arrays-equivalent-in-c
 * 	@details 
 * 	
 * 	COMPILATION TIP
 * 	-Wall all warnings -g debugger
 * 	gcc -Wall -g ptr_vs_arr.c -o ptr_vs_arr
 * 	gdb ./ptr_vs_arr # go into gdb, and then break main , run, bt, next, next, ... print a, print b, ... x arr, x ptr, ...
 * */
#include <stdio.h>  

int main(int argc, char *argv[]) 
{
	/* =============== from Eli Bendersky's website =============== */
	/* ========== ptr arithmetic and array indexing are equivalent */
	char arr[] = "don't panic\n";
	char* ptr = arr;
	
	printf("%c %c \n", arr[4], ptr[4]);
	printf("%c %c \n", *(arr+2), *(ptr+2));
	
	/* ========== how they're difference ========== */
	
	char array_place[100] = "don't panic";
	char* ptr_place = "don't panic";
	
	char a = array_place[7];
	char b = ptr_place[7]; 
	
	return 0;
	
}
