/**
 * 	@file 	ptr_vs_arr_onlyassign.c
 * 	@brief 	Pointers vs. arrays; differences; only the assignment part  
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

char array_place[100] = "don't panic";
char* ptr_place = "don't panic";
	

int main(int argc, char *argv[]) 
{
	/* ========== how they're difference ========== */
	
	char a = array_place[7];
	char b = ptr_place[7]; 
	
	return 0;
	
}
