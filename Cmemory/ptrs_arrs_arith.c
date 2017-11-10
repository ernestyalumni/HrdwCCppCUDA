/**
 * 	@file 	ptrs_arrs_arith.c
 * 	@brief 	Pointers, arrays, ptr (pointer) arithmetic; based on Exercise 15. Pointers, Dreaded Pointers of Shaw (2015)    
 * 	@ref	Zed A. Shaw.  Learn C the Hard Way (2015); Exercise 15  
 * 			https://eli.thegreenplace.net/2009/10/21/are-pointers-and-arrays-equivalent-in-c
 * 	@details 
 * 
 * */
#include <stdio.h>  

int main(int argc, char* argv[]) {
	
	// create arrays  
	int arr_int[] = { 23, 43, 12, 89, 2, 5 } ;
	char *arr_str[] = {"Alan", "Frank", "Mary", "John", "Lisa" };  
	
	float arr_f[]= { .63f, .69f, 1.5833f, 5.12f, 48.3f, 2.9f, 444.3f }; 
	
	printf(" sizeof(int) : %d \n ", sizeof(int) ); 
	printf(" sizeof(char) : %d \n ", sizeof(char) ); 
	printf(" sizeof(float) : %d \n ", sizeof(float) ); 
	
	int* ptr_int = arr_int; 
	char **ptr_str = arr_str;
	float* ptr_f = arr_f;
	
	int i =0;
	for (i=0; *(ptr_str+i) != "\001"; i++) {
		printf(" %s in arr_str . \n", *(ptr_str +i) ); }
/*	
	while ( *(ptr_str+i) != "\001" ) {
		printf(" %s in arr_str . \n", *(ptr_str+i) ); 
		i++;		
	} 
	*/
	
}
