/**
 * 	@file 	chararr_outbnds.c
 * 	@brief 	out of the "bounds" for char arrays
 * 	@ref	https://github.com/zedshaw/learn-c-the-hard-way-lectures/blob/master/ex13/ex13.c 
 * 	@details C treats strings as just arrays of bytes, 
 * 	string and array of bytes are the same thing
 * 
 * 	gcc -Wall -g chararr_outbnds.c
 * 	./chararr_outbnds 
 * */  
#include <stdio.h> // you don't even have to include it; it's noted by compiler
 
int main(int argc, char *argv[]) {
	char* arr_states[] = { 
		"123", "124", "125", "126" };
	int num_of_states = 6; // 6 will guarantee Segmentation Fault, outside of the End of Strings \00
	
	int i=0;
	// this for-loop exits normally
	for (i=0; i<num_of_states;i++){
		arr_states[i];  // it'll exit normally for num_of_states = 5,6 
		
	}	
	
	/* =============== Segmentation Fault =============== */
	/* =============== this for-loop DOES NOT exit normally =============== */
	for (i=0; i<num_of_states;i++){
		printf("state %d %s \n", 
			i, 
			arr_states[i]);  // it'll NOT exit normally for num_of_states = 5,6; segmentation fault
			
	}	

	
	
} 
 
