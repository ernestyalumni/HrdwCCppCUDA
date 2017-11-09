/**
 * 	@file 	ex13_states.c
 * 	@brief 	Exercise 13. For-Loops and Arrays of Strings  
 * 			code that'll print out any command line arguments you pass it
			
 * 	@ref	Zed A. Shaw.  Learn C the Hard Way (2015)
 * 	ex13.c on Shaw's github is DIFFERENT from code in the book 
 * 	https://github.com/zedshaw/learn-c-the-hard-way-lectures/blob/master/ex13/ex13.c
 * 	@details C treats strings as just arrays of bytes, 
 * 	string and array of bytes are the same thing
 * 	array of strings char *argv[]
 * 
 * 	make ex13_states
 * 	./ex13_states "i am a bunch of arguments"
 * CFLAGs="-Wall -g" make ex13_states
 * */
#include <stdio.h>

int main(int argc, char* argv[]) 
{
	if (argc != 2) {
		printf("ERROR: You need one argument.\n");
		// this is how you abort a program
		return 1;
	}
	
	int i = 0;
	for (i=0; argv[1][i] != '\0'; i++) {
		char letter = argv[1][i]; 
		
		switch (letter) {
			case 'a': 
			case 'A':
			printf("%d: 'A'\n", i);
			break;
			
			case 'e':
			case 'E':
			printf("%d: 'E'\n",i);
			
			case 'i':
			case 'I': 
			printf("%d: 'I'\n",i);
			break;
			
			case 'o':
			case 'O':
			printf("%d: 'O'\n",i);
			break;
			
			case 'u':
			case 'U':
			printf("%d: 'U'\n",i);
			break;
			
			case 'y':
			case 'Y':
			if (i>2) {
				// it's only sometimes Y
				printf("%d: 'Y'\n", i);
			}
			break;
			
			default:
			printf("%d: %c is not a vowel\n", i, letter);
		}
	} // END of for loop 
	
	// let's make our own array of strings
	char *states[] = {
		"California", "Oregon", "Washington", "Texas" };
	
	int num_states = 5;
//	int num_states = 4;

	
	for (i=0; i<num_states; i++) {
		printf("state %d: %s\n", i , states[i]);
	}
	
	return 0;
}
