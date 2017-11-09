/**
 * 	@file 	ex14.c
 * 	@brief 	Exercise 14. Writing and Using Functions   
 * 			functions to print out characters and ASCII codes 
 * 	@ref	Zed A. Shaw.  Learn C the Hard Way (2015); Exercise 14
 * 	@details isalpha, isblank do all work of figuring out if given character is a letter or blank
 * 
 * 	make ex14
 * 	./ex14 
 * */
#include <stdio.h>
#include <ctype.h> // isalpha, isblank

// forward declarations
int can_print_it(char ch);
void print_letters(char arg[]);

void print_arguments(int argc, char*argv[])
{
	int i=0;
	for (i=0; i<argc;i++) {
		print_letters(argv[i]);
	}
}

void print_letters(char arg[]) {
	int i=0;
	
	for (i=0; arg[i] != '\0'; i++) {
		char ch = arg[i];
		
		if (can_print_it(ch)) {
			printf("'%c' == %d ", ch, ch); 
		}
	}
	
	printf("\n");
}

int can_print_it(char ch) {
	return isalpha(ch) || isblank(ch); 
}

int main(int argc, char *argv[]) {
	print_arguments(argc, argv) ; 
	return 0;
}


