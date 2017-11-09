/**
 * 	@file 	ex15.c
 * 	@brief 	Exercise 15. Pointers, Dreaded Pointers     
 * 	@ref	Zed A. Shaw.  Learn C the Hard Way (2015); Exercise 15
 * 	@details isalpha, isblank do all work of figuring out if given character is a letter or blank
 * */
#include <stdio.h>  

int main(int argc, char *argv[]) {
	// create 2 arrays we care about
	int ages[] = { 23, 43, 12, 89, 2 };
	char *names[] = { "Alan", "Frank", "Mary", "John", "Lisa" };
	// safely get the size of ages
	int count = sizeof(ages) / sizeof(int);
	int i = 0;
	// first way using indexing /* Looping through 2 arrays, printing indexed value.  This is using i to index into the array. */
	for (i=0; i < count; i++) { 
		printf("%s has %d years alive.\n", names[i], ages[i]);
	}
	
	printf("---\n");
	
	// set up the pointers to the start of the arrays   /* Create ptr that points at ages.   */
	int *cur_age = ages;  /* Create ptr that points at names */
	char **cur_name = names;
	
	// second way using pointers  /* Loop through ages and names but use ptr arithmetic with offset i 
	for (i=0; i < count; i++) {
		printf("%s is %d years old.\n", 
			*(cur_name + i), *(cur_age + i));
	}
	
	printf("---\n");
	
	// third way, pointers are just arrays /* Shows how syntax to access an element of array is same for ptr and array */ 
	for (i=0; i < count; i++) {
		printf("%s is %d years old again.\n", cur_name[i], cur_age[i]);
	}
	
	printf("---\n");
	
	// fourth way with pointers in a stupid complex way /* uses ptr arighmetic methods */
	for (cur_name = names, cur_age = ages; 
		(cur_age - ages) < count; cur_name++, cur_age++) {
			printf("%s lived %d years so far.\n", *cur_name, *cur_age);
	}
	
	return 0;
}
	
	
