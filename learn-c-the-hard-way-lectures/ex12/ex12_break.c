/**
 * 	@file 	ex12_break.c
 * 	@brief 	Exercise 12. Sizes and Arrays. 
 * 	@ref	Zed A. Shaw.  Learn C the Hard Way (2015)
 * 	@details C treats strings as just arrays of bytes, 
 * 	and it's only the different printing functions that recognize a difference.  
 * 
 * 	Array syntax type name[] = {initializer}; 
 * 	"I want an array of type that is initialized to { }.  When C sees this, it knows to
 * 	* Look at the type, 
 *  * Look at [] and see there's no length given 
 *  * Look at initializer and figure you want those 5 ints in array
 *  * Create piece of memory in computer that can hold 5 integers 1 after another
 * 	* take name you want, areas, and assign it this location  
 * 
 * 	break by 
 * 	* removing, from a char [] array, '\0'
 * 	* areas[10]
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * gcc -Wall -g ex12_break.c -o ex12_break
 * */
#include <stdio.h>

int main(int argc, char *argv[])
{
	char full_name[] = { 'Z', 'e', 'd', 
							' ', 'A', '.', ' ', 
							'S', 'h', 'a','w'   // ,'\0' break by removing
						};


	int areas[] = {10,12,13,14,20};
	char name[] = "Zed";
	
	// WARNING: On some systems you may have to change the 
	// %ld in this code to a %u since it will use unsigned ints 
	printf("The size of an int: %ld\n", sizeof(int));
	printf("The size of areas (int[]): %ld\n", sizeof(areas));
	printf("The number of ints in areas: %ld\n",
		sizeof(areas)/sizeof(int));
	printf("The first area is %d, the 2nd %d.\n", areas[0], areas[1]);

	printf("\n 10th area is %d . \n", areas[10]);
	int outside_area_int = areas[10];
	printf("\n 10th area, assigned to another int, is %d . \n", outside_area_int);
	
	printf("\n Addresses of areas 0,1,10, and assigned int to 10th area, : %d %d %d %d . \n", 
		&areas[0], &areas[1], &areas[10], &outside_area_int);
	 
	
	printf("The size of a char: %ld\n", sizeof(char));
	printf("The size of name (char[]): %ld\n", sizeof(name));
	printf("the number of chars: %ld\n", sizeof(name) /sizeof(char));
	
	printf("The size of full_name (char[]): %ld\n", sizeof(full_name));
	
	printf("The number of chars: %ld\n", sizeof(full_name) /sizeof(char));

	printf("name=\"%s\" and full_name=\"%s\"\n", name, full_name);
	
	
	/* *************** Extra Credit *************** */
	// Try assigning to elements in the areas array with areas[0] = 100;
	areas[0] = 300;

	printf("The first area is %d, the 2nd %d.\n", areas[0], areas[1]);
	printf("\n Addresses of areas 0,1,10, and assigned int to 10th area, : %d %d %d %d . \n", 
		&areas[0], &areas[1], &areas[10], &outside_area_int);
	
	
	return 0;
}
