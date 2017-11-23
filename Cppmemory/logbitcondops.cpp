/**
 * 	@file 	logbitcondops.cpp
 * 	@brief 	Examples of logical, bitwise logical, and conditional operators and expressions      
 * 	@ref Stroustrup. The C++ Programming Language, 4th Ed. 11.1. Etc. Operators  
 * 	@details Examples of logical operators (&&, ||, and !), bitwise logical operators (&, |, ~, <<, >>), 
 * 		conditional expressions (?:), and increment and decrement operators (++ and --)
 * 
 * std::ios_base::iostate - type for stream state flags, bitmask type to represent stream error state flags 
 * 
 * 	
 * 	COMPILATION TIP
 * 	-Wall all warnings -g debugger
 * 	g++ -Wall -g logbitcondops.cpp -o logbitcondops
 * 	gdb ./objlifecycle # go into gdb, and then break main , run, bt, next, next, ... print a, print b, ... x arr, x ptr, ...
 * */ 
#include <iostream> // std::ios_base::iostate
#include <cstring> // strlen

// enum std::ios_base::iostate { goodbit =0, eofbit=1,failbit=2,badbit=4};   

constexpr unsigned short middle(int a)
{
	static_assert(sizeof(int)==4,"unexpected int size");
	static_assert(sizeof(short)==2,"unexpected short size");
	return (a>>8)&0xFFFF;
}

/* *************** 11.1.4 Increment and Decrement of Stroustrup *************** */
void cpy(char* p, const char* q) 
{
	while(*p++ = *q++);
}

int main() {
	int old = std::cin.rdstate(); 	// rdstate() returns the state 
	// ... use cin ...

	// ^ (exclusive or, xor)  
	if (std::cin.rdstate()^old) { 		// has anything changed?
		// ...
		std::cout << " xor on std::cin occurred - " << std::endl;
	} 

	// & (and)  
	if (std::cin.rdstate()&old) { 		
		std::cout << " and on std::cin occurred - " << std::endl;
	} 

	// | (or)  
	if (std::cin.rdstate()|old) { 		
		std::cout << " or on std::cin occurred - " << std::endl;
	} 

	// << (left shift)
	if (std::cin.rdstate()<<old) { 		
		std::cout << " << on std::cin occurred - " << std::endl;
	} 

	// >> (right shift)
	if (std::cin.rdstate()>>old) { 		
		std::cout << " >> on std::cin occurred - " << std::endl;
	} 


	int x = 0xFF00FF00; // assume sizeof(int)==4
	short y = middle(x); // y = 0x00FF  


	/* *************** 11.1.4 Increment and Decrement of Stroustrup *************** */
	char *example_char_arr = "Example char array";  
	
	/*
	 * This is wasteful.  
	 * Length of a 0-terminated string is found by reading the string looking for the terminating zero. 
	 * Thus, we read string twice: once to find its length and once to copy it.  
	 * */
	int length = strlen(example_char_arr); 
	char example_p_char_arr[length]; 
	for (int i =0 ; i <= length; i++) 
	{
		example_p_char_arr[i] = example_char_arr[i]; 
	}
	std::cout << std::endl << " example_p_char_arr is : " << example_p_char_arr << std::endl; 

	char q[] = "Example str for q"; 
	char p[] = "aaaaaaaaaaaaaaaaa";
	// try this
	int i;
	for (i=0; q[i]!=0;i++) {
		p[i] = q[i]; }
	p[i]=0;	// terminating zero
	std::cout << std::endl << " p is : " << p << std::endl;
	
	char *q_ptr = "Example str for q"; 
	
	// because p and q are pointers 
	while (*q_ptr != 0) {
		std::cout << " *q_ptr " << *q_ptr << std::endl; 
		q_ptr++; 	// point to next character
	}
}
