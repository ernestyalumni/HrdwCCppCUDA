/**
 * 	@file 	ptrsizeof.cpp
 * 	@brief 	What is the size of a pointer? 
 * 	@ref	https://stackoverflow.com/questions/6751749/what-is-the-size-of-a-pointer/6751914#6751914 
 * 	@details Functions pointers can have very different sizes, from 4 to 20 bytes on an x86, depending on compiler.  
 * 64 bit it's 64 bit (8 bytes) 
 *  
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g ptrsizeof.cpp -o ptrsizeof
 * */
#include <iostream>  

int main() 
{
	int x = 10;
	int * xPtr = &x; 
	char y = 'a';
	char * yPtr = &y;
	float z = 3.f;
	float *zptr = &z; 
	
	std::cout << sizeof(x) << "\n"; 	// 4 byte, 32 bit
	std::cout << sizeof(xPtr) << "\n";	// 8 byte, 64 bit
	std::cout << sizeof(y) << "\n";		// 1 byte, 8 bit
	std::cout << sizeof(yPtr) << "\n";	// 8 byte, 64 bit
	std::cout << sizeof(z) << "\n";		// 4 byte, 32 bit
	std::cout << sizeof(zptr) << "\n";	// 8 byte, 64 bit

}
