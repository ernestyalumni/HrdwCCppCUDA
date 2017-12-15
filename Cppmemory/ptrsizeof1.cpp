/**
 * 	@file 	ptrsizeof1.cpp
 * 	@brief 	What is the size of a pointer? 
 * 	@ref	https://stackoverflow.com/questions/6751749/what-is-the-size-of-a-pointer/6751914#6751914 
 * 	@details Functions pointers can have very different sizes, from 4 to 20 bytes on an x86, depending on compiler.  
 * 64 bit it's 64 bit (8 bytes) 
 *  
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g ptrsizeof.cpp -o ptrsizeof
 * */
#include <iostream>  

class IntCls
{
	private:
		int n;
		float *x;
	public:
		// constructor 
		IntCls(); 
		IntCls(int); // named constructor 
		~IntCls();	// destructor
		IntCls(const IntCls &); 	// copy constructor
		IntCls &operator=(const IntCls &); // copy assignment
		IntCls(IntCls &&); // move constructor
		IntCls &operator=(IntCls &&); // move assignment
		
		int getn() const;  
};

// constructor 
IntCls::IntCls() = default; 

// named constructor
IntCls::IntCls(int n) : n(n) {} 

// destructor
IntCls::~IntCls() = default; 

// copy constructor
IntCls::IntCls(const IntCls &old_obj) {
	n = old_obj.n; 
}		
		

int main() 
{
	IntCls intclsobj;
	IntCls intclsobj1(1);
	
	IntCls *intclsobj2 = new IntCls(2); 
	delete intclsobj2;

	std::cout << sizeof(intclsobj) << std::endl; 	// 16 bytes, 128 bits
	std::cout << sizeof(intclsobj1) << std::endl; 	// 16 bytes, 128 bits 
	std::cout << sizeof(intclsobj2) << std::endl; 	// 8 bytes, 64 bits  


	delete intclsobj2;  
	return 0;
}
