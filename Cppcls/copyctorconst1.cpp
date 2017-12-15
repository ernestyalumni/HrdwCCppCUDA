/**
 * 	@file 	copyctorconst1.cpp
 * 	@brief 	Why copy constructor argument should be const in C++?  pass object by reference, as a const ref. 
 * 	@ref	http://www.geeksforgeeks.org/copy-constructor-argument-const/  
 * 	@details Why copy constructor argument should be const in C++?  
 * 	When we create our own copy constructor, we pass object by reference and pass it as const reference;     
 * 1. we should use const in C++ wherever possible so objects aren't accidentally modified  
 * 2. fun() returns by value, so compiler creates temporary objct which is copied to t2 using copy constructor 
 * (because this temporary object is passed as argument to copy constructor since compiler generates temp. object) 
 * Compiler error is compiler created temporary objects cannot be bound to non-const references   
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g copyctor1.cpp -o copyctor1
 * */
#include <iostream>  

class Test
{
	/* Class data members */
	public:
		Test(const Test &t) 	{ /* Copy data members from t */ } 
		Test()			{ /* Initialize data members */ }
};

Test fun() 
{
	std::cout << "fun() Called\n";
	Test t;
	return t;
};

int main()
{
	Test t1;
	Test t2 = fun(); // error: invalid initialization of non-const reference of type ‘Test&’ from an rvalue of type ‘Test’

		
	
	return 0;
}
