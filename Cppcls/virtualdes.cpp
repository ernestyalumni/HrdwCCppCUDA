/**
 * 	@file 	virtualdes.cpp
 * 	@brief 	write virtual destructor in C++
 * 	@ref	http://web.archive.org/web/20100209040010/http://www.codersource.net/published/view/325/virtual_functions_in.aspx
 * 	@details In this case, type of the pointer would be considered.  
 * 	Hence, as pointer is of type base, base class destructor would be called but the derived class destructor wouldn't be called at all.  
 *  Result could be memory leak.  
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g virtualdes.cpp -o virtualdes
 * */
#include <iostream>

class base
{
	public:
		~base() {
		std::cout << " Destructing from the base class " << std::endl; 
	}
};

class derived : public base {
	public:
		~derived()
	{ 
		std::cout << "Destructing from the derived inherited class " << std::endl;
	}
};

int main() 
{
	base *ptr = new derived();
	// some code
	std::cout << " Prior to executing delete ptr, with base *ptr = new derived() " << std::endl;
	delete ptr; 
}
