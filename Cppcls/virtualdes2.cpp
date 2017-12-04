/**
 * 	@file 	virtualdes2.cpp
 * 	@brief 	write virtual destructor in C++, now using the virtual keyword
 * 	@ref	http://web.archive.org/web/20100209040010/http://www.codersource.net/published/view/325/virtual_functions_in.aspx
 * 	@details Make destructor virtual in the base class.  
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g virtualdes2.cpp -o virtualdes2
 * */
#include <iostream>

class base
{
	public:
		virtual ~base() 
	{
		std::cout << " Destructing from the base class " << std::endl; 
	}
};

class derived : public base {
	public:
		~derived() // override does the same
	{ 
		std::cout << "Destructing from the derived inherited class " << std::endl;
	}
};

int main() 
{
	base *ptr = new derived();
	// some code
	std::cout << std::endl << " Prior to executing delete ptr, with base *ptr = new derived() " << std::endl << std::endl;
	delete ptr; // Destructing from the derived inherited class // Destructing from the base class 

	std::cout << std::endl << " deleted ptr.  Now doing new for a pointer to derived class instance. " << std::endl << std::endl ; 
	derived *ptrder = new derived(); // ptrder- pointer to a derived  
	delete ptrder;  
	
	base instbase; 
	std::cout << std::endl << " Declared an object instbase of class base.  Now declaring an object instder of class derived. " 
		<< std::endl << std::endl; 
	derived instder;
	std::cout << " Finished declaring an object instder of class derived. " << std::endl; 
}
