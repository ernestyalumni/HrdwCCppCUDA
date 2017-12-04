/**
 * 	@file 	virtualfunc.c
 * 	@brief 	write virtual function in C++, then use same to achieve dynamic or runtime polymorphism  
 * 	@ref	http://web.archive.org/web/20100209040010/http://www.codersource.net/published/view/325/virtual_functions_in.aspx
 * 	@details In this example, the pointer is of type base but it points to the derived class object.  
 * 	The method display() is virtual in nature.  
 * 	Hence in order to resolve the virtual method call, the context of the pointer is considered, 
 * 	i.e., the display method of the derived class is called and not the base.  
 *  If the method was non-virtual in nature, the display() method of the base class would have been called.  
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g virtualfunc.cpp -o virtualfunc
 * */
#include <iostream>

class base
{
	public:
		virtual void display() 
		{
			std::cout << "\n Base " << std::endl; 
		}
};

class derived : public base 
{
	public:
		void display()
		{
			std::cout << "\n Derived" << std::endl; 
		}
};

int main() {
	base *ptr = new derived(); 
	ptr->display();  // Derived  
	
	base instbase ; // instbase - instance of class base
	instbase.display();  	// Base 
	derived instder ; // instder - instance of class derived
	instder.display();	// Derived  

	base *ptrbase = new base(); // ptrbase - pointer to a class base object 
	ptrbase->display(); // Base
	
	derived *ptrder = new derived(); // ptrder - pointer to a class derived object 
	ptrder->display();  // Derived
	
}
