/**
 * 	@file 	vtable.cpp
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

class B1 {
	public:
		virtual ~B1() {}
		void f0() {}
		virtual void f1() {}
		int int_in_b1;
};  

class B2 {
	public:
		virtual ~B2() {}
		virtual void f2() {}
		int int_in_b2;
};

class D : public B1, public B2 {
	public:
		void d() {}
		void f2() {}  // override B2::f2()
		int int_in_d;
}; 

int main() {
	B2 *b2 = new B2(); 
	D *d   = new D(); 
	
//	B1 *b1 = d;
//	B2 *b22 = d; 
	
//	delete b2;
//	delete d; 
//	return 0;
}
		

