/**
 * 	@file 	copyctor.cpp
 * 	@brief 	copy constructor of class T is a non-template constructor whose 1st parameter is T&, const T&, volatile T&, const volatile T&
 * 	@ref	http://en.cppreference.com/w/cpp/language/copy_constructor 
 * 	@details copy constructor called whenever object is initialized (by direct-initialization or copy-initialization) 
 * from another object os same type (unless overload resolution), which includes  
 * initialization 
 * function argument passing
 * function return 
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g virtualfunc.cpp -o virtualfunc
 * */

#include <iostream>

struct A
{
	int n;
	// constructor
	A(int n = 1) : n(n) { } 
	A(const A& a): n(a.n) { } // user-defined copy ctor
};

struct B : A
{
	// implicit default ctor B::B()
	// implicit copy ctor B::B(const B&)  
};

struct C : B
{
	C() : B() { } 
	private:
		C(const C&); // non-copyable, C++98 style  
		
};

int main()
{
	A a1(7); 
	A a2(a1); 	// calls the copy ctor
	B b;
	B b2 = b;
	A a3 = b; 	// conversion to A& and copy ctor
	volatile A va(10);
	// A a4 = va; // compile error 
	
	C c;
	// C c2 = c; // compile error
}

