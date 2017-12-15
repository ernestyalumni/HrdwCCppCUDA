/**
 * 	@file 	copyctor1.cpp
 * 	@brief 	copy constructor is a member function which initializes an object using another object of the same class  
 * 	@ref	http://en.cppreference.com/w/cpp/language/copy_constructor 
 * http://www.geeksforgeeks.org/copy-constructor-in-cpp/
 * 	@details Simple examples of copy constructors.    
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g copyctor1.cpp -o copyctor1
 * */

#include <iostream>

/** 
 * @ref http://en.cppreference.com/w/cpp/language/copy_constructor   
 */
struct A
{
	int n;
	// constructor
	A(int n = 1) : n(n) { 
		std::cout << std::endl << " 'normal constructor' for struct A called " << std::endl; } 
	A(const A& a): n(a.n) { 
		std::cout << std::endl << " user-defined copy constructor for struct A called " << std::endl; } // user-defined copy ctor
};

struct B : A
{
	// implicit default ctor B::B()
	// implicit copy ctor B::B(const B&)  
};

struct C : B
{
	C() : B() { 
		std::cout << std::endl << " 'normal constructor' for struct C, inheriting from B, called " << std::endl; } 
	private:
		C(const C&); // non-copyable, C++98 style  
		
};

/**
 * @ref http://www.geeksforgeeks.org/copy-constructor-in-cpp/ 
 * @brief simple example of copy constructor 
 */

class Point
{
	private: 
		int x, y;
	public:
		// constructor 
		Point(int x1, int y1) {
			x = x1;
			y = y1;
		}
		
		// Copy constructor
		Point(const Point &p2) {
			x = p2.x;
			y = p2.y; }
		
		// getting member functions
		int getX() { return x; }
		int getY() { return y; }
};

int main()
{
	/** 
	 * @ref http://en.cppreference.com/w/cpp/language/copy_constructor   
	 */
	std::cout << std::endl << " A a1(7) initialization " << std::endl; 
	A a1(7); 
	std::cout << std::endl << " A a2(a1) initialization " << std::endl; 
	A a2(a1); 	// calls the copy ctor
	std::cout << std::endl << " B b initialization " << std::endl; 
	B b;
	std::cout << std::endl << " B b2 = b initialization " << std::endl; 
	B b2 = b;
	std::cout << std::endl << " A a3 = b initialization " << std::endl; 
	A a3 = b; 	// conversion to A& and copy ctor
	std::cout << std::endl << " volatile A va(10) initialization " << std::endl; 
	volatile A va(10);
	// A a4 = va; // compile error 
	std::cout << std::endl << " C c initialization " << std::endl; 
	C c;
	// C c2 = c; // compile error

	/**
	 * @ref http://www.geeksforgeeks.org/copy-constructor-in-cpp/ 
	 */
	Point p1(10, 15); // Normal constructor is called here 
	Point p2 = p1; 	// Copy constructor is called here 
	
	// Let us access values assigned by constructors 
	std::cout << "p1.x = " << p1.getX() << ", p1.y = " << p1.getY(); 
	std::cout << "\np2.x = " << p2.getX() << ", p2.y = " << p2.getY(); 

	return 0;
}

