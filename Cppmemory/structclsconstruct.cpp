/**
 * 	@file 	structclsconstruct.cpp
 * 	@brief 	string object life cycle example      
 * 	@ref Stroustrup. The C++ Programming Language, 4th Ed. 17.2. Constructors and Destructors  
 * 		17.2. Constructors and Invariants	
 * 	@details std::clog - standard output stream for logging
 * 		class invariant concept at class Vector
 * 	
 * 	COMPILATION TIP
 * 	-Wall all warnings -g debugger
 * 	g++ -Wall -g objlifecycle.cpp -o objlifecycle
 * 	gdb ./objlifecycle # go into gdb, and then break main , run, bt, next, next, ... print a, print b, ... x arr, x ptr, ...
 * */
#include <string>  // std::string, std::to_string
#include <iostream> // std::clog
#include <vector> // std::vector

/* *************** 17.2 Constructors and Destructors *************** */
struct Tracer { 
	std::string mess;
	Tracer(const std::string& s) : mess(s) { std::clog << mess; } 	// constructor 
	~Tracer() { std::clog << "~" << mess; }	// define destructor to ensure "cleanp" at point of destruction of an object 
};  

void f(const std::vector<int>& v)
{
	Tracer tr {"In f()\n"};
	for (auto x: v) {
		Tracer tr {std::string{"v loop "}+ std::to_string(x) + '\n'};	
	}
}

/* *************** 17.2.1. Constructors and Invariants *************** */

/** @struct S
 * 	@details constructor declaration specifies argument list (exactly as for a function), but has no return type.  
 * 	Name of class can't be used for ordinary member function, data member, member type, etc., within the class
 * */
struct S {
	S();	// fine
/*	void S(int);	// error: no type can be specified for a constructor
	int S;			// error: no type can be specified for a constructor
	enum S {foo, bar};  // error: the class name must denote a constructor  
	*/
};

/* *************** 17.2.1. Constructors and Invariants and *************** */
/* *************** 17.2.2. Destructors and Resources *************** */

/** @class Vector
 * 	@details class invariant - something that must hold whenever a member function is called (from outside the class) 
 *  Often the case, invariant stated as comments: the constructor must make that true 
 * */
class Vector {
	public:
		Vector(int s);
		~Vector() { delete[] elem; }		// destructor: release memory 
	private:
		double* elem; 	// elem points to an array of sz doubles
		int sz; 		// sz is non-negative
};

// constructor must make class invariant true
Vector::Vector(int s)
{
//	if (s<0) throw Bad_size(s);  
	if (s<0) throw;
	sz = s;
	elem = new double[s];
}

/**
 * 	@fn fVec
 * 	@details v1 destroyed upon exit from fVec; also the Vector created on the free store by fVec()
 * using new is destroyed by the call of delete  


int main() {
	f({2,3,5});	// logging stream
	
	Vector test_Vec = Vector(5);

}
