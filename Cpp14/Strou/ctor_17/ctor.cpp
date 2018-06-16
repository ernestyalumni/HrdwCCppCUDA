/**
 * 	@file 	ctor.cpp
 * 	@brief 	C++14 program to demonstrate construction  
 * 	@ref	Construction, Cleanup, Copy, and Move Ch. 17 Bjarne Stroustrup, The C++ Programming Language, 4th Ed. 2013 Addison-Wesley 
 * 	@details 
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g ctor.cpp -o ctor
 * */

#include <string>

#include <iostream>
#include <vector>

#include <memory>
#include <stdexcept> 

// move constructor moving value of a `string` (from `arg` out of `identy()` into a temporary variable holding result of `ident(s1)`)
std::string ident(std::string arg) 	// string passed by value (copied into arg)
{
	return arg;		// return string (move the value of arg out of ident() to a caller
}

// error: call of overloaded 'ident(string&) is ambiguous
/*std::string ident (std::string& arg)
{
	std::cout << " Taking in a & ref, l-value & for arg: " << std::endl;
	return arg;
}
* */

// error: call of overloaded 'ident(string&&) is ambiguous
std::string ident (std::string&& arg)
{
	std::cout << " Taking in a && ref, r-value & for arg: " << " " << arg << 
		std::endl;
	return arg;
}

// error: call of overloaded 'ident(string&) is ambiguous
/*std::string ident (const std::string& arg)
{
	std::cout << " Taking in a & ref, l-value & for arg: " << std::endl;
	return arg;
}
*/


//------------------------------------------------------------------------------
// cf. 17.2 Constructors and Destructors

struct Tracer {
	std::string mess;
	Tracer(const std::string& s) : mess{s}
	{
		std::clog << mess; 
	}
	~Tracer() 
	{
		std::clog << "~" << mess; 
	}
};

void f(const std::vector<int>& v) 
{
	Tracer tr {"in f()\n"}; 
	for (auto x: v) 
	{
		Tracer tr { std::string { "v loop " } + std::to_string(x) + '\n' }; 
	}
	
}

class f_uniq_RAII
{
	public:
		f_uniq_RAII(int n);
	private:
		std::unique_ptr<float[]> elem; // elem points to an array of n float
		int n;							// n is non-negative
};

f_uniq_RAII::f_uniq_RAII(int n)
{
	if (n <0) 
	{
		throw std::runtime_error("bad size ");
	}
	n = n;
	elem = std::make_unique<float[]>(n);
}

int main()
{
	//  a constructor initializing a string with string literal 
	std::string s1 {"Adams"}; 	// initialize string (construct in s1)  

	//	copy constructor copying a `string` (into function argument `arg`)
	s1 = ident(s1); 			// copy s1 into ident()
								// move the result of ident(s1) into s1;
								// s1's value is "Adams".

	//  a constructor initializing a string with string literal 
	std::string s2 {"Pratchett"}; 	// initialize string (construct in s2)
	// copy assignment copying a `string` (from `s2` into `s1`) (l-value to l-value)
	s1 = s2; 						// copy the value of s2 into s1
									// both s1 and s2 have the value "Pratchett".
									
	std::cout << "s1 == s2 " << (s1 == s2) << std::endl; 
	
	
	ident( s1 ) ; 
//	ident( " str literal " ); // error call of overloaded ambiguous

	f({2,3,5});

	f_uniq_RAII(3);
//	f_uniq_RAII(-3); // terminate called after throwing an instance of 'std::runtime_error` what(): bad size Aborted (core dumped)

	
}
