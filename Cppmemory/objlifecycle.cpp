/**
 * 	@file 	objlifecycle.cpp
 * 	@brief 	string object life cycle example      
 * 	@ref Stroustrup. The C++ Programming Language, 4th Ed. 17.1. Introduction	
 * 	@details 
 * 	
 * 	COMPILATION TIP
 * 	-Wall all warnings -g debugger
 * 	g++ -Wall -g objlifecycle.cpp -o objlifecycle
 * 	gdb ./objlifecycle # go into gdb, and then break main , run, bt, next, next, ... print a, print b, ... x arr, x ptr, ...
 * */
#include <string>  // std::string

std::string ident(std::string arg)  // string passed by value (copied into arg)
{
    return arg; // return string (move the value of arg out of ident() to a caller)
}

int main() 
{
    std::string s1("Adams");    // initialize string (construct in s1). 
    s1 = ident(s1);     // copy s1 into ident()
                        // move the result of ident(s1) into s1;
                        // s1's value is "Adams". 
    std::string s2("Pratchett");    // initialize string (construct in s2) 
    s1 = s2;            // copy the value of s2 into s1 
                        // both s1 and s2 have the value "Pratchett"                         
}