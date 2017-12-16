/**
 * 	@file 	autotype8.cpp
 * 	@brief 	C++11/14/17 program to demonstrate auto type of 8   
 * 	@ref	https://stackoverflow.com/questions/38060436/what-are-the-new-features-in-c17
 * 	@details http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3922.html
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g autotype8.cpp -o autotype8
 * */
#include <iostream>  

#include <typeinfo> // typeid

int main() {

	auto x{8}; 
	
	auto x1 = { 1, 2}; // decltype(x1) is std::initializer_list<int> 
	
	auto x4 = { 3 }; // decltype(x4) is std::initializer_list<int> 
	
	std::cout << typeid( decltype( x )).name() << std::endl;  // i

	std::cout << typeid( decltype( x1 )).name() << std::endl;  // St16initializer_listIiE

	std::cout << typeid( decltype( x4 )).name() << std::endl;  // St16initializer_listIiE

}
