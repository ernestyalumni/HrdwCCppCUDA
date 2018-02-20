/**
 * 	@file 	unordered.cpp
 * 	@brief 	C++14 program to demonstrate unordered
 * 	@ref	Ch. 31 *STL Containers*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.
 * 	@details 
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g unordered.cpp -o unordered
 * */
#include <map>
#include <unordered_map>

#include <string>
#include <iostream>
#include <utility>

/*template<typename X, typename Y>
std::ostream& operator<<(std::ostream& os, std::pair<X,Y>& p)
{
	return os << '{' << p.first << ',' << p.second << '}';
};
*/
/*
template<typename X, typename Y>
std::ofstream& operator<<(std::ofstream& os, std::pair<X,Y>& p)
{
	return os << '{' << p.first << ',' << p.second << '}';
};
*/

std::ostream& operator<<(std::ostream& os, std::pair<std::string,int> p)
{
	return os << '{' << p.first << ',' << p.second << '}';
};

/*
inline std::ostream& operator<<(std::ostream& os, std::pair<std::string,int>& p)
{
	return os << '{' << p.first << ',' << p.second << '}';
};
*/

int main()
{

	std::unordered_map<std::string, int> score1 { 
		{"andy",7},{"al",9},{"bill",-3},{"barbara",12}
	};
	
	std::map<std::string,int> score2 {
		{"andy",7},{"al",9},{"bill",-3},{"barbara",12}
	};

	std::cout << "unordered: " ; 
	for (const auto& x: score1) 
	{
//		std::cout << x << ", ";
	}
	
	std::cout << "\nordered: "; 
	for (const auto& x : score2)
	{
//		std::cout << &x << ", ";
//		std::cout << x.first << ", ";
		std::cout << x << ", ";
	}
		std::cout << std::endl;
	
	
}
