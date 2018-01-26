/**
 * @file   : stdforward.cpp
 * @brief  : std::forward C++11, examples     
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20180125
 * @ref    : https://stackoverflow.com/questions/3582001/advantages-of-using-forward
 * http://en.cppreference.com/w/cpp/utility/forward
 * @details : 
 * 
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on 
 * physics, math, and engineering have helped students with their studies, 
 * and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material 
 * open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.  
 *  Just don't be an asshole and not give credit where credit is due.  
 * Peace out, never give up! -EY
 * 
 * */
/* 
 * COMPILATION TIP
 * g++ -std=c++14 -pthread thread2.cpp -o thread2
 * 
 * */
#include <iostream>
#include <string>
#include <utility> // std::forward

#include <memory> // std::unique_ptr

// @ref https://stackoverflow.com/questions/3582001/advantages-of-using-forward
void overloaded_function(std::string& param) {
	std::cout << "std::string& version" << std::endl;
}

void overloaded_function(std::string&& param) {
	std::cout << "std::string&& version" << std::endl;
}

template<typename T>
void pass_through(T&& param) {
	overloaded_function(std::forward<T>(param) );
}

// @ref http://en.cppreference.com/w/cpp/utility/forward
// this example demonstrates perfect forwarding of the parameter(s) to the argument of the constructor of class T.
struct A { 
	A(int&& n) { std::cout << "rvalue overload, n=" << n << "\n"; }
	A(int& n) { std::cout << "lvalue overload, n=" << n << "\n"; }
}; 

class B {
	public:
		template<class T1, class T2, class T3> 
		B(T1&& t1, T2&& t2, T3&& t3) : 
			a1_{std::forward<T1>(t1)},
			a2_{std::forward<T2>(t2)},
			a3_{std::forward<T3>(t3)}
		{
		}
	
	private:
		A a1_, a2_, a3_;
}; 

// Also, perfect forwarding of parameter packs is demonstrated.  

template<class T, class U>
std::unique_ptr<T> make_unique1(U&& u)
{
	return std::unique_ptr<T>(new T(std::forward<U>(u)));
}

template<class T, class... U>
std::unique_ptr<T> make_unique(U&&... u)
{
	return std::unique_ptr<T>(new T(std::forward<U>(u)...));
}

	

int main() 
{
	// @ref https://stackoverflow.com/questions/3582001/advantages-of-using-forward
	std::string pes;
	pass_through(pes); // std::string& version
	pass_through(std::move(pes)); // std::string&& version
 
	// me trying stuff
	overloaded_function(pes);
	overloaded_function(std::move(pes));

	auto p1 = make_unique1<A>(2); // rvalue // rvalue overload, n = 2
	int i = 1;
	auto p2 = make_unique1<A>(i); // lvalue  // lvalue overload, n = 1
	
	std::cout << "B\n";
	auto t = make_unique<B>(2,i,3); 
	

	A test_a0 { 5 }; // rvalue overload n = 5
	int test_i1 { 7} ; 
	A test_a1 { test_i1 };  // lvalue overload
	std::unique_ptr<int> test_uniqptr_int2 = std::make_unique<int>( 8) ; 
	A test_a2 { *test_uniqptr_int2.get() }; // lvalue overload, n=8

	


}
