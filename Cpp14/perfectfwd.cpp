/**
 * @file   : perfectfwd.cpp
 * @brief  : perfect forwarding C++11, examples, i.e.    
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20180124
 * @ref    : http://en.cppreference.com/w/cpp/language/parameter_pack
 * https://eli.thegreenplace.net/2014/variadic-templates-in-c/
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
#include <vector> // std::vector

#include <algorithm> // std::fill

class MyKlass {
	public:
		MyKlass(int ii_, float ff_) : m_ii_{ii_}, m_ff_{ff_}
			{}

	private:
		int m_ii_;
		float m_ff_;
};

void some_function (){ 
	std::vector<MyKlass> v;
	
	v.push_back(MyKlass(2,3.14f)); 
	v.emplace_back(2,3.14f);
}

// /ref https://agop.me/post/perfect-forwarding-cpp-11.html

class Foo
{
	public:
		std::string member;
		
		template<typename T>
		Foo(T&& member): member{std::forward<T>(member)} {}
}; 
		
class Foob
{
	public:
		std::string member;
		
		template<typename T>
		Foob(T&& member): member{std::forward<T>(member)} {
			std::cout << member << std::endl; }
}; 

class Foo2
{
	public:
		std::string member;
		std::string member2;
		
		template<
			typename T, 	// Parameter 1
			typename U, 	// Parameter 2,
			// Template type checking,
			typename = typename std::enable_if< // condition to check
				// Check type of parameter 1.
				std::is_constructible<std::string, T>::value &&
				// Check type of parameter 2. 
				std::is_constructible<std::string, U>::value>::type>				
		Foo2(T&& member, U&& member2): 
			member{std::forward<T>(member)},
			member2{std::forward<U>(member2)}
		{
			std::cout << member << " " << member2 << std::endl; 
			
			}
};


int main(int argc, char* argv[]) 
{
	some_function(); 
	
	// "boilerplate" test values
	constexpr const unsigned int N_fvec { 32 }; 
	std::vector<float> fvec( N_fvec );
	std::fill(fvec.begin(), fvec.end(), 4.f) ; 
	
	
	// sanity check
	for (auto ele : fvec) { std::cout << ele << " " ; } std::cout << std::endl; 
	
	/** @fn .emplace  
	 *  @details inserts new element into container directly before pos. 
	 * element is constructed through std::allocator_traits::construct, 
	 * which typically uses placement-new to construct the element in-place 
	 * at a location provided by the container.  Arguments args... are forwarded to constructor as 
	 * std::forward<Args>(args)... 
	 */
	fvec.emplace( fvec.begin() + 4, 5.f ); 
	for (auto ele : fvec) { std::cout << ele << " " ; } std::cout << std::endl; 

	// "boilerplate" test values
	std::string test_str = "Hello world."; 
	std::string& test_str_ref1 = test_str; 
	std::string&& test_str_ref2 = "Hello hello world.";

	std::cout << "Testing perfect forwarding : " << std::endl; 
	Foob test_foob { test_str };
	Foob test_foob1 { test_str_ref1 }; 
	Foob test_foob2 {test_str_ref2 };
	Foob test_foob3 { "Fboo 4" };
	
	Foo2 test_foo2 { test_str, test_str_ref1 };  
	Foo2 test_foo2_1(test_str_ref2, "Fboo 5" );  

	
}
