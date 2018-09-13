/**
 * @file   : allocator.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : delete before a throw or leak
 * @ref    : http://en.cppreference.com/w/cpp/memory/allocator
 * 	13.6.1 A Simple vector Ch. 13 Exception Handling; 
 *  Bjarne Stroustrup, The C++ Programming Language, 4th Ed., Stroustrup  
 * @detail std::allocator class template is default Allocator if no 
 * 	user-specified allocator is provided. 
 *	Default allocator is stateless, i.e. all instances of the given allocator 
 *	are interchangeable, compare equal and can deallocate memory allocated by 
 * 	any other instance of same allocator type.
 *
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on physics, math, and engineering have 
 * helped students with their studies, and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 *  feel free to copy, edit, paste, make your own versions, share, use as you wish.    
 * Peace out, never give up! -EY
 * 
 * */
#include <memory>		// std::allocator, std::unique_ptr 
#include <array>
#include <iostream>
#include <string> 

int main()
{
	std::allocator<int> a1;			// default allocator for ints
	int* a = a1.allocate(1); 	// space for 1 int
	a1.construct(a, 7);					// construct the int
	std::cout << a[0] << '\n';
	a1.deallocate(a, 1);				// deallocate space for 1 int

	// default allocator for strings
	std::allocator<std::string> a2;

	// same, but obtained by rebinding from the type of a1
	decltype(a1)::rebind<std::string>::other a2_1;

	// same, but obtained by rebinding from the type of a1 via allocator_traits
	std::allocator_traits<decltype(a1)>::rebind_alloc<std::string> a2_2;

	std::string* s = a2.allocate(2);		// space for 2 strings

	a2.construct(s, "foo");
	a2.construct(s + 1, "bar");

	std::cout << s[0] << ' ' << s[1] << '\n';

	a2.destroy(s);
	a2.destroy(s + 1);
	a2.deallocate(s, 2);

  constexpr const unsigned long N {4};

  std::allocator<std::array<double, N>> allocates_4array;
  // allocate allocates uninitialized storage
  std::array<double, N>* x1 = allocates_4array.allocate(1); 

  x1[0] = {1., 2., 3., 4.};
  for (auto x1_i : x1[0]) { std::cout << x1_i << ' '; } std::cout << '\n';

  std::allocator<double> allocates_RR;
 	std::unique_ptr<double[]> y1(allocates_RR.allocate(1));

  y1[0] = 32.;
  std::cout << "y1[0] : " << y1[0] << '\n'; 	// 32.
  std::cout << "y1[1] : " << y1[1] << '\n';		// 0.

 	std::unique_ptr<double[]> y(allocates_RR.allocate(2));
  y[0] = 32.;
  y[1] = 64.;
  std::cout << "y[0] : " << y[0] << '\n'; 	// 32.
  std::cout << "y[1] : " << y[1] << '\n';		// 0.


  // AOS - array of structs?
}
